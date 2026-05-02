"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    输入：
        model：语言模型（可能被 DDP 包装）。
        input_ids：token id 张量，形状 [B, T]；B=批大小，T=序列长度。
        n_keep：标量整数，只计算序列末尾 n_keep 个 token 的 log 概率。
        attention_mask：None 或形状 [B, T] 的张量；1 表示有效位置，0 表示 padding。
    输出：
        torch.Tensor：形状 [B, n_keep]；每个 token 的对数概率。
    作用：
        调用模型前向推理得到 logits，使用 log_softmax 转为对数概率，
        然后 gather 出每个位置预测的 token 的概率值。
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    """
    Rollout 生成结果的数据容器，用于传递完整的生成过程信息。
    
    属性：
        output_ids：生成的完整序列（包括 prompt），形状 [B*num_gen, P+R]。
        completion_ids：只有生成部分的 token id，形状 [B*num_gen, R]；R=生成长度。
        per_token_logps：生成部分每个 token 的对数概率，形状 [B*num_gen, R]。
        completions：生成文本的解码字符串列表，长度 B*num_gen。
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    """
    Rollout 引擎抽象基类，定义生成和更新策略模型的统一接口。
    """
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """
        输入：
            prompt_ids：prompt token id，形状 [B, P]；B=批大小，P=prompt 长度。
            attention_mask：形状 [B, P]；1 表示有效 prompt，0 表示 padding。
            num_generations：标量整数，每个 prompt 生成几条回复。
            max_new_tokens：标量整数，每条回复最多生成多少个 token。
            temperature：标量浮点数，sampling 的温度参数。
        输出：
            RolloutResult：包含生成序列、token 对数概率、文本等信息。
        作用：
            使用策略模型生成文本，并计算每个 token 的对数概率用于后续 PPO 计算。
        """
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """
        输入：
            model：更新后的策略模型。
        输出：
            无显式返回值。
        作用：
            更新引擎内部保存的策略模型权重（本地引擎）或远程同步（SGLang 引擎）。
        """
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        """
        输入：
            policy_model：策略模型（MiniMindForCausalLM），用于生成文本。
            tokenizer：tokenizer 对象，用于 decode 和 padding token id。
            device：设备字符串，例如 'cuda:0'。
            autocast_ctx：自动混合精度上下文，例如 torch.cuda.amp.autocast(dtype=torch.bfloat16)。
        输出：
            无显式返回值；初始化成员变量。
        作用：
            初始化本地 PyTorch 推理引擎，直接调用模型的 generate 方法生成文本。
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """
        输入：
            prompt_ids：prompt token id，形状 [B, P]。
            attention_mask：形状 [B, P]。
            num_generations：每个 prompt 生成的文本数。
            max_new_tokens：最大生成长度。
            temperature：sampling 温度。
        输出：
            RolloutResult：包含完整序列、生成部分、对数概率、文本。
        作用：
            调用模型 generate 方法生成文本（num_generations 倍扩展），
            然后计算每个 token 的对数概率用于 PPO 的策略学习。
        """
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # generate 会扩展 batch 维度 num_generations 倍
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    def update_policy(self, model: torch.nn.Module):
        """
        输入：
            model：更新后的策略模型。
        输出：
            无显式返回值。
        作用：
            更新本地引擎保存的策略模型引用（仅改变对象指针）。
        """
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        """
        输入：
            base_url：SGLang 服务器 URL，例如 'http://localhost:8998'。
            model_path：模型（含 tokenizer）路径，SGLang 用此初始化。
            shared_ckpt_path：本地权重存储目录，用于与 SGLang 同步。
            timeout：HTTP 请求超时时间（秒）。
        输出：
            无显式返回值；初始化 tokenizer 和 HTTP 客户端。
        作用：
            初始化远程 SGLang 推理引擎，通过 HTTP API 与服务通信加快生成速度。
        """
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """
    输入：
        engine_type：str，'torch' 或 'sglang'；决定使用本地 PyTorch 推理还是远程 SGLang 服务。
        policy_model：当 engine_type='torch' 时需要传入的模型对象（MiniMindForCausalLM）。
        tokenizer：tokenizer 对象，用于 decode 生成的 token 序列。
        device：设备字符串，例如 'cuda:0'；torch 引擎使用此参数。
        autocast_ctx：自动混合精度上下文；torch 引擎使用此参数加速推理。
        sglang_base_url：当 engine_type='sglang' 时的服务器 URL，例如 'http://localhost:8998'。
        sglang_model_path：SGLang 使用的 tokenizer 路径。
        sglang_shared_path：SGLang 权重存储路径。
    输出：
        RolloutEngine：子类实例，要么 TorchRolloutEngine 要么 SGLangRolloutEngine。
    作用：
        根据 engine_type 参数创建合适的生成引擎。PyTorch 引擎用于单机推理，
        SGLang 引擎用于分布式/集中式服务加速。
    """
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
