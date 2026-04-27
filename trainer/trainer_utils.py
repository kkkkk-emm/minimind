"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """
    输入：
        model：已经实例化的 MiniMindForCausalLM 或兼容的 nn.Module。
        config：模型配置对象，至少包含 MoE 相关字段或 dense 模型默认字段。
    输出：
        无显式返回值；仅通过 Logger 打印总参数量和 MoE 激活参数量。
    作用：
        统计模型参数规模。MoE 模型会区分“总参数量”和每个 token 实际激活的参数量，dense 模型只打印总参数量。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    输入：
        无。
    输出：
        bool：未初始化分布式时返回 True；分布式环境中只有 rank 0 返回 True。
    作用：
        判断当前进程是否负责打印日志、保存 checkpoint 等只应执行一次的操作。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    输入：
        content：要打印到控制台的日志内容。
    输出：
        无显式返回值。
    作用：
        只在主进程打印日志，避免分布式训练时多个 rank 重复输出同一条信息。
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    输入：
        current_step：当前全局训练 step，用于定位在总训练进度中的位置。
        total_steps：计划训练的总 step 数，通常为 epochs * 每个 epoch 的 step 数。
        lr：初始学习率，也是余弦衰减的基准值。
    输出：
        float：当前 step 应使用的学习率。
    作用：
        使用带 10% 下限的余弦退火策略调整学习率，让训练后期学习率平滑降低但不降为 0。
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    输入：
        无显式参数；通过环境变量 RANK、LOCAL_RANK 判断是否由 torchrun 启动。
    输出：
        int：当前进程绑定的 local_rank；非分布式模式固定返回 0。
    作用：
        初始化 NCCL 分布式进程组，并把当前进程绑定到对应 GPU；普通单进程训练则直接跳过。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    输入：
        seed：随机种子整数，会同步设置 Python、NumPy、PyTorch CPU/CUDA 的随机状态。
    输出：
        无显式返回值。
    作用：
        尽量固定训练过程中的随机性，便于复现实验；同时关闭 cuDNN benchmark 以减少非确定性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    输入：
        lm_config：MiniMindConfig，提供 hidden_size 和 use_moe，用于拼接 checkpoint 文件名。
        weight：权重文件名前缀，例如 pretrain、full_sft。
        model：可选模型对象；传入时进入保存模式，不传时进入加载模式。
        optimizer：保存模式下的优化器对象，用于记录 AdamW 等优化器状态。
        epoch：当前训练到的 epoch，恢复训练时从这里继续。
        step：当前 epoch 内已经完成的 step，恢复训练时用于跳过已训练 batch。
        wandb：可选实验日志对象，用于保存 run id，方便断点续写同一条日志。
        save_dir：checkpoint 保存目录。
        **kwargs：额外需要保存的状态对象，例如 GradScaler；有 state_dict 的对象会保存其 state_dict。
    输出：
        保存模式：无显式返回值。
        加载模式：若找到 resume checkpoint，返回包含模型、优化器、epoch、step 等状态的 dict；否则返回 None。
    作用：
        统一处理预训练/微调的 checkpoint 保存与恢复。保存时同时写轻量推理权重和可续训状态；加载时自动适配 GPU 数量变化后的 step。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    输入：
        lm_config：MiniMindConfig，决定模型层数、hidden_size、MoE 等结构。
        from_weight：要加载的权重前缀；为 'none' 时从随机初始化开始训练。
        tokenizer_path：tokenizer 文件目录，默认使用项目内 model 目录。
        save_dir：已有权重所在目录。
        device：模型加载和最终放置的设备，例如 cuda:0 或 cpu。
    输出：
        (model, tokenizer)：已移动到目标 device 的 MiniMindForCausalLM，以及对应 tokenizer。
    作用：
        初始化预训练所需的 tokenizer 和模型；如指定 from_weight，则加载已有权重继续训练或迁移训练。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        输入：
            sampler：样本索引来源，可以是普通索引列表，也可以是 DistributedSampler。
            batch_size：每个 batch 包含的样本数。
            skip_batches：迭代开始时跳过的完整 batch 数，用于断点恢复。
        输出：
            无显式返回值；初始化 sampler、batch_size、skip_batches 三个属性。
        作用：
            构造一个支持“从第 N 个 batch 继续”的 batch sampler，供 DataLoader 使用。
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        输入：
            无显式参数；迭代 self.sampler 产生的样本索引。
        输出：
            逐个 yield batch 索引列表，每个列表长度最多为 batch_size。
        作用：
            把样本索引聚合成 batch，并在开头丢弃 skip_batches 个完整 batch，实现断点续训跳步。
        """
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        输入：
            无。
        输出：
            int：跳过指定 batch 后，当前 epoch 还会产生的 batch 数。
        作用：
            让 DataLoader 和训练日志能知道恢复后的剩余迭代次数。
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
