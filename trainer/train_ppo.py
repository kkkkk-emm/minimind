import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    输入：
        text：生成的回复文本，字符串；无张量维度。
        n：标量整数 n，表示 n-gram 的长度，默认 3；用于检测重复短语。
        cap：标量浮点数 cap，无张量维度；惩罚分数的上限，默认 0.5。
    输出：
        float：重复率分数，范围 [0, cap]；数值越大表示重复内容越多。
    作用：
        计算 n-gram 重复率，用于奖励函数中惩罚模型生成重复内容，促进多样性生成。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        """
        输入：
            params：MiniMindConfig 配置对象，无 batch 维度；决定隐藏层维度 H 和网络深度。
        输出：
            无显式返回值；创建基础 MiniMind 模型，并将最后的语言建模头替换为单值输出头。
        作用：
            初始化 Critic 模型用于价值估计。继承 MiniMindForCausalLM 的编码器和层数，
            但用 nn.Linear(H, 1) 替代 lm_head，使模型为每个位置输出一个标量价值估计。
        """
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        输入：
            input_ids：token id 张量，形状 [B, T]；B=批大小，T=序列长度。
            attention_mask：None 或形状 [B, T] 的张量；1 表示有效，0 表示 padding。
            **kwargs：其他传向基础模型的参数。
        输出：
            torch.Tensor：shape 为 [B, T]；每个位置的价值估计标量值。
        作用：
            通过基础 MiniMind 编码器获取隐藏状态，在最后一层归一化后通过 value_head
            输出每个位置的价值估计，用于 PPO 的优势计算和价值函数损失。
        """
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """
    输入：
        prompts：对话 prompt 列表，每个元素为字符串；长度 B，B=批大小。
        responses：模型生成的回复列表，每个元素为字符串；长度 B，对应 prompts 的回答。
        reward_model：LMForRewardModel 对象，负责评估回复质量的外部奖励模型。
    输出：
        torch.Tensor：奖励张量，形状 [B]；每个位置为对应回复的综合奖励分数。
    作用：
        根据多个维度计算每条回复的奖励：
        1. 长度奖励：20-800 字符内为正，否则为负，鼓励适当长度回答。
        2. 思考过程奖励：如果回复中有 thinking 标签，奖励其长度和完整性。
        3. 重复惩罚：使用 n-gram 检测降低重复回复的奖励。
        4. 模型奖励：通过外部奖励模型评分后添加到总奖励。
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从 im_start/im_end 标签中提取对话内容，解析为消息列表
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            
            # 长度奖励：鼓励 20-800 字符的回复
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            
            # 思考过程奖励（如果存在 thinking 标签）
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                # 思考内容长度在 20-300 字符范围内为正
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                # 思考标签完整性（仅一个 </think> 为正）
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            
            # n-gram 重复惩罚
            rewards[i] -= rep_penalty(answer)

            # 使用外部奖励模型评分
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 转换奖励模型分数为张量，并添加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    """
    输入：
        epoch：标量整数，当前训练的 epoch 索引（0-indexed）。
        loader：DataLoader 对象，每次迭代返回包含 prompt 的 batch 字典。
        iters：标量整数，当前 epoch 的总 batch 数，用于日志和学习率衰减。
        rollout_engine：RolloutEngine 对象（TorchRolloutEngine 或 SGLangRolloutEngine），
            负责使用策略模型生成回复并计算 log 概率。
        ref_model：参考模型（MiniMindForCausalLM），用于计算 KL 散度惩罚，权重冻结。
        actor_scheduler：Actor 模型优化器的学习率调度器（CosineAnnealingLR）。
        critic_scheduler：Critic 模型优化器的学习率调度器（CosineAnnealingLR）。
        reward_model：LMForRewardModel 对象，用于评估生成回复的质量。
        start_step：标量整数 >= 0，本 epoch 从第几个 step 开始；用于断点续训时跳过已完成的 batch。
        wandb：可选的日志记录对象（SwanLab），每个 step 记录一次损失、奖励等指标。
        use_sglang：bool，指示是否使用 SGLang 推理引擎加速。
    输出：
        无显式返回值。
    作用：
        执行一个完整 PPO epoch 的训练循环：
        1. 从 prompt 生成回复，计算奖励（长度、思考过程、重复、模型评分）。
        2. 计算优势估计（GAE）和回报（returns）。
        3. 多次迭代更新 Actor 和 Critic 模型：
           - 前向计算：Actor 的策略 loss（PPO 裁剪损失 + KL 惩罚）和 Critic 的价值 loss。
           - 梯度累积和参数更新。
           - 在梯度积累达到指定步数时执行优化器步骤。
        4. 定期保存 checkpoint 和日志。
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        # ========== Rollout 阶段：使用 Actor 生成回复 ==========
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids  # [B, P+R]，P=prompt_length, R=response_length
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions  # list[str]，模型生成的文本回复
        
        # 计算每条回复的多维奖励
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        # resp_policy_mask: [B, R]，1 表示 token 被认为是策略生成的有效 token（用于 policy loss）
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        # resp_value_mask: [B, R]，同 policy_mask（用于 value loss）
        resp_value_mask = resp_policy_mask.clone()

        # ========== 计算参考模型 logp 和 Critic 价值 ==========
        with torch.no_grad():  # Rollout 阶段只需推理获取 old_logp 和 old_values，切断梯度省显存
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
            
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]，每个样本最后一个有效 token 的位置
            # 广播方式加奖励：batch 维和 token 维同时索引
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            # GAE 计算：从末尾往前递归计算优势
            gen_len = old_resp_values.size(1)  # R，回复长度
            lastgaelam = torch.zeros(B, device=args.device)  # [B]，前一个 GAE 项的值
            advs_rev = []  # 倒序存储的优势列表
            for t in reversed(range(gen_len)):  # 从 R-1 反向到 0
                # nv: 下一步的价值（t > 0 时从缓存读，否则为 0）
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # TD error: reward + gamma * V(t+1) - V(t)
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                # lastgaelam = delta + gamma * lambda * lastgaelam
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            # 反向重构优势，使得 advantages[t] 已经包含 t..T 的累积衰减项
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            # 回报 = 优势 + 价值
            returns = advantages + old_resp_values  # [B, R]

            # ========== 优势归一化 ==========
            # 计算优势的均值和方差（仅在有效位置上）
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            # 标准化优势：(adv - mean) / sqrt(var + eps)；然后乘以 mask 以忽略 padding 位置
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        
        # 获取原始模型（不是 DDP 包装）用于推理
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        
        # ========== PPO mini-batch 迭代更新 ==========
        for ppo_epoch in range(args.ppo_update_iters):  # 通常为 2-4 次
            if stop_ppo:
                break
            # 随机打乱样本索引，用于每次迭代都有不同的 mini-batch 组合
            b_inds = torch.randperm(B, device=args.device)
            
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]  # 当前 mini-batch 的样本索引
                
                # ========== 前向传播：Critic 部分 ==========
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                # ========== 前向传播：Actor 部分 ==========
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    # aux_loss：通常是 MoE 路由的辅助损失，非 MoE 模型返回 0
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                # 计算新的策略 logp（用于比率计算）
                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])  # [mb_size, R]
                
                # ========== 计算 PPO 裁剪相关项 ==========
                # log_ratio = log(pi_new/pi_old) = logp_new - logp_old
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                # 近似 KL 散度：0.5 * (log_ratio)^2（一阶泰勒展开）
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                # 早停条件：如果 KL 散度过大，停止优化以防策略偏离参考模型太远
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                # ========== 计算损失 ==========
                # ratio = exp(log_ratio) = pi_new / pi_old
                ratio = torch.exp(log_ratio)
                # clip_frac：超出 (1-eps, 1+eps) 范围的 token 比例
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                
                # ========== KL 散度惩罚项 ==========
                # kl_ref_penalty = E[exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1]
                # 这是 KL 散度的一种近似（reversed KL），鼓励新策略不离参考模型太远
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # ========== 策略损失（PPO 裁剪损失 + KL 惩罚） ==========
                # PPO 裁剪：max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))
                # 本质：当 adv > 0（好回复）时，我们想增大 pi_new/pi_old，但不超过 1+eps；
                #       当 adv < 0（差回复）时，我们想减小 pi_new/pi_old，但不低于 1-eps。
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)  # 加上 KL 惩罚项
                
                # ========== 价值损失（PPO 价值裁剪） ==========
                # 类似策略，对价值预测也做裁剪：防止 Critic 的更新过于激进
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                # 提取当前纪录（用于日志）
                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    # 总损失 = 策略损失 + vf_coef * 价值损失 + 辅助损失
                    # 除以 accumulation_steps 用于梯度累积
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                # 反向传播
                loss.backward()

                # 累积统计数据（用于日志）
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # 每累积 accumulation_steps 次梯度后执行一次参数更新
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        # ========== 更新 Rollout 引擎的策略模型权重 ==========
        if step % args.save_interval == 0 or step == iters: 
            rollout_engine.update_policy(actor_model)

        # ========== 日志记录 ==========
        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            # 记录到 wandb（如果启用）
            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            # 控制台日志
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ========== Checkpoint 保存 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取原始模型并提取 state_dict
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            # 转换为 float16 并移到 CPU 以节省存储空间
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）用于续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # init_distributed_mode 检测 RANK 环境变量；若不存在返回 0 表示非分布式，否则初始化 NCCL 后返回 local_rank
    local_rank = init_distributed_mode()
    # 根据 local_rank 把当前进程绑定到对应 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 为了增加多 GPU 训练的随机性，每个 rank 使用不同的种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 根据命令行参数创建模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 检查是否存在断点续训 checkpoint；若存在则加载，否则为 None
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bfloat16 更稳定但需要 Ampere 及以上 GPU；float16 兼容性更好但需要 loss scaling
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 创建自动混合精度上下文；CPU 上禁用 AMP
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb 日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 从 checkpoint 恢复已有的 wandb run ID，以继续同一条日志链
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # 加载 Actor 模型和 tokenizer
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 加载参考模型（用于 KL 散度计算），权重冻结
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # 加载基础权重用于初始化 Critic 模型
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    # 创建 Critic 模型（继承 MiniMind 编码器，输出单个价值估计）
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    
    # 加载外部奖励模型（用于多维奖励计算）
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    
    # 创建 Rollout 引擎（生成回复并计算对数概率）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,  # torch 或 sglang
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    
    # 加载 RLAIF 训练数据集
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    # 分布式情况下使用 DistributedSampler，单进程使用 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建 Actor 和 Critic 优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总训练 step 数，用于学习率衰减
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    # 总 step = epoch数 * batch数 * PPO更新次数 * mini_batch数 / 梯度累积步数
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    # 使用余弦退火学习率衰减
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        # 恢复优化器状态（梯度、动量等）
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        # 恢复学习率调度器状态
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        # 恢复训练起点
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # torch.compile 可加速但与某些 monkey-patch 不兼容
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    # 分布式数据并行化：把模型复制到每个 GPU
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式 sampler 的随机种子，确保每个 epoch 数据顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 为数据索引重新设置种子
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 计算本 epoch 需要跳过的 batch 数（用于恢复中断的训练）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 使用自定义 batch sampler，支持跳过前 N 个 batch
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0: 
            # 断点续训：打印恢复信息
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 执行 PPO 训练循环
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    # 销毁分布式进程组，释放相关资源
    if dist.is_initialized(): dist.destroy_process_group()