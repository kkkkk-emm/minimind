import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    输入：
        epoch：标量整数，当前训练轮数（0-indexed）；用于计算学习率衰减中的全局 step。
        loader：数据加载器，每次迭代返回 (input_ids, labels)，
            input_ids 形状 [B, T, H]（B=batch_size, T=max_seq_len, H=hidden_size）；
            labels 形状 [B, T]，-100 位置表示 padding，不计入 loss。
        iters：标量整数，当前 epoch 中 batch 的总数，用于学习率衰减和日志判断。
        lora_params：list of torch.Tensor，所有 LoRA 层参数的列表，为这些参数计算梯度并更新。
        start_step：标量整数 >= 0，本 epoch 从第几个 step 开始；用于断点续训时跳过已完成的 batch。
        wandb：可选的日志记录对象，每 log_interval 步记录一次 loss 等指标。
    输出：
        无显式返回值。
    作用：
        执行一个完整 epoch 的训练循环：
        1. 按 start_step 跳过已完成的 batch
        2. 对每个 batch：前向计算 loss + aux_loss，梯度累积，定期更新梯度、裁剪范数、清空梯度
        3. 定期打印日志、保存 checkpoint、记录到 wandb
        4. 处理余数 batch 的梯度更新（确保最后一个不完整积累块的权重也得到更新）
    """
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # 根据全局 step 计算当前学习率，使用余弦退火衰减：lr * (0.1 + 0.45 * (1 + cos(π * step / total)))
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向计算与反向传播
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss  # 主任务 loss + 辅助 loss（通常是 MoE router 损失）
            loss = loss / args.accumulation_steps  # 梯度累积：平均化

        scaler.scale(loss).backward()  # 混合精度下的反向传播

        # 梯度累积：每 accumulation_steps 次反向传播后执行一次参数更新
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 混合精度：反缩放梯度准备 norm clipping
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)  # 梯度范数裁剪
            scaler.step(optimizer)  # 混合精度：缩放后执行优化器 step
            scaler.update()  # 更新 GradScaler 的缩放因子
            optimizer.zero_grad(set_to_none=True)  # 清空梯度，set_to_none=True 更节省内存

        # 定期打印日志与保存 checkpoint
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

    # 处理最后一个不完整的梯度累积块
    # 当 epoch 结束时，如果最后几个 step 尚未积累满 accumulation_steps 个，需要执行一次参数更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    # ===== 保存和 checkpoint 相关参数 =====
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    
    # ===== 训练超参数 =====
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    
    # ===== 设备和精度相关参数 =====
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    
    # ===== 优化和梯度相关参数 =====
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # ===== 日志和保存间隔 =====
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # ===== 模型结构参数 =====
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # ===== 数据和权重相关参数 =====
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # ===== 日志和编译相关参数 =====
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境，检测是否由 torchrun 启动；返回 local_rank，非分布式模式返回 0
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，确保复现性；分布式情况下每个 rank 使用不同的种子以增加多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 根据命令行参数创建模型配置；use_moe 决定是否使用 MoE 架构
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 检查是否存在断点续训 checkpoint；args.from_resume=1 时尝试加载，否则为 None 表示从头训练
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 根据设备类型和 dtype 参数选择自动混合精度（AMP）上下文管理器
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bfloat16 更稳定但需要 Ampere 及以上 GPU；float16 兼容性更好但需要 loss scaling
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb 日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 如果恢复训练，使用之前的 wandb run id 继续同一条日志链
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用 LoRA、冻结非 LoRA 参数 ==========
    # 初始化模型，从指定的预训练权重加载；如 from_weight='full_sft' 则加载微调后的权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 为所有方形 Linear 层添加 LoRA 适配器（LoRA rank 默认 16）
    apply_lora(model)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结原始模型参数，只训练 LoRA 层
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # 加载 SFT 数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式情况下使用 DistributedSampler，单进程模式下为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 混合精度的梯度缩放器；float16 需要，bfloat16 不需要但也可以用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 仅对 LoRA 参数进行优化
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从 checkpoint 恢复状态 ==========
    # 初始化训练起点；恢复训练时从 checkpoint 指定的 epoch/step 继续
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # 注意：LoRA 使用 monkey-patch forward，与 torch.compile 不兼容，自动关闭
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    if dist.is_initialized():
        # 分布式数据并行化：把模型复制到每个 GPU，梯度自动在 backward 时累加
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式情况下：设置随机种子确保 shuffle 后不同 rank 数据不重复
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 计算本 epoch 的跳过 batch 数；恢复训练时仅第一个 epoch 需要跳过
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 自定义 batch sampler，支持跳过前 N 个 batch（用于恢复中断训练）
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建 DataLoader；pin_memory=True 加速 CPU->GPU 传输
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 显示断点续训的跳过信息
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    # 销毁分布式进程组，释放相关资源
    if dist.is_initialized(): dist.destroy_process_group()