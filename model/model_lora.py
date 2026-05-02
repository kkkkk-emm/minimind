import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        """
        输入：
            in_features：标量整数，LoRA 输入维度，通常等于原始 Linear 层的 in_features。
            out_features：标量整数，LoRA 输出维度，通常等于原始 Linear 层的 out_features。
            rank：标量整数 R，LoRA 的秩，控制低秩矩阵 A 和 B 的中间维度；R 越小参数越少、计算越快。
        输出：
            无显式返回值；创建两个可学习的低秩矩阵。
        作用：
            为 Linear 层创建 LoRA 适配器，在保持原层冻结的前提下，通过 A（in_features→R）和 B（R→out_features）
            的低秩变换来微调模型，使总参数增量为 O((in_features + out_features) × R) 而非 O(in_features × out_features)。
        """
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A：[in_features, rank]
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B：[rank, out_features]
        # 矩阵A高斯初始化，std=0.02 是标准初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，训练初期 B 的输出为全 0，相当于 LoRA 适配器无作用，然后逐步学习差异
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        输入：
            x：形状 [B, ..., in_features] 的张量；B 是 batch size，中间维度任意，最后一维是输入特征维度。
        输出：
            torch.Tensor：形状 [B, ..., out_features]，通过两层低秩线性变换得到。
        作用：
            计算 LoRA 的低秩适配：先投影到秩维度 R，再投影回输出维度，得到形状与原始线性层输出相同的增量。
        """
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """
    输入：
        model：已实例化的 nn.Module 或 MiniMindForCausalLM，包含多个 nn.Linear 层待适配。
        rank：标量整数 R，所有创建的 LoRA 适配器的秩；默认 16。
    输出：
        无显式返回值；原地修改 model，为所有满足条件的 Linear 层绑定 LoRA 适配器。
    作用：
        遍历模型的所有 Linear 层，为方阵层（in_features == out_features）添加 LoRA 适配器，
        并通过 monkey-patch 修改其 forward 方法，使输出 = 原始输出 + LoRA 适配器输出。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定参数，避免闭包中 lora 被后续迭代改写
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                """
                输入：
                    x：形状 [B, ..., in_features] 的输入张量。
                    layer1：捕获原始 Linear 层的 forward 方法。
                    layer2：捕获当前模块的 LoRA 适配器实例。
                输出：
                    torch.Tensor：原始输出 + LoRA 适配增量，形状 [B, ..., out_features]。
                作用：
                    在模型前向传播中，每个被适配的 Linear 层都执行原始线性变换加上低秩适配，
                    相当于原参数 W 被替换为 W + BA，其中 B, A 是 LoRA 权重。
                """
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """
    输入：
        model：已通过 apply_lora() 添加了 LoRA 适配器的 nn.Module。
        path：字符串，保存有 LoRA 权重的 .pth 文件路径；
            字典键格式为 'module_name.lora.A.weight' / 'module_name.lora.B.weight'。
    输出：
        无显式返回值；原地加载 LoRA 权重到 model 中。
    作用：
        从 checkpoint 恢复 LoRA 适配器权重，支持分布式训练后的权重加载（自动处理 'module.' 前缀）。
    """
    state_dict = torch.load(path, map_location=model.device)
    # 移除 DistributedDataParallel 的 'module.' 前缀，以便加载到单进程模型中
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 为当前 Linear 层的 LoRA 适配器提取对应的权重，去掉全路径前缀只保留 'A.weight'/'B.weight'
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    输入：
        model：已应用 LoRA 适配器的 nn.Module；可能被 torch.compile 或 DistributedDataParallel 包装。
        path：字符串，要保存的 .pth 文件路径。
    输出：
        无显式返回值；将 LoRA 权重以 float16 精度存储到 path。
    作用：
        只保存 LoRA 部分的权重（不保存原始模型权重），可显著减小 checkpoint 文件大小。
        自动处理模型包装（torch.compile 的 _orig_mod、DDP 的 module 等），并转换为 CPU + float16 进行存储。
    """
    # 获取原始模型，处理 torch.compile 的包装
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 移除 DistributedDataParallel 的 'module.' 前缀
            clean_name = name[7:] if name.startswith("module.") else name
            # 收集该层的 LoRA 权重：A.weight 和 B.weight，转为 CPU + float16 节省空间
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    输入：
        model：已应用 LoRA 适配器的 nn.Module。
        lora_path：字符串，之前用 save_lora() 保存的 LoRA 权重文件路径。
        save_path：字符串，要保存合并后权重的目标 .pth 文件路径。
    输出：
        无显式返回值；生成一个包含原始模型权重 + LoRA 适配增量的完整权重文件。
    作用：
        推理优化：把 LoRA 适配器的低秩更新合并到原始模型权重中，
        即 W_merged = W_orig + BA（其中 B, A 是 LoRA 权重），
        这样推理时就无需保留 LoRA 适配器结构，可以像普通模型一样使用。
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    # 先复制所有非 LoRA 的原始权重到 CPU + float16
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 复制原始线性层权重
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 计算 LoRA 增量：BA，其中 A 是 [in_features, rank]，B 是 [rank, out_features]
                # BA 的结果是 [in_features, out_features]，与原权重形状相同
                lora_delta = module.lora.B.weight.data @ module.lora.A.weight.data
                # 合并到权重：W_merged = W_orig + BA
                state_dict[f'{name}.weight'] += lora_delta.cpu().half()
    torch.save(state_dict, save_path)
