import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """
        输入：
            hidden_size：隐藏层维度，也是 token embedding 和 transformer block 内部主通道维度。
            num_hidden_layers：Transformer block 层数。
            use_moe：是否把普通 MLP 替换为 MoEFeedForward。
            **kwargs：额外配置项，如 vocab_size、head 数、RoPE 参数、MoE expert 数等。
        输出：
            无显式返回值；初始化 PretrainedConfig 所需字段和 MiniMind 模型结构字段。
        作用：
            集中保存模型结构、训练/推理超参数，使模型构建、权重保存和 Transformers 接口能读取同一份配置。
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        输入：
            dim：最后一维的特征维度。
            eps：防止除零的极小常数。
        输出：
            无显式返回值；创建可学习缩放参数 weight。
        作用：
            初始化 RMSNorm 层，用均方根归一化替代 LayerNorm 中的均值方差归一化。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """
        输入：
            x：任意前缀维度、最后一维为 hidden/head_dim 的张量。
        输出：
            torch.Tensor：按最后一维 RMS 归一化后的张量。
        作用：
            计算 x / sqrt(mean(x^2) + eps)，保留方向信息并稳定激活尺度。
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        输入：
            x：需要归一化的隐藏状态。
        输出：
            torch.Tensor：归一化并乘以可学习 weight 后的结果，dtype 与输入保持一致。
        作用：
            在注意力和 MLP 前后稳定数值范围，减少深层网络训练不稳定。
        """
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    输入：
        dim：每个注意力头的维度。
        end：预计算的最大位置长度。
        rope_base：RoPE 频率基数，值越大可支持更长上下文。
        rope_scaling：可选 YaRN 缩放配置，用于推理时扩展上下文长度。
    输出：
        (freqs_cos, freqs_sin)：形状为 [end, dim] 的 cos/sin 位置编码表。
    作用：
        预先计算旋转位置编码需要的 cos/sin，forward 时按序列位置切片使用，避免每次重复计算。
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    输入：
        q：query 张量，通常形状为 [batch, seq_len, heads, head_dim]。
        k：key 张量，通常形状为 [batch, seq_len, kv_heads, head_dim]。
        cos：当前位置对应的 RoPE cos 表。
        sin：当前位置对应的 RoPE sin 表。
        unsqueeze_dim：把 cos/sin 扩展到 q/k 形状时插入的维度。
    输出：
        (q_embed, k_embed)：应用旋转位置编码后的 query 和 key。
    作用：
        把绝对位置信息以旋转方式注入 q/k，使注意力分数天然包含相对位置信息。
    """
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    输入：
        x：key 或 value 张量，形状为 [batch, seq_len, num_key_value_heads, head_dim]。
        n_rep：每个 KV head 需要复制的次数。
    输出：
        torch.Tensor：复制后的 KV 张量，head 数变为 num_key_value_heads * n_rep。
    作用：
        支持 GQA/MQA：让较少的 KV heads 复用到更多 query heads，降低 KV 计算和缓存成本。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        输入：
            config：MiniMindConfig，提供 hidden_size、head 数、head_dim、dropout、flash_attn 等注意力参数。
        输出：
            无显式返回值；创建 q/k/v/o 投影、q/k RMSNorm 和 dropout 层。
        作用：
            初始化自注意力模块。该实现支持 GQA、RoPE、KV cache，并在条件满足时使用 PyTorch Flash Attention。
        """
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        输入：
            x：当前层输入隐藏状态，形状为 [batch, seq_len, hidden_size]。
            position_embeddings：RoPE 的 (cos, sin) 切片，对应当前 token 位置。
            past_key_value：可选历史 KV cache，生成阶段用于避免重复计算历史 token。
            use_cache：是否返回新的 KV cache。
            attention_mask：可选注意力 mask，padding 位置会被屏蔽。
        输出：
            (output, past_kv)：注意力输出隐藏状态，以及可选的当前层 KV cache。
        作用：
            计算因果自注意力：生成 q/k/v、注入 RoPE、拼接历史 KV、执行 attention，再映射回 hidden_size。
        """
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        # RoPE 只作用在 q/k 上，value 不携带位置信息。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            # 生成时把历史 KV 与当前 token 的 KV 拼接，避免重复计算历史上下文。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # GQA 中 KV head 数可能少于 query head 数，因此需要 repeat 到相同 head 数后再做注意力。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        """
        输入：
            config：MiniMindConfig，提供 hidden_size、intermediate_size、hidden_act 等 MLP 参数。
            intermediate_size：可选中间层维度；不传时使用 config.intermediate_size。
        输出：
            无显式返回值；创建 gate/up/down 三个线性层和激活函数。
        作用：
            初始化 SwiGLU 风格前馈网络，用于每个 Transformer block 的非线性特征变换。
        """
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        输入：
            x：隐藏状态，形状通常为 [batch, seq_len, hidden_size]。
        输出：
            torch.Tensor：前馈网络输出，形状与输入一致。
        作用：
            计算 down_proj(act(gate_proj(x)) * up_proj(x))，用门控结构增强 MLP 表达能力。
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        输入：
            config：MiniMindConfig，提供 expert 数、每 token 激活 expert 数、MoE 中间层维度和辅助损失系数。
        输出：
            无显式返回值；创建 router gate 和多个 FeedForward experts。
        作用：
            初始化 MoE 前馈层，让不同 token 路由到不同 expert，从而提高模型容量但减少单 token 激活计算量。
        """
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        输入：
            x：隐藏状态，形状为 [batch, seq_len, hidden_size]。
        输出：
            torch.Tensor：MoE 聚合后的隐藏状态，形状与输入一致。
        作用：
            对每个 token 计算 expert 路由概率，选 top-k experts 处理并加权合并；训练时额外记录负载均衡 aux_loss。
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 只把被路由到当前 expert 的 token 送入该 expert，减少无效计算。
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # aux_loss 鼓励 token 更均匀地分配到各 expert，避免少数 expert 过载。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        输入：
            layer_id：层编号，本实现中主要用于构造列表时标识层序。
            config：MiniMindConfig，决定注意力、归一化和 MLP/MoE 结构。
        输出：
            无显式返回值；创建一个 Transformer block 的注意力、归一化和前馈模块。
        作用：
            初始化 MiniMind 的单层 decoder block，结构为 RMSNorm -> self-attention -> residual -> RMSNorm -> MLP/MoE -> residual。
        """
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        输入：
            hidden_states：当前层输入，形状为 [batch, seq_len, hidden_size]。
            position_embeddings：当前序列位置对应的 RoPE cos/sin。
            past_key_value：可选历史 KV cache。
            use_cache：是否返回当前层 KV cache。
            attention_mask：可选 padding mask。
        输出：
            (hidden_states, present_key_value)：本层输出隐藏状态和可选 KV cache。
        作用：
            执行一层 Transformer decoder：先做带残差的因果注意力，再做带残差的前馈网络。
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        输入：
            config：MiniMindConfig，包含词表大小、层数、hidden_size、RoPE 和 MoE 等模型结构参数。
        输出：
            无显式返回值；创建 embedding、dropout、多层 MiniMindBlock、最终 RMSNorm 和 RoPE 缓冲区。
        作用：
            初始化不含 lm_head 的主体 decoder 模型，把 token id 转换为最终隐藏状态。
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        输入：
            input_ids：token id 张量，形状为 [batch, seq_len]。
            attention_mask：可选 mask，1 表示有效 token，0 表示需要屏蔽。
            past_key_values：可选每层历史 KV cache，生成阶段使用。
            use_cache：是否返回新的 KV cache。
            **kwargs：兼容 Transformers 调用的额外参数，本函数不主动使用。
        输出：
            (hidden_states, presents, aux_loss)：最终隐藏状态、每层 KV cache 列表和 MoE 辅助损失。
        作用：
            完成主体模型前向：token embedding、按位置切 RoPE、多层 decoder block、最终归一化，并汇总 MoE aux_loss。
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            # 每层都复用同一段 RoPE 位置表，但拥有独立的注意力、MLP/MoE 参数。
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        """
        输入：
            config：MiniMindConfig；为 None 时使用默认配置。
        输出：
            无显式返回值；创建 MiniMindModel 主体和 lm_head。
        作用：
            封装用于自回归语言模型训练/推理的完整模型，并接入 Transformers 的 PreTrainedModel/GenerationMixin 接口。
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """
        输入：
            input_ids：输入 token id，形状为 [batch, seq_len]。
            attention_mask：可选 mask，用于屏蔽 padding。
            past_key_values：可选 KV cache，生成阶段用于增量推理。
            use_cache：是否返回新的 KV cache。
            logits_to_keep：只计算/保留最后若干位置的 logits；训练时默认为 0，表示保留全部。
            labels：可选训练标签；传入时计算 causal LM loss。
            **kwargs：传给 MiniMindModel 的额外参数。
        输出：
            MoeCausalLMOutputWithPast：包含 loss、aux_loss、logits、past_key_values、hidden_states。
        作用：
            调用主体模型得到 hidden_states，再用 lm_head 映射到词表 logits；训练时右移 logits/labels 计算下一个 token 的交叉熵。
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 语言模型训练目标是用位置 t 的输出预测位置 t+1 的 token，因此 logits 去掉最后一位，labels 去掉第一位。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
