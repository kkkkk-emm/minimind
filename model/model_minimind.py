import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 维度符号说明：
# B=batch size（批大小），T=当前输入序列长度，P=KV cache 中已经缓存的历史序列长度，
# H=hidden_size（隐藏层维度），V=vocab_size（词表大小），L=num_hidden_layers（层数），
# Nh=num_attention_heads（query 注意力头数），Nkv=num_key_value_heads（key/value 注意力头数），
# D=head_dim（每个注意力头的维度），I=intermediate_size（普通 MLP 中间层维度），
# E=num_experts（MoE expert 数），K=num_experts_per_tok（每个 token 选择的 expert 数）。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """
        输入：
            hidden_size：标量整数 H，表示每个 token 的隐藏向量长度；embedding、残差流、lm_head 输入都使用该维度。
            num_hidden_layers：标量整数 L，表示 MiniMindBlock 的层数。
            use_moe：bool，无张量维度；True 时每层 MLP 使用 MoEFeedForward，否则使用 FeedForward。
            **kwargs：额外标量配置，无 batch 维度：
                vocab_size=V，词表大小；num_attention_heads=Nh，query 头数；
                num_key_value_heads=Nkv，key/value 头数；head_dim=D，每个注意力头维度；
                intermediate_size=I，普通 MLP 中间层维度；max_position_embeddings，RoPE 最大位置数；
                num_experts=E，MoE expert 数；num_experts_per_tok=K，每个 token 激活的 expert 数。
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
            dim：标量整数，表示要归一化的最后一维长度；可等于 H（隐藏层维度）或 D（注意力头维度）。
            eps：标量浮点数，无张量维度；防止除零的极小常数。
        输出：
            无显式返回值；创建可学习参数 weight，形状为 [dim]，每个位置对应最后一维的一个通道。
        作用：
            初始化 RMSNorm 层，用均方根归一化替代 LayerNorm 中的均值方差归一化。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """
        输入：
            x：形状为 [..., dim] 的张量；... 表示任意前缀维度，例如：
                [B, T, H] 中 B=批大小、T=序列长度、H=隐藏层维度；
                [B, T, Nh, D] 中 Nh=query 头数、D=单头维度；
                [B, T, Nkv, D] 中 Nkv=key/value 头数、D=单头维度。
        输出：
            torch.Tensor：形状仍为 [..., dim]，只沿最后一维 dim 做 RMS 归一化。
        作用：
            计算 x / sqrt(mean(x^2) + eps)，保留方向信息并稳定激活尺度。
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        输入：
            x：形状为 [..., dim] 的张量；最后一维 dim 与 self.weight 的长度一致。
        输出：
            torch.Tensor：形状为 [..., dim]；归一化并乘以 [dim] 的可学习 weight，dtype 与输入保持一致。
        作用：
            在注意力和 MLP 前后稳定数值范围，减少深层网络训练不稳定。
        """
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    输入：
        dim：标量整数 D，表示每个注意力头的维度；RoPE 会在这个维度内部做两两旋转。
        end：标量整数 MaxPos，表示预计算的位置数量，即支持的位置下标范围为 [0, MaxPos-1]。
        rope_base：标量浮点数，无张量维度；RoPE 频率基数，值越大可支持更长上下文。
        rope_scaling：None 或 dict，无 batch 维度；启用 YaRN 时包含 original_max_position_embeddings、factor 等缩放参数。
    输出：
        (freqs_cos, freqs_sin)：
            freqs_cos 形状 [MaxPos, D]，第 0 维是位置 position，第 1 维是单头内部通道；
            freqs_sin 形状 [MaxPos, D]，维度含义相同。
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
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base)) # 返回维度索引
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1) 
            freqs = freqs * (1 - ramp + ramp / factor) # 缩放
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    输入：
        q：query 张量，形状 [B, T, Nh, D]；B=批大小，T=当前序列长度，Nh=query 头数，D=单头维度。
        k：key 张量，形状 [B, T, Nkv, D]；Nkv=key/value 头数，其余维度含义同 q。
        cos：RoPE cos 切片，形状 [T, D]；第 0 维对应当前 token 位置，第 1 维对应单头通道。
        sin：RoPE sin 切片，形状 [T, D]；维度含义同 cos。
        unsqueeze_dim：标量整数；默认 1，使 cos/sin 从 [T, D] 变成 [T, 1, D]，以广播到 q/k 的 head 维度。
    输出：
        (q_embed, k_embed)：
            q_embed 形状 [B, T, Nh, D]，与 q 相同；
            k_embed 形状 [B, T, Nkv, D]，与 k 相同。
    作用：
        把绝对位置信息以旋转方式注入 q/k，使注意力分数天然包含相对位置信息。
    """
    def rotate_half(x):
        """
        输入：
            x：形状 [..., D] 的 query/key 张量；... 可以是 [B, T, Nh] 或 [B, T, Nkv]，D=单头维度且应为偶数。
        输出：
            torch.Tensor：形状 [..., D]；把最后一维前后两半交换，并给后半部分取负。
        作用：
            实现 RoPE 公式中的二维旋转辅助变换，用于和 cos/sin 组合得到旋转后的 q/k。
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    输入：
        x：key 或 value 张量，形状 [B, T, Nkv, D]；B=批大小，T=序列长度，Nkv=key/value 头数，D=单头维度。
        n_rep：标量整数 R，表示每个 KV head 复制成多少份，通常 R = Nh / Nkv。
    输出：
        torch.Tensor：形状 [B, T, Nkv * R, D]；当 Nkv * R = Nh 时，可与 query 的 Nh 个头对齐。
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
            config：MiniMindConfig 配置对象，无 batch 维度；关键维度包括：
                H=config.hidden_size，输入/输出隐藏层维度；
                Nh=config.num_attention_heads，query 头数；
                Nkv=config.num_key_value_heads，key/value 头数；
                D=config.head_dim，每个注意力头的维度。
        输出：
            无显式返回值；创建投影层：
                q_proj: H -> Nh*D，k_proj/v_proj: H -> Nkv*D，o_proj: Nh*D -> H。
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
            x：当前层输入隐藏状态，形状 [B, T, H]；B=批大小，T=当前输入 token 数，H=隐藏层维度。
            position_embeddings：(cos, sin) 二元组：
                cos 形状 [T, D]，sin 形状 [T, D]；
                T 对应当前输入位置数量，D=单头维度。
            past_key_value：None 或 (past_k, past_v)：
                past_k 形状 [B, P, Nkv, D]，past_v 形状 [B, P, Nkv, D]；
                P=已经缓存的历史 token 数，Nkv=key/value 头数。
            use_cache：bool，无张量维度；True 时返回当前层新的 KV cache。
            attention_mask：None 或形状 [B, T] / [B, P+T]；
                第 0 维 B 是 batch，第 1 维是可见 token 位置，1 表示有效，0 表示 padding。
        输出：
            (output, past_kv)：
                output 形状 [B, T, H]，与输入 x 的 batch、序列和隐藏维度一致；
                past_kv 为 None 或 (k, v)，其中 k/v 形状 [B, P+T, Nkv, D]。
        作用：
            计算因果自注意力：生成 q/k/v、注入 RoPE、拼接历史 KV、执行 attention，再映射回 hidden_size。
        """
        bsz, seq_len, _ = x.shape
        # x: [B, T, H] -> xq: [B, T, Nh*D], xk/xv: [B, T, Nkv*D]。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆出 head 维度：xq [B, T, Nh, D]，xk/xv [B, T, Nkv, D]。
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
        # 转置后 xq/xk/xv 都是 [B, Nh, T或P+T, D]，符合 attention 的输入习惯。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention 输入/输出：xq [B, Nh, T, D] -> output [B, Nh, T, D]。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # scores: [B, Nh, T, P+T]，最后一维表示每个 query token 可关注的 key 位置。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax 后仍为 [B, Nh, T, P+T]，乘 xv [B, Nh, P+T, D] 得到 [B, Nh, T, D]。
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # [B, Nh, T, D] -> [B, T, Nh*D]，再经 o_proj 回到 [B, T, H]。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        """
        输入：
            config：MiniMindConfig 配置对象，无 batch 维度；H=config.hidden_size，I=config.intermediate_size。
            intermediate_size：None 或标量整数 I；表示 MLP 中间层维度，不传时使用 config.intermediate_size。
        输出：
            无显式返回值；创建线性层：
                gate_proj: H -> I，up_proj: H -> I，down_proj: I -> H。
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
            x：隐藏状态，形状 [B, T, H]；B=批大小，T=序列长度，H=隐藏层维度。
        输出：
            torch.Tensor：形状 [B, T, H]；中间会经过 [B, T, I]，I=MLP 中间层维度。
        作用：
            计算 down_proj(act(gate_proj(x)) * up_proj(x))，用门控结构增强 MLP 表达能力。
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        输入：
            config：MiniMindConfig 配置对象，无 batch 维度；关键维度包括：
                H=config.hidden_size，输入/输出隐藏层维度；
                E=config.num_experts，expert 总数；
                K=config.num_experts_per_tok，每个 token 选择的 expert 数；
                moe_intermediate_size，单个 expert 内部 MLP 中间层维度。
        输出：
            无显式返回值；创建 gate: H -> E，以及 E 个 FeedForward expert。
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
            x：隐藏状态，形状 [B, T, H]；B=批大小，T=序列长度，H=隐藏层维度。
        输出：
            torch.Tensor：形状 [B, T, H]；内部会展平成 [B*T, H] 做路由。
            额外状态 self.aux_loss：标量张量 []，训练时表示 MoE 负载均衡辅助损失。
        作用：
            对每个 token 计算 expert 路由概率，选 top-k experts 处理并加权合并；训练时额外记录负载均衡 aux_loss。
        """
        batch_size, seq_len, hidden_dim = x.shape
        # [B, T, H] -> [B*T, H]，把每个 token 当作一个独立路由单位。
        x_flat = x.view(-1, hidden_dim)
        # scores: [B*T, E]，每行是一个 token 分配到 E 个 expert 的概率。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # topk_weight/topk_idx: [B*T, K]，K 是每个 token 选中的 expert 数。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20) # 权重归一化
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 只把被路由到当前 expert 的 token 送入该 expert，减少无效计算。
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                # expert(x_flat[token_idx]) 输出 [选中token数, H]，乘路由权重后累加回 y 的对应 token 行。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # aux_loss 鼓励 token 更均匀地分配到各 expert，避免少数 expert 过载。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0) # [K, E]每个选位（第一选择、第二选择...)中各个专家的负载情况
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        # [B*T, H] -> [B, T, H]，恢复 batch 和序列两个维度。
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        输入：
            layer_id：标量整数，无张量维度；表示当前 block 在 L 层中的编号。
            config：MiniMindConfig 配置对象，无 batch 维度；提供 H、Nh、Nkv、D、I、MoE 等结构维度。
        输出：
            无显式返回值；创建一个输入/输出均为 [B, T, H] 的 Transformer block。
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
            hidden_states：当前层输入，形状 [B, T, H]；B=批大小，T=当前 token 数，H=隐藏层维度。
            position_embeddings：(cos, sin)，cos/sin 均为 [T, D]；D=单头维度。
            past_key_value：None 或 (past_k, past_v)，past_k/past_v 均为 [B, P, Nkv, D]。
            use_cache：bool，无张量维度；True 时返回本层当前 KV cache。
            attention_mask：None 或 [B, T] / [B, P+T]；B=batch，第二维是 token 位置。
        输出：
            (hidden_states, present_key_value)：
                hidden_states 形状 [B, T, H]；
                present_key_value 为 None 或 (k, v)，k/v 形状 [B, P+T, Nkv, D]。
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
            config：MiniMindConfig 配置对象，无 batch 维度；关键维度包括：
                V=config.vocab_size，词表大小；
                H=config.hidden_size，隐藏层维度；
                L=config.num_hidden_layers，block 层数；
                D=config.head_dim，RoPE 和注意力单头维度；
                max_position_embeddings，预计算 RoPE 的最大位置数。
        输出：
            无显式返回值；创建：
                embed_tokens: V -> H；
                layers: L 个 MiniMindBlock；
                freqs_cos/freqs_sin: [max_position_embeddings, D]。
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
            input_ids：token id 张量，形状 [B, T]；B=批大小，T=当前输入 token 数。
            attention_mask：None 或 [B, T] / [B, P+T]；1 表示有效 token，0 表示 padding。
            past_key_values：None 或长度为 L 的列表：
                每个元素为 None 或 (past_k, past_v)，past_k/past_v 形状 [B, P, Nkv, D]；
                L=层数，P=历史缓存长度，Nkv=key/value 头数，D=单头维度。
            use_cache：bool，无张量维度；True 时返回每层新的 KV cache。
            **kwargs：兼容 Transformers 调用的额外参数；本函数不主动读取，无固定张量维度。
        输出：
            (hidden_states, presents, aux_loss)：
                hidden_states 形状 [B, T, H]；
                presents 长度为 L，每个元素为 None 或 (k, v)，k/v 形状 [B, P+T, Nkv, D]；
                aux_loss 是标量张量 []，dense 模型为 0，MoE 模型为各层辅助损失之和。
        作用：
            完成主体模型前向：token embedding、按位置切 RoPE、多层 decoder block、最终归一化，并汇总 MoE aux_loss。
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos=P；没有 cache 时 P=0，有 cache 时 P 是历史 key/value 的序列长度。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # input_ids [B, T] -> token embedding [B, T, H]。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 从 RoPE 表中切出当前 token 的位置编码：cos/sin 均为 [T, D]。
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
        # 最后一层 RMSNorm 仍保持 [B, T, H]。
        hidden_states = self.norm(hidden_states)
        # MoE 时把每层标量 aux_loss 相加；dense 模型没有 MoE 层，结果为标量 0。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        """
        输入：
            config：MiniMindConfig 或 None，无 batch 维度；关键维度 V=词表大小，H=隐藏层维度。
        输出：
            无显式返回值；创建 MiniMindModel 主体，以及 lm_head: H -> V。
            当 tie_word_embeddings=True 时，embed_tokens.weight 和 lm_head.weight 共享形状 [V, H] 的权重。
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
            input_ids：输入 token id，形状 [B, T]；B=批大小，T=当前输入 token 数。
            attention_mask：None 或 [B, T] / [B, P+T]；用于屏蔽 padding。
            past_key_values：None 或长度为 L 的 KV cache 列表；每层 k/v 形状 [B, P, Nkv, D]。
            use_cache：bool，无张量维度；True 时返回新的 past_key_values。
            logits_to_keep：标量整数或索引；为 0 时保留全部 T 个位置，>0 时保留最后 logits_to_keep 个位置。
            labels：None 或训练标签张量 [B, T]；padding/忽略位置为 -100，用于计算 causal LM loss。
            **kwargs：传给 MiniMindModel 的额外参数，无固定张量维度。
        输出：
            MoeCausalLMOutputWithPast：
                loss：None 或标量张量 []；
                aux_loss：标量张量 []；
                logits：形状 [B, T_keep, V]，T_keep=T 或 logits_to_keep 指定的保留长度，V=词表大小；
                past_key_values：长度为 L 的列表，每层 k/v 为 [B, P+T, Nkv, D] 或 None；
                hidden_states：形状 [B, T, H]。
        作用：
            调用主体模型得到 hidden_states，再用 lm_head 映射到词表 logits；训练时右移 logits/labels 计算下一个 token 的交叉熵。
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep # 推理阶段只取最后一位，训练阶段全都要
        # hidden_states[:, slice_indices, :]: [B, T_keep, H] -> logits: [B, T_keep, V]。
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 语言模型训练目标是用位置 t 的输出预测位置 t+1 的 token，因此 logits 去掉最后一位，labels 去掉第一位。
            # x: [B, T-1, V]，y: [B, T-1]；view 后变成 [B*(T-1), V] 和 [B*(T-1)] 计算交叉熵。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        输入：
            inputs：初始 prompt token id，形状 [B, T0]；B=输入 batch，T0=prompt 长度。
                也可用 kwargs["input_ids"] 传入同形状张量。
            attention_mask：None 或 [B, T0]；1 表示有效 token，0 表示 padding。
                复制 num_return_sequences 后会变为 [B*N, T0]，N=num_return_sequences。
            max_new_tokens：标量整数 M，无张量维度；最多生成 M 个新 token。
            temperature：标量浮点数，无张量维度；用于缩放 logits。
            top_p：标量浮点数，无张量维度；核采样累计概率阈值。
            top_k：标量整数，无张量维度；只保留 logits 最高的 k 个 token。
            eos_token_id：标量整数或 None；结束 token id。
            streamer：可选对象，无张量维度；需要支持 put/end 方法。
            use_cache：bool，无张量维度；True 时每步维护 KV cache。
            num_return_sequences：标量整数 N；每条输入复制为 N 条候选，因此生成 batch 维变为 B*N。
            do_sample：bool，无张量维度；True 按概率采样，False 使用 argmax。
            repetition_penalty：标量浮点数，无张量维度；重复惩罚系数。
            **kwargs：额外参数：
                past_key_values 可为长度 L 的列表，每层 k/v 形状 [B*N, P, Nkv, D]；
                return_kv 为 bool 时决定是否额外返回 KV cache。
        输出：
            默认返回 input_ids，形状 [B*N, T0+G]；
                G 是实际生成长度，满足 0 <= G <= M，遇到 eos 可提前停止。
            当 kwargs["return_kv"] 为 True 时，返回：
                generated_ids：形状 [B*N, T0+G]；
                past_kv：长度 L 的列表，每层 k/v 形状 [B*N, P+G, Nkv, D]。
        作用：
            手写自回归生成循环：每次用模型预测下一个 token，经过 temperature/top-k/top-p/重复惩罚后采样或贪心选择，并维护 KV cache。
        """
        # 兼容 Transformers 常用的 input_ids 参数名；repeat 用于一次生成多条候选。
        # [B, T0] -> [B*N, T0]，N=num_return_sequences。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # 输入复制了 num_return_sequences 次，mask 也要按相同倍数复制。
        # attention_mask: [B, T0] -> [B*N, T0]。
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # 如果外部传入已有 KV cache，就从该 cache 后的位置继续生成。
        past_key_values = kwargs.pop("past_key_values", None) # [Num_layers][k/v][B*N][P][Nkv][D]
        # finished 记录每条序列是否已经生成 eos，避免已结束序列继续产生普通 token。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        # streamer 用于边生成边把 token 交给外部 UI/控制台显示。
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 使用 KV cache 时，只需要把还没进 cache 的新 token 喂给模型。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # input_ids[:, past_len:] 的形状是 [B*N, 当前未缓存长度]，通常首轮为 [B*N, T0]，之后为 [B*N, 1]。
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 生成一个新 token 后，attention_mask 也追加一个有效位置。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 只取最后一个位置的 logits，因为自回归生成每轮只决定下一个 token。
            # outputs.logits: [B*N, 当前步长度, V] -> logits: [B*N, V]。
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 对已经出现过的 token 降低分数，减少机械重复。
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                # top-k：把第 k 大分数以下的 token 置为 -inf，使它们采样概率为 0。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                # top-p：按概率从高到低累加，只保留累计概率阈值内的候选 token。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 至少保留最高分 token，避免所有候选都被 mask 掉。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # scatter 把排序后的 mask 放回原词表索引位置。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 采样模式按 softmax 概率抽样；非采样模式直接选最大 logits。
            # next_token: [B*N, 1]，每行是该序列本轮生成的一个 token id。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 已经结束的序列后续强制补 eos，保证 batch 内长度一致。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 把本轮生成的 token 接到完整序列末尾。
            # input_ids: [B*N, 当前总长度] -> [B*N, 当前总长度+1]。
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # forward 返回的新 KV cache 覆盖旧 cache，下一轮可以增量推理。
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                # 更新每条序列的结束状态；全部结束时提前退出生成循环。
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        # 通知 streamer 生成完成。
        if streamer: streamer.end()
        # 调试或多轮生成时可以把 KV cache 一起返回，供下次继续使用。
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
