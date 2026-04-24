"""Transformer architecture and checkpoint contract logic for OPF models."""

import dataclasses
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .._common.env import get_env_bool
from .weights import Checkpoint

try:
    from .triton_moe import grouped_matmul, grouped_swiglu_w2
except ModuleNotFoundError as exc:
    grouped_matmul = None
    grouped_swiglu_w2 = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_param_dtype(value: str | None) -> torch.dtype:
    """Resolve the configured parameter dtype alias."""
    if value is None:
        return torch.bfloat16
    key = value.lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported param_dtype {value!r} (expected bf16 or fp32)")
    return _DTYPE_ALIASES[key]


def _configure_torch_math() -> None:
    """Configure torch math flags for deterministic high-precision inference."""
    allow_tf32 = get_env_bool("OPF_ALLOW_TF32")
    if allow_tf32:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _batched_linear_with_parity(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply one batched expert linear projection."""
    # x: [B, E, K], weight: [B, E, K, O], bias: [B, E, O] -> [B, E, O]
    bsz, experts, k_dim = x.shape
    _, _, _, o_dim = weight.shape
    x_bmm = x.reshape(bsz * experts, 1, k_dim)
    w_bmm = weight.reshape(bsz * experts, k_dim, o_dim)
    out = torch.bmm(x_bmm, w_bmm).reshape(bsz, experts, o_dim)
    if bias is not None:
        out = out + bias
    return out


PRIVACY_FILTER_MODEL_TYPE = "privacy_filter"
REQUIRED_ENCODER_CONFIG_KEYS: tuple[str, ...] = (
    "model_type",
    "encoding",
    "num_hidden_layers",
    "num_experts",
    "experts_per_token",
    "vocab_size",
    "num_labels",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "bidirectional_context",
    "bidirectional_left_context",
    "bidirectional_right_context",
    "initial_context_length",
    "rope_theta",
    "rope_scaling_factor",
    "rope_ntk_alpha",
    "rope_ntk_beta",
    "param_dtype",
)


def _require_triton() -> None:
    """Raise a clear error when Triton-backed kernels are requested but unavailable."""
    if _TRITON_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Triton-backed MoE kernels require the optional `triton` dependency. "
            "Install `triton` or unset OPF_MOE_TRITON."
        ) from _TRITON_IMPORT_ERROR


def _validate_encoder_artifact_contract(*, json_config: dict[str, object]) -> None:
    """Validate that an encoder checkpoint matches the runtime contract."""
    model_type = json_config.get("model_type")
    missing_keys = sorted(
        k for k in REQUIRED_ENCODER_CONFIG_KEYS if k not in json_config
    )
    errors: list[str] = []
    if missing_keys:
        errors.append(f"missing config keys: {', '.join(missing_keys)}")

    if model_type != PRIVACY_FILTER_MODEL_TYPE:
        errors.append(
            f"model_type must be {PRIVACY_FILTER_MODEL_TYPE!r} (got {model_type!r})"
        )

    _required_string_config(json_config, "encoding", errors)
    bidirectional_context = _required_bool_config(
        json_config, "bidirectional_context", errors
    )
    bidirectional_left_context = _required_nonnegative_int_config(
        json_config, "bidirectional_left_context", errors
    )
    bidirectional_right_context = _required_nonnegative_int_config(
        json_config, "bidirectional_right_context", errors
    )
    sliding_window = _required_nonnegative_int_config(
        json_config, "sliding_window", errors
    )

    if bidirectional_context is False:
        errors.append(
            "bidirectional_context must be true. "
            "Only bidirectional OPF checkpoints are supported."
        )

    if (
        bidirectional_context is True
        and bidirectional_left_context is not None
        and bidirectional_right_context is not None
        and sliding_window is not None
    ):
        expected_bandwidth = (
            bidirectional_left_context + bidirectional_right_context + 1
        )
        if sliding_window != expected_bandwidth:
            errors.append(
                "config.bidirectional_context=True requires "
                "sliding_window=bidirectional_left_context+bidirectional_right_context+1 "
                f"(got sliding_window={sliding_window}, expected={expected_bandwidth})"
            )

    if errors:
        raise ValueError(
            "Incompatible OPF encoder artifact configuration: " + "; ".join(errors)
        )


def _required_bool_config(
    json_config: dict[str, object], key: str, errors: list[str]
) -> bool | None:
    """Read a required boolean checkpoint config value."""
    if key not in json_config:
        return None
    value = json_config[key]
    if not isinstance(value, bool):
        errors.append(f"{key} must be boolean (got {type(value).__name__})")
        return None
    return value


def _required_string_config(
    json_config: dict[str, object], key: str, errors: list[str]
) -> str | None:
    """Read a required nonempty string checkpoint config value."""
    if key not in json_config:
        return None
    value = json_config[key]
    if not isinstance(value, str):
        errors.append(f"{key} must be a string (got {type(value).__name__})")
        return None
    if not value:
        errors.append(f"{key} must be non-empty")
        return None
    return value


def _required_nonnegative_int_config(
    json_config: dict[str, object], key: str, errors: list[str]
) -> int | None:
    """Read a required nonnegative integer checkpoint config value."""
    if key not in json_config:
        return None
    value = json_config[key]
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{key} must be an integer (got {type(value).__name__})")
        return None
    if value < 0:
        errors.append(f"{key} must be >= 0 (got {value})")
        return None
    return value


@dataclass
class ModelConfig:
    """Configuration for an OPF transformer checkpoint."""

    model_type: str = PRIVACY_FILTER_MODEL_TYPE
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    num_labels: int | None = None
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    packed_geglu: bool = False
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    bidirectional_context: bool = False
    bidirectional_left_context: int = 0
    bidirectional_right_context: int = 0
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    torch_ops_batch: int = 32
    param_dtype: str = "bfloat16"


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        """Initialize RMSNorm parameters."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one tensor over its final dimension."""
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply cached rotary embeddings to one query or key tensor."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).reshape(x.shape)


class RotaryEmbedding(torch.nn.Module):
    """RoPE cache manager with optional YaRN-style scaling."""

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        """Initialize rotary caches for the configured attention head size."""
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        if get_env_bool("OPF_ATTN_LOW_PRECISION"):
            dtype = torch.bfloat16
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        # precompute rotary caches on CPU and move to target device.
        max_positions = int(self.initial_context_length * self.scaling_factor)
        max_positions = max(max_positions, self.initial_context_length)
        self.max_position_embeddings = max_positions
        cos, sin = self._compute_cos_sin(
            self.max_position_embeddings, device=torch.device("cpu")
        )
        target_device = device or torch.device("cpu")
        self.register_buffer("cos_cache", cos.to(target_device), persistent=False)
        self.register_buffer("sin_cache", sin.to(target_device), persistent=False)

    def _compute_concentration_and_inv_freq(
        self, device: torch.device | None = None
    ) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        device = device or self.device
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, device: torch.device | None = None):
        """Compute rotary cosine and sine caches."""
        concentration, inv_freq = self._compute_concentration_and_inv_freq(
            device=device
        )
        device = device or self.device
        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos.to(self.dtype), sin.to(self.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        if query.dim() != 3 or key.dim() != 3:
            raise ValueError("RotaryEmbedding expects batched 3D query/key tensors")
        batch_size, num_tokens, _ = query.shape
        if num_tokens > self.cos_cache.shape[0]:
            # Extend caches if needed.
            cos, sin = self._compute_cos_sin(num_tokens, device=torch.device("cpu"))
            self.cos_cache = cos.to(query.device)
            self.sin_cache = sin.to(query.device)
        if self.cos_cache.device != query.device:
            cos_cache = self.cos_cache.to(query.device)
            sin_cache = self.sin_cache.to(query.device)
        else:
            cos_cache = self.cos_cache
            sin_cache = self.sin_cache
        cos = cos_cache[:num_tokens]
        sin = sin_cache[:num_tokens]

        query_shape = query.shape
        query = query.view(batch_size, num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos[None, ...], sin[None, ...])
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(batch_size, num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos[None, ...], sin[None, ...])
        key = key.reshape(key_shape)
        return query, key


def sdpa(
    Q,
    K,
    V,
    S,
    sm_scale,
    sliding_window=0,
    *,
    attention_mask: torch.Tensor | None = None,
    bidirectional_context=False,
    bidirectional_left_context=0,
    bidirectional_right_context=0,
):
    """Run the model's production attention path for causal or local windows."""
    if Q.dim() != 5:
        raise ValueError(
            "sdpa expects batched Q with shape [batch, tokens, heads, q_mult, d_head]"
        )
    bsz, n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (bsz, n_tokens, n_heads, d_head)
    assert V.shape == (bsz, n_tokens, n_heads, d_head)
    attn_low_precision = get_env_bool("OPF_ATTN_LOW_PRECISION")
    if attention_mask is not None:
        if attention_mask.shape != (bsz, n_tokens):
            raise ValueError(
                "attention_mask shape mismatch: "
                f"expected {(bsz, n_tokens)}, got {tuple(attention_mask.shape)}"
            )
        attention_mask = attention_mask.to(device=Q.device, dtype=torch.bool)
    if bidirectional_context or sliding_window > 0:
        left_ctx = (
            int(bidirectional_left_context)
            if bidirectional_context
            else int(sliding_window)
        )
        right_ctx = int(bidirectional_right_context) if bidirectional_context else 0
        if left_ctx < 0 or right_ctx < 0:
            raise ValueError(
                "bidirectional_left_context and bidirectional_right_context must be >= 0 "
                f"(got {left_ctx}/{right_ctx})"
            )
        window = left_ctx + right_ctx + 1
        Kp = F.pad(K, (0, 0, 0, 0, left_ctx, right_ctx))
        Vp = F.pad(V, (0, 0, 0, 0, left_ctx, right_ctx))
        Kwin = Kp.unfold(1, window, 1).permute(0, 1, 4, 2, 3)
        Vwin = Vp.unfold(1, window, 1).permute(0, 1, 4, 2, 3)
        idx = torch.arange(window, device=Q.device) - left_ctx
        pos = torch.arange(n_tokens, device=Q.device)[:, None] + idx[None, :]
        valid = (pos >= 0) & (pos < n_tokens)
        scores = torch.einsum("bthqd,btwhd->bthqw", Q, Kwin)
        if not attn_low_precision:
            scores = scores.float()
        scores *= sm_scale
        score_valid = valid[None, :, None, None, :]
        if attention_mask is not None:
            padded_valid = F.pad(attention_mask, (left_ctx, right_ctx), value=False)
            key_valid = padded_valid.unfold(1, window, 1)
            score_valid = score_valid & key_valid[:, :, None, None, :]
        scores = scores.masked_fill(~score_valid, -float("inf"))
        sink_scores = (S * math.log(2.0)).reshape(n_heads, q_mult)
        if attn_low_precision:
            sink_scores = sink_scores.to(V.dtype)
        sink_scores = sink_scores[None, None, :, :, None].expand(
            bsz, n_tokens, -1, -1, 1
        )
        scores = torch.cat([scores, sink_scores], dim=-1)
        if attn_low_precision:
            scores = scores.to(V.dtype)
        W = torch.softmax(scores, dim=-1)
        W = W[..., :-1].to(V.dtype)
        attn = torch.einsum("bthqw,btwhd->bthqd", W, Vwin)
        return attn.reshape(bsz, n_tokens, -1)
    # sliding_window == 0 means no sliding window
    K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    # Sink values are stored in log2 space; convert to natural log for this kernel.
    sink_scores = (S * math.log(2.0)).reshape(n_heads, q_mult)
    if attn_low_precision:
        sink_scores = sink_scores.to(V.dtype)
    mask: torch.Tensor | None = None
    if bidirectional_context:
        left_ctx = int(bidirectional_left_context)
        right_ctx = int(bidirectional_right_context)
        if left_ctx < 0 or right_ctx < 0:
            raise ValueError(
                "bidirectional_left_context and bidirectional_right_context must be >= 0 "
                f"(got {left_ctx}/{right_ctx})"
            )
        mask = torch.zeros((n_tokens, n_tokens), device=Q.device, dtype=torch.float32)
        # Keep asymmetric local band [-left_ctx, +right_ctx].
        mask += torch.triu(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=right_ctx + 1,
        )
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-(left_ctx + 1),
        )
    else:
        mask = torch.triu(
            torch.full(
                (n_tokens, n_tokens),
                -float("inf"),
                device=Q.device,
                dtype=torch.float32,
            ),
            diagonal=1,
        )
        if sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens, n_tokens), -float("inf")),
                diagonal=-sliding_window,
            )
    scores = torch.einsum("bthqd,bshqd->bthqs", Q, K)
    if not attn_low_precision:
        scores = scores.float()
    scores *= sm_scale
    if mask is not None:
        scores += mask[None, :, None, None, :]
    if attention_mask is not None:
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, None, :], -float("inf")
        )
    sink_scores = sink_scores[None, None, :, :, None].expand(bsz, n_tokens, -1, -1, 1)
    scores = torch.cat([scores, sink_scores], dim=-1)
    if attn_low_precision:
        scores = scores.to(V.dtype)
    W = torch.softmax(scores, dim=-1)
    W = W[..., :-1].to(V.dtype)
    attn = torch.einsum("bthqs,bshqd->bthqd", W, V)
    return attn.reshape(bsz, n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    """Transformer attention block with RoPE, sink logits, and output projection."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """Initialize one transformer attention block."""
        super().__init__()
        param_dtype = _resolve_param_dtype(config.param_dtype)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Default attn_banded=(True,) implies banding for every layer.
        self.sliding_window = config.sliding_window
        self.bidirectional_context = bool(config.bidirectional_context)
        self.bidirectional_left_context = int(config.bidirectional_left_context)
        self.bidirectional_right_context = int(config.bidirectional_right_context)
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.float32)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=param_dtype
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=param_dtype,
        )
        self.qk_scale = 1 / math.sqrt(math.sqrt(config.head_dim))
        self.sm_scale = 1.0
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the attention block and residual connection."""
        if x.dim() != 3:
            raise ValueError("AttentionBlock expects batched 3D tensor input")
        if attention_mask is not None and attention_mask.shape != x.shape[:2]:
            raise ValueError(
                "attention_mask shape mismatch: "
                f"expected {tuple(x.shape[:2])}, got {tuple(attention_mask.shape)}"
            )
        t = self.norm(x)
        if t.dtype != self.qkv.weight.dtype:
            t = t.to(self.qkv.weight.dtype)
        qkv = F.linear(
            t,
            self.qkv.weight,
            self.qkv.bias,
        )
        q = qkv[:, :, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            :,
            self.num_attention_heads * self.head_dim : (
                self.num_attention_heads + self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (
                self.num_attention_heads + 2 * self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()

        q, k = self.rope(q, k)
        q = q * self.qk_scale
        k = k * self.qk_scale
        sinks = self.sinks
        bsz, n_tokens, _ = q.shape
        q = q.view(
            bsz,
            n_tokens,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
        attn_out = sdpa(
            q,
            k,
            v,
            sinks,
            self.sm_scale,
            self.sliding_window,
            attention_mask=attention_mask,
            bidirectional_context=self.bidirectional_context,
            bidirectional_left_context=self.bidirectional_left_context,
            bidirectional_right_context=self.bidirectional_right_context,
        )

        if attn_out.dtype != self.out.weight.dtype:
            attn_out = attn_out.to(self.out.weight.dtype)
        proj_bias = self.out.bias
        proj = F.linear(
            attn_out,
            self.out.weight,
            proj_bias,
        )
        proj = proj.to(x.dtype)
        return x + proj


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, packed: bool = False):
    """Apply the SwiGLU nonlinearity."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"swiglu expects even last dim, got {x.shape[-1]}")
    if packed:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = x.chunk(2, dim=-1)
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Packed GeGLU variant adds a +1 bias to the linear half.
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    """Mixture-of-experts feed-forward block used inside each transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """Initialize one MoE feed-forward block."""
        super().__init__()
        param_dtype = _resolve_param_dtype(config.param_dtype)
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.packed_geglu = config.packed_geglu
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.torch_ops_batch = int(config.torch_ops_batch)
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=param_dtype
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size * 2 // self.world_size,
                ),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=param_dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MoE block and residual connection."""
        if x.dim() != 3:
            raise ValueError("MLPBlock expects batched 3D tensor input")
        batch_shape = x.shape[:-1]
        t = self.norm(x).reshape(-1, x.shape[-1])
        g = F.linear(
            t.float(),
            self.gate.weight.float(),
            self.gate.bias.float(),
        )
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        expert_weights = expert_weights / self.experts_per_token
        experts_per_token_eff = self.experts_per_token
        # Triton kernels are CUDA-only; auto-enable only on CUDA devices. MPS
        # and CPU fall back to the torch-ops path unless the user explicitly
        # opts in via OPF_MOE_TRITON=1.
        is_cuda_device = t.device.type == "cuda"
        use_triton = get_env_bool("OPF_MOE_TRITON", default=is_cuda_device)
        if use_triton:
            _require_triton()

        def _moe_chunk(
            t_chunk: torch.Tensor,
            expert_indices_chunk: torch.Tensor,
            expert_weights_chunk: torch.Tensor,
        ) -> torch.Tensor:
            if use_triton:
                n_tokens = t_chunk.shape[0]
                k = expert_indices_chunk.shape[1]
                expert_ids = expert_indices_chunk.reshape(-1)
                weights = expert_weights_chunk.reshape(-1)
                token_ids = torch.arange(
                    n_tokens, device=t_chunk.device
                ).repeat_interleave(k)
                sort_idx = torch.argsort(expert_ids)
                expert_ids_sorted = expert_ids[sort_idx]
                token_ids_sorted = token_ids[sort_idx]
                weights_sorted = weights[sort_idx]

                counts = torch.bincount(
                    expert_ids_sorted, minlength=self.num_experts
                ).to(torch.int32)
                offsets = torch.zeros_like(counts)
                if counts.numel() > 1:
                    offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
                a_packed = t_chunk[token_ids_sorted]
                w1 = self.mlp1_weight
                if a_packed.dtype != w1.dtype:
                    a_packed = a_packed.to(w1.dtype)
                w2 = self.mlp2_weight
                h_pre = grouped_matmul(
                    a_packed, w1, offsets, counts, out_dtype=w1.dtype
                )
                b1 = self.mlp1_bias[expert_ids_sorted]
                h_pre = h_pre + b1
                use_fused_w2 = (not self.packed_geglu) and get_env_bool(
                    "OPF_MOE_FUSED_SWIGLU_W2", default=True
                )
                if use_fused_w2:
                    if h_pre.dtype != w2.dtype:
                        h_pre = h_pre.to(w2.dtype)
                    o = grouped_swiglu_w2(
                        h_pre,
                        w2,
                        self.mlp2_bias,
                        offsets,
                        counts,
                        out_dtype=w2.dtype,
                        limit=self.swiglu_limit,
                    )
                else:
                    h = swiglu(
                        h_pre,
                        limit=self.swiglu_limit,
                        packed=self.packed_geglu,
                    )
                    if h.dtype != w2.dtype:
                        h = h.to(w2.dtype)
                    o = grouped_matmul(h, w2, offsets, counts, out_dtype=w2.dtype)
                    b2 = self.mlp2_bias[expert_ids_sorted]
                    o = o + b2
                if self.world_size > 1:
                    dist.all_reduce(o, op=dist.ReduceOp.SUM)
                if o.dtype != weights_sorted.dtype:
                    o = o.to(weights_sorted.dtype)
                o = o * weights_sorted[:, None]
                out_accum = torch.zeros(
                    (n_tokens, t_chunk.shape[1]),
                    device=t_chunk.device,
                    dtype=torch.float32,
                )
                out_accum.index_add_(0, token_ids_sorted, o.float())
                out_accum = out_accum * experts_per_token_eff
                return out_accum.to(x.dtype)
            # MLP #1
            mlp1_weight = self.mlp1_weight[expert_indices_chunk, ...]
            mlp1_bias = self.mlp1_bias[expert_indices_chunk, ...]
            mlp1_weight = mlp1_weight.float()
            mlp1_bias = mlp1_bias.float()
            t_expanded = (
                t_chunk.float()
                .unsqueeze(1)
                .expand(-1, expert_indices_chunk.shape[1], -1)
            )
            out = _batched_linear_with_parity(
                t_expanded,
                mlp1_weight,
                mlp1_bias,
            )
            out = swiglu(out, limit=self.swiglu_limit, packed=self.packed_geglu)

            # MLP #2
            mlp2_weight = self.mlp2_weight[expert_indices_chunk, ...]
            mlp2_bias = self.mlp2_bias[expert_indices_chunk, ...]
            mlp2_weight = mlp2_weight.float()
            mlp2_bias = mlp2_bias.float()
            out = out.float()
            out = _batched_linear_with_parity(
                out,
                mlp2_weight,
                mlp2_bias,
            )
            if self.world_size > 1:
                dist.all_reduce(out, op=dist.ReduceOp.SUM)

            # Weighted sum of experts (gate scales applied after MLP2).
            if out.dtype != expert_weights_chunk.dtype:
                out = out.to(expert_weights_chunk.dtype)
            out = torch.einsum("bec,be->bc", out, expert_weights_chunk)
            out = out * experts_per_token_eff
            return out.to(x.dtype)

        if use_triton:
            effective_batch = 0
        else:
            effective_batch = self.torch_ops_batch
        if effective_batch and t.shape[0] > effective_batch:
            chunks = []
            for start in range(0, t.shape[0], effective_batch):
                end = start + effective_batch
                chunks.append(
                    _moe_chunk(
                        t[start:end],
                        expert_indices[start:end],
                        expert_weights[start:end],
                    )
                )
            t = torch.cat(chunks, dim=0)
        else:
            t = _moe_chunk(t, expert_indices, expert_weights)
        t = t.reshape(*batch_shape, -1)
        return x + t


class TransformerBlock(torch.nn.Module):
    """One transformer layer composed of attention followed by MoE MLP."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        """Initialize one transformer block."""
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, device)
        self.mlp = MLPBlock(config, device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention then MLP for one transformer block."""
        x = self.attn(x, attention_mask=attention_mask)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    """Full OPF transformer model with embedding, blocks, and output head."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """Initialize the full transformer model."""
        super().__init__()
        param_dtype = _resolve_param_dtype(config.param_dtype)
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=param_dtype
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        output_size = (
            config.num_labels if config.num_labels is not None else config.vocab_size
        )
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            output_size,
            bias=False,
            device=device,
            dtype=param_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the transformer forward pass."""
        if x.dim() != 2:
            raise ValueError(
                "Transformer expects batched token ids with shape [batch, tokens]"
            )
        if attention_mask is not None:
            if attention_mask.shape != x.shape:
                raise ValueError(
                    "attention_mask shape mismatch: "
                    f"expected {tuple(x.shape)}, got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
        x = self.embedding(x)
        for block in self.block:
            x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        x = F.linear(
            x,
            self.unembedding.weight,
            None,
        )
        return x

    @staticmethod
    def from_checkpoint(
        path: str,
        device: str | torch.device = "cuda",
    ) -> "Transformer":
        """Construct and populate a transformer from a checkpoint directory."""
        if not isinstance(device, torch.device):
            device = torch.device(device)

        _configure_torch_math()

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
        checkpoint = Checkpoint(path, device)
        _validate_encoder_artifact_contract(json_config=json_config)
        field_names = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered_config = {k: v for k, v in json_config.items() if k in field_names}
        config = ModelConfig(**filtered_config)
        override_experts_per_token = os.environ.get("OPF_EXPERTS_PER_TOKEN")
        if override_experts_per_token:
            try:
                value = int(override_experts_per_token)
            except ValueError as exc:
                raise ValueError(
                    f"OPF_EXPERTS_PER_TOKEN must be an int (got {override_experts_per_token!r})"
                ) from exc
            if value <= 0:
                raise ValueError("OPF_EXPERTS_PER_TOKEN must be > 0")
            if value > int(config.num_experts):
                raise ValueError(
                    "OPF_EXPERTS_PER_TOKEN must be <= num_experts "
                    f"({value} > {config.num_experts})"
                )
            config.experts_per_token = value

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load weights
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Note: it would be more efficient to do sharding before upcasting from MXFP4,
            # but for simplicity we do it after.
            if "mlp1_weight" in name or "mlp1_bias" in name:
                # Shard on the output (2 * intermediate) dimension.
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank * 2 * per_rank_intermediate_size : (my_rank + 1)
                    * 2
                    * per_rank_intermediate_size,
                ]
            elif "mlp2_weight" in name:
                # Shard on the intermediate (input) dimension.
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                    ...,
                ]
            if param.data.shape != loaded_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch for {name!r}: expected {tuple(param.data.shape)}, "
                    f"got {tuple(loaded_tensor.shape)}"
                )
            param.data.copy_(loaded_tensor)

        return model
