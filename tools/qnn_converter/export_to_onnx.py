import argparse
import json
import math
import re
import itertools
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Type, Union

import onnx
import safetensors
import torch
from onnx import shape_inference
from torch import nn
from torch.nn.utils import skip_init
from tqdm import tqdm
from transformers import AutoTokenizer


profile_prune_ffn = False
prune_ffn_threshold = 1e-6
perform_prune_ffn = False


class ModelParams:
    has_qkv_bias: bool
    use_drelu: bool
    tie_embedding: bool

    n_layers: int
    vocab_size: int
    ffn_hidden_dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int

    rope_theta: float
    rms_norm_eps: float
    attention_mask_value: float

    fp16_attention_layers: List[int]
    fp16_ffn_layers: List[int]

    @property
    def embed_dim(self) -> int:
        return self.head_dim * self.n_heads

    @property
    def group_size(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


class Llama3_1_8B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = False

    n_layers = 32
    vocab_size = 128256
    ffn_hidden_dim = 14336
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 5e5
    rms_norm_eps = 1e-5
    attention_mask_value = -1e5

    fp16_attention_layers = []
    fp16_ffn_layers = []


class Llama3_2_1B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = True

    n_layers = 16
    vocab_size = 128256
    ffn_hidden_dim = 8192
    head_dim = 64
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 5e5
    rms_norm_eps = 1e-5
    attention_mask_value = -1e5

    fp16_attention_layers = []
    fp16_ffn_layers = []


class Llama2_7B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = False
    tie_embedding = False

    n_layers = 32
    vocab_size = 32000
    ffn_hidden_dim = 11008
    head_dim = 128
    n_heads = 32
    n_kv_heads = 32

    rope_theta = 10000.0
    rms_norm_eps = 1e-5
    attention_mask_value = -1e6

    fp16_attention_layers = []
    fp16_ffn_layers = []


class Mistral_7B_Params(ModelParams):
    has_qkv_bias = False
    use_drelu = True
    tie_embedding = False

    n_layers = 32
    vocab_size = 32000
    ffn_hidden_dim = 14336
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8

    rope_theta = 1e4
    rms_norm_eps = 1e-5
    attention_mask_value = -1e2

    fp16_attention_layers = []
    fp16_ffn_layers = []


class Qwen2_7B_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = True
    # use_drelu = False
    tie_embedding = False

    n_layers = 28
    vocab_size = 152064
    ffn_hidden_dim = 18944
    head_dim = 128
    n_heads = 28
    n_kv_heads = 4

    rope_theta = 1e4
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = [0, 27]
    fp16_ffn_layers = [27]


class Qwen2_0_5B_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 24
    vocab_size = 151936
    ffn_hidden_dim = 4864
    head_dim = 64
    n_heads = 14
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    fp16_attention_layers = [0, 1, 2, 10, 23]
    fp16_ffn_layers = [23]


class SmallThinker_3B_Params(ModelParams):
    has_qkv_bias = True
    use_drelu = False
    tie_embedding = True

    n_layers = 36
    vocab_size = 151936
    ffn_hidden_dim = 11008
    head_dim = 128
    n_heads = 16
    n_kv_heads = 2

    rope_theta = 1e6
    rms_norm_eps = 1e-6
    attention_mask_value = -5e4

    # fp16_attention_layers = list(range(36))
    # fp16_ffn_layers = list(range(36))
    fp16_attention_layers = [0, 27, 33]
    fp16_ffn_layers = [2, 3, 4, 29, 30, 31, 32, 33, 34, 35]


model_map: Dict[str, ModelParams] = {
    "mistral_7b": Mistral_7B_Params,
    "qwen2_7b": Qwen2_7B_Params,
    "qwen2_0.5b": Qwen2_0_5B_Params,
    "llama3_1_8b": Llama3_1_8B_Params,
    "llama3_2_1b": Llama3_2_1B_Params,
    "llama2_7b": Llama2_7B_Params,
    "smallthinker_3b": SmallThinker_3B_Params,
}


class GraphParams:
    batch_size: int
    cache_size: int
    context_size: int


class Batch1_Params(GraphParams):
    batch_size = 1
    cache_size = 1920
    context_size = 2048


class Batch4_Params(GraphParams):
    batch_size = 4
    cache_size = 1920
    context_size = 2048


class Batch8_Params(GraphParams):
    batch_size = 8
    cache_size = 1920
    context_size = 2048


class Batch16_Params(GraphParams):
    batch_size = 16
    cache_size = 1920
    context_size = 2048


class Batch32_Params(GraphParams):
    batch_size = 32
    cache_size = 1920
    context_size = 2048


class Batch128_Params(GraphParams):
    batch_size = 128
    cache_size = 1920
    context_size = 2048


if profile_prune_ffn:
    Batch128_Params.cache_size = 16384


graph_map: Dict[str, GraphParams] = {
    "batch_1": Batch1_Params,
    "batch_4": Batch4_Params,
    "batch_8": Batch8_Params,
    "batch_16": Batch16_Params,
    "batch_32": Batch32_Params,
    "batch_128": Batch128_Params,
}


parser = argparse.ArgumentParser()
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--model-folder", type=Path, required=True)
parser.add_argument("--model-name", choices=model_map.keys(), required=True)
parser.add_argument("--graph-name", choices=graph_map.keys(), required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--system-prompt-file", type=Path)
parser.add_argument("--prompt-file", type=Path, required=True)
parser.add_argument("--n-model-chunks", type=int, default=4)
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--max-n-tokens", type=int, required=True)
parser.add_argument("--output-folder", type=Path)
parser.add_argument("--fp16-lm-head", action="store_true")
args = parser.parse_args()

torch.manual_seed(42)
torch.set_num_threads(args.n_threads)
device = torch.device(args.device)


def export_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class ModelLoader:
    """Helper class to load weight tensors from safetensors"""

    def __init__(self, folder: Path):
        """Load tensors from safetensor files and create a mapping between tensor names and tensors"""

        self.tensor_map: Dict[str, torch.Tensor] = {}
        for model_shard_file in folder.glob("*.safetensors"):
            tensors = safetensors.safe_open(model_shard_file, "pt")
            for name in tensors.keys():
                self.tensor_map[name] = tensors.get_tensor(name)

    def contain(self, name: str) -> bool:
        return name in self.tensor_map

    def load(self, dest: Union[nn.Module, torch.Tensor], name: str, transposed: bool = False):
        """Look up tensor in tensor map and copy data to destination tensor"""

        tensor = self.tensor_map[name]

        target = None
        if isinstance(dest, nn.Module):
            target = dest.weight.data
        elif isinstance(dest, torch.Tensor):
            target = dest.data
        else:
            raise RuntimeError

        if transposed:
            tensor = tensor.T

        assert target.shape == tensor.shape, f"Expect {tuple(target.shape)}, got {tuple(tensor.shape)}"
        target.copy_(tensor.to(torch.float32))


class Monitor:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.history: List[float] = []

    def disable(self):
        self.enabled = False

    def track(self, values: torch.Tensor):
        assert len(values.shape) == 1
        self.history.extend(values.tolist())

    def track_norm(self, tensor: torch.Tensor):
        if self.enabled:
            self.track(tensor.norm(dim=-1))

    def track_absolute_max(self, tensor: torch.Tensor):
        if self.enabled:
            self.track(tensor.abs().max(dim=-1).values)

    def print(self, threshold: float = 100):
        def format_value(value: float):
            return f"{value:.1f}"

        def format_item(item: Tuple[int, float]):
            return f"({item[0]}, {item[1]:.1f})"

        items_top5 = sorted(enumerate(self.history), key=lambda item: -item[1])[:5]
        values_top5 = [value for _, value in items_top5]

        if values_top5[0] < threshold:
            return

        items = ", ".join(map(format_item, items_top5))
        values = ", ".join(map(format_value, values_top5))
        print(f"NOTE: {self.name}: [{values}] [{items}]")

    def disable_and_print(self):
        self.disable()
        self.print()


class QLinearMode(Enum):
    UNINITIALIZED = auto()
    TRACKING = auto()
    COMPUTED = auto()
    FINALIZED = auto()


class QLinear(nn.Linear):
    """
    QNN-style quantized linear layer.
    16bit per-tensor asymmetric quantization for activations.
    4bit per-channel symmetric quantization for weights.
    Fake quantization performs quantize+dequantize to mimic the effect of quantization.
    Actual computations still perform in FP32.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mode = QLinearMode.UNINITIALIZED

    @staticmethod
    def fake_quantize(x: torch.Tensor, bitwidth: int, symmetric: bool, per_channel: bool) -> torch.Tensor:
        """Quantize to int, and then dequantize back to float"""

        max_quant_value = 2 ** (bitwidth - 1) - 1

        if per_channel:
            assert symmetric
            scales = x.abs().max(dim=1).values / max_quant_value
            inverted_scales = 1 / scales
            quants = (x * inverted_scales[:, None]).round().clamp(-max_quant_value, max_quant_value)
            return quants * scales[:, None]
        else:
            if symmetric:
                scale = x.abs().max() / max_quant_value
                inverted_scale = 1 / scale
                quants = (x * inverted_scale).round().clamp(-max_quant_value, max_quant_value)
                return quants * scale
            else:
                x_max = x.max()
                x_min = x.min()
                zero = (x_max + x_min) / 2
                scale = (x_max - zero) / max_quant_value

                inverted_scale = 1 / scale
                quants = ((x - zero) * inverted_scale).round().clamp(-max_quant_value, max_quant_value)

        return quants * scale + zero

    def fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quantize(x, bitwidth=16, symmetric=False, per_channel=False)

    @property
    def uninitialized(self):
        return self.mode == QLinearMode.UNINITIALIZED

    @property
    def tracking(self):
        return self.mode == QLinearMode.TRACKING

    @property
    def computed(self):
        return self.mode == QLinearMode.COMPUTED

    @property
    def finalized(self):
        return self.mode == QLinearMode.FINALIZED

    def enable_impl(self):
        pass

    def enable(self):
        assert self.uninitialized
        self.enable_impl()
        self.mode = QLinearMode.TRACKING

    def compute_impl(self):
        self.original_weight = self.weight.clone()
        self.weight.data.copy_(self.fake_quantize(self.weight, bitwidth=4, symmetric=True, per_channel=True))

    def compute(self):
        assert self.tracking
        self.compute_impl()
        self.mode = QLinearMode.COMPUTED

    def finalize_impl(self):
        self.weight.data = self.original_weight

    def finalize(self):
        assert self.computed
        self.finalize_impl()
        self.mode = QLinearMode.FINALIZED

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.computed:
            x = self.fake_quantize_activation(x)
        return super().forward(x)


# TODO: Only support linear RoPE now
class LlamaRoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def compute_embeds(
        dim: int, start_position: int, end_position: int, theta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert dim % 2 == 0
        inv_freq = 1 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(start=start_position, end=end_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq).to(device)  # (n_positions, dim / 2)
        return (freqs.cos(), freqs.sin())

    def forward(self, x: torch.Tensor, rope_embeds: Tuple[torch.Tensor]) -> torch.Tensor:
        rope_cos = rope_embeds[0]  # (batch_size, dim / 2)
        rope_sin = rope_embeds[1]  # (batch_size, dim / 2)

        head_dim = x.shape[-1]
        x0 = x[:, : head_dim // 2]
        x1 = x[:, head_dim // 2 :]
        return torch.cat((x0 * rope_cos - x1 * rope_sin, x0 * rope_sin + x1 * rope_cos), dim=-1)


class LlamaRMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32, device=device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).reciprocal()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) * self.weight


class LlamaAttentionCore(nn.Module):
    def __init__(self, layer_id: int, n_kv_heads: int, group_size: int, context_size: int):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.group_size = group_size
        self.context_size = context_size
        self.neg_inf = torch.tensor([-1e4], dtype=torch.float32, device=device)

        self.score_monitors = [Monitor(f"attn_{layer_id}_head_{i}.score") for i in range(n_kv_heads)]

    def forward(
        self,
        queries: List[torch.Tensor],
        keys: List[torch.Tensor],
        key_t_caches: List[torch.Tensor],
        values: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        attn_bias: torch.Tensor,
        kq_scale: float,
    ) -> torch.Tensor:
        n_heads = len(queries)
        assert len(keys) == self.n_kv_heads
        assert self.n_kv_heads * self.group_size == n_heads

        batch_size = values[0].shape[0]
        cache_size = value_caches[0].shape[0]
        kv_pad_size = self.context_size - batch_size - cache_size

        scaled_keys = []
        scaled_values = []
        head_outs = []
        for i in range(self.n_kv_heads):
            scaled_key = keys[i] * kq_scale
            scaled_value = torch.max(values[i], self.neg_inf)  # No scaling
            scaled_keys.append(scaled_key)
            scaled_values.append(scaled_value)

            scaled_key_t = scaled_key.transpose(0, 1)
            padded_key_t = nn.functional.pad(scaled_key_t, (0, kv_pad_size))
            padded_value = nn.functional.pad(scaled_value, (0, 0, 0, kv_pad_size))

            all_keys_t = torch.cat((key_t_caches[i], padded_key_t), dim=-1)
            all_values = torch.cat((value_caches[i], padded_value), dim=-2)

            for query in queries[i * self.group_size : (i + 1) * self.group_size]:
                score = query @ all_keys_t
                self.score_monitors[i].track_absolute_max(score)

                score = score + attn_bias
                score = score.softmax(dim=-1)
                out = score @ all_values
                head_outs.append(out)

        out = torch.cat(head_outs, dim=-1)
        return out, scaled_keys, scaled_values


class LlamaAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        has_qkv_bias: bool,
        rms_norm_eps: float,
        linear_class: Type[nn.Linear],
    ):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.has_qkv_bias = has_qkv_bias

        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.head_dim = embed_dim // n_heads
        self.group_size = n_heads // n_kv_heads

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps)
        self.rope = LlamaRoPE()
        self.core = LlamaAttentionCore(
            layer_id=layer_id, n_kv_heads=n_kv_heads, group_size=self.group_size, context_size=context_size
        )

        self.q_heads = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_heads)
        ])
        self.k_heads = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_kv_heads)
        ])
        self.v_heads = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.head_dim,
                bias=has_qkv_bias,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_kv_heads)
        ])
        self.o_proj = skip_init(
            linear_class, in_features=embed_dim, out_features=embed_dim, bias=False, dtype=torch.float32, device=device
        )

        self.key_monitors = [Monitor(f"attn_{layer_id}_head_{i}.key") for i in range(n_kv_heads)]
        self.value_monitors = [Monitor(f"attn_{layer_id}_head_{i}.value") for i in range(n_kv_heads)]
        self.output_monitor = Monitor(f"attn_{layer_id}.output")

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, f"model.layers.{self.layer_id}.input_layernorm.weight")

        wq = torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32, device=device)
        loader.load(wq, f"model.layers.{self.layer_id}.self_attn.q_proj.weight")
        if self.has_qkv_bias:
            bq = torch.empty(self.embed_dim, dtype=torch.float32, device=device)
            loader.load(bq, f"model.layers.{self.layer_id}.self_attn.q_proj.bias")

        for i in range(self.n_heads):
            self.q_heads[i].weight.data.copy_(wq[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.q_heads[i].bias.data.copy_(bq[i * self.head_dim : (i + 1) * self.head_dim])

        wk = torch.empty(self.head_dim * self.n_kv_heads, self.embed_dim, dtype=torch.float32, device=device)
        loader.load(wk, f"model.layers.{self.layer_id}.self_attn.k_proj.weight")
        if self.has_qkv_bias:
            bk = torch.empty(self.head_dim * self.n_kv_heads, dtype=torch.float32, device=device)
            loader.load(bk, f"model.layers.{self.layer_id}.self_attn.k_proj.bias")

        for i in range(self.n_kv_heads):
            self.k_heads[i].weight.data.copy_(wk[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.k_heads[i].bias.data.copy_(bk[i * self.head_dim : (i + 1) * self.head_dim])

        wv = torch.empty(self.head_dim * self.n_kv_heads, self.embed_dim, dtype=torch.float32, device=device)
        loader.load(wv, f"model.layers.{self.layer_id}.self_attn.v_proj.weight")
        if self.has_qkv_bias:
            bv = torch.empty(self.head_dim * self.n_kv_heads, dtype=torch.float32, device=device)
            loader.load(bv, f"model.layers.{self.layer_id}.self_attn.v_proj.bias")

        for i in range(self.n_kv_heads):
            self.v_heads[i].weight.data.copy_(wv[i * self.head_dim : (i + 1) * self.head_dim, :])
            if self.has_qkv_bias:
                self.v_heads[i].bias.data.copy_(bv[i * self.head_dim : (i + 1) * self.head_dim])

        loader.load(self.o_proj, f"model.layers.{self.layer_id}.self_attn.o_proj.weight")

    def disable_monitors(self):
        for i in range(self.n_kv_heads):
            self.core.score_monitors[i].disable_and_print()
            self.key_monitors[i].disable_and_print()
            self.value_monitors[i].disable_and_print()

        self.output_monitor.disable_and_print()

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, embed_dim)
        key_t_caches: Tuple[torch.Tensor],  # Transposed keys: (head_dim, cache_size) * n_kv_heads
        value_caches: Tuple[torch.Tensor],  # (cache_size, head_dim) * n_kv_heads
        attn_bias: torch.Tensor,  # (batch_size, context_size)
        rope_embeds: Tuple[torch.Tensor],  # (batch_size, head_dim / 2) * 2
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Returns the attention output, keys and values of input x"""

        attn_input = self.norm(x)
        queries = [self.rope(q_head(attn_input), rope_embeds) for q_head in self.q_heads]
        keys = [self.rope(k_head(attn_input), rope_embeds) for k_head in self.k_heads]
        values = [v_head(attn_input) for v_head in self.v_heads]

        for key, monitor in zip(keys, self.key_monitors):
            monitor.track_norm(key)
        for value, monitor in zip(values, self.value_monitors):
            monitor.track_norm(value)

        out, scaled_keys, scaled_values = self.core(
            queries, keys, key_t_caches, values, value_caches, attn_bias, kq_scale=(1 / math.sqrt(self.head_dim))
        )

        out = self.o_proj(out)

        self.output_monitor.track_norm(out)

        return out + x, tuple(scaled_keys), tuple(scaled_values)


class LlamaFeedForward(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        linear_class: Type[nn.Linear],
        use_drelu: bool,
        n_shards: int = 1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim

        if perform_prune_ffn:
            with open(f"prune/ffn_{layer_id}.txt", "r") as f:
                self.mask = f.read().strip()
                assert len(self.mask) == ffn_hidden_dim
                ffn_hidden_dim = sum(1 for x in self.mask if x == "1")

        self.ffn_hidden_dim = ffn_hidden_dim
        self.n_shards = n_shards
        assert ffn_hidden_dim % n_shards == 0
        self.shard_dim = ffn_hidden_dim // n_shards
        self.use_drelu = use_drelu

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps)

        if use_drelu:
            self.relu = nn.ReLU()
        else:
            self.silu = nn.SiLU()

        self.gate_shards = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.shard_dim,
                bias=False,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_shards)
        ])
        self.up_shards = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=embed_dim,
                out_features=self.shard_dim,
                bias=False,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_shards)
        ])
        self.down_shards = nn.ModuleList([
            skip_init(
                linear_class,
                in_features=self.shard_dim,
                out_features=embed_dim,
                bias=False,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(n_shards)
        ])

        self.output_monitor = Monitor(f"ffn_{layer_id}.output")

        if profile_prune_ffn:
            self.n_activation_samples = 0
            self.activation_count = torch.zeros(ffn_hidden_dim, dtype=torch.int32, device=device)

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, f"model.layers.{self.layer_id}.post_attention_layernorm.weight")

        if perform_prune_ffn:
            original_ffn_hidden_dim = len(self.mask)
        else:
            original_ffn_hidden_dim = self.ffn_hidden_dim

        gate_proj = torch.empty((original_ffn_hidden_dim, self.embed_dim), dtype=torch.float32, device=device)
        up_proj = torch.empty((original_ffn_hidden_dim, self.embed_dim), dtype=torch.float32, device=device)
        down_proj = torch.empty((self.embed_dim, original_ffn_hidden_dim), dtype=torch.float32, device=device)

        loader.load(gate_proj, f"model.layers.{self.layer_id}.mlp.gate_proj.weight")
        loader.load(up_proj, f"model.layers.{self.layer_id}.mlp.up_proj.weight")
        loader.load(down_proj, f"model.layers.{self.layer_id}.mlp.down_proj.weight")

        if perform_prune_ffn:
            mask = torch.zeros(len(self.mask), dtype=torch.bool)
            for i, x in enumerate(self.mask):
                if x == "1":
                    mask[i] = True

            gate_proj = gate_proj[mask, :]
            up_proj = up_proj[mask, :]
            down_proj = down_proj[:, mask]
            print(f"FFN #{self.layer_id} removed {original_ffn_hidden_dim - self.ffn_hidden_dim} rows")

        for i in range(self.n_shards):
            self.gate_shards[i].weight.data.copy_(gate_proj[i * self.shard_dim : (i + 1) * self.shard_dim, :])
            self.up_shards[i].weight.data.copy_(up_proj[i * self.shard_dim : (i + 1) * self.shard_dim, :])
            self.down_shards[i].weight.data.copy_(down_proj[:, i * self.shard_dim : (i + 1) * self.shard_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn_input = self.norm(x)

        outs = []
        for i in range(self.n_shards):
            gate = self.gate_shards[i](ffn_input)
            if self.use_drelu:
                gate = self.relu(gate)
            else:
                gate = self.silu(gate)

            up = self.up_shards[i](ffn_input)
            if self.use_drelu:
                up = self.relu(up)

            out = gate * up

            if profile_prune_ffn:
                self.n_activation_samples += out.shape[0]
                self.activation_count.add_((out > prune_ffn_threshold).to(torch.int32).sum(dim=0))

            out = self.down_shards[i](out)
            outs.append(out)

        # Reduce like a binary tree
        k = 1
        while k < len(outs):
            for i in range(0, len(outs), 2 * k):
                outs[i] = outs[i] + outs[i + k]
            k *= 2
        out = outs[0]

        self.output_monitor.track_norm(out)

        return out + x


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        has_qkv_bias: bool,
        use_drelu: bool,
        attention_linear_class: Type[nn.Linear],
        ffn_linear_class: Type[nn.Linear],
    ):
        super().__init__()
        self.layer_id = layer_id

        self.attn = LlamaAttention(
            layer_id=layer_id,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            context_size=context_size,
            has_qkv_bias=has_qkv_bias,
            rms_norm_eps=rms_norm_eps,
            linear_class=attention_linear_class,
        )
        self.ffn = LlamaFeedForward(
            layer_id=layer_id,
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            rms_norm_eps=rms_norm_eps,
            linear_class=ffn_linear_class,
            use_drelu=use_drelu,
        )

    def load_weights(self, loader: ModelLoader):
        self.attn.load_weights(loader)
        self.ffn.load_weights(loader)

    def forward(
        self,
        x: torch.Tensor,
        key_t_caches: Tuple[torch.Tensor],
        value_caches: Tuple[torch.Tensor],
        attn_bias: torch.Tensor,
        rope_embeds: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        attn_out, keys, values = self.attn(x, key_t_caches, value_caches, attn_bias, rope_embeds)
        ffn_out = self.ffn(attn_out)
        return ffn_out, keys, values


class Sample(NamedTuple):
    inputs: Tuple[torch.Tensor]
    outputs: Tuple[torch.Tensor]


class KVCache:
    def init_kv_caches(self, start_layer_id: int, end_layer_id: int, n_kv_heads: int, head_dim: int, cache_size: int):
        self.start_layer_id = start_layer_id
        self.end_layer_id = end_layer_id
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_size = cache_size

        self.kv_cache_position = 0
        self.saved_kv: Optional[List[torch.Tensor]] = None

        self.key_t_cache_data = torch.zeros(
            size=(self.n_kv_layers, n_kv_heads, head_dim, cache_size), dtype=torch.float32, device=device
        )
        self.value_cache_data = torch.zeros(
            size=(self.n_kv_layers, n_kv_heads, cache_size, head_dim), dtype=torch.float32, device=device
        )

    @property
    def n_kv_layers(self) -> int:
        return self.end_layer_id - self.start_layer_id

    @property
    def n_kv_caches(self) -> int:
        return self.n_kv_layers * self.n_kv_heads

    @property
    def kv_cache_names(self) -> List[str]:
        return [
            *[
                f"layer_{i}_key_t_cache_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
            *[
                f"layer_{i}_value_cache_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
        ]

    @property
    def kv_names(self) -> List[str]:
        return [
            *[
                f"layer_{i}_key_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
            *[
                f"layer_{i}_value_{j}"
                for i in range(self.start_layer_id, self.end_layer_id)
                for j in range(self.n_kv_heads)
            ],
        ]

    @property
    def key_t_caches(self) -> Tuple[torch.Tensor]:
        return tuple(self.key_t_cache_data[i, j] for i in range(self.n_kv_layers) for j in range(self.n_kv_heads))

    @property
    def value_caches(self) -> Tuple[torch.Tensor]:
        return tuple(self.value_cache_data[i, j] for i in range(self.n_kv_layers) for j in range(self.n_kv_heads))

    @property
    def kv_caches(self) -> Tuple[torch.Tensor]:
        return self.key_t_caches + self.value_caches

    def update_kv_caches(self, kv: Tuple[torch.Tensor]):
        assert len(kv) == 2 * self.n_kv_caches
        assert all(kv[0].shape == tensor.shape for tensor in kv)

        batch_size, _ = kv[0].shape
        beg = self.kv_cache_position
        end = self.kv_cache_position + batch_size
        self.kv_cache_position += batch_size
        assert end <= self.cache_size

        keys = kv[: self.n_kv_caches]
        values = kv[self.n_kv_caches :]
        for cache, key in zip(self.key_t_caches, keys):
            cache[:, beg:end] = key.transpose(0, 1)
        for cache, value in zip(self.value_caches, values):
            cache[beg:end, :] = value

    def reset_kv_cache_position(self, position: int):
        self.kv_cache_position = position
        for cache in self.key_t_caches:
            cache[:, position:] = 0
        for cache in self.value_caches:
            cache[position:, :] = 0


class ExportableModule(nn.Module):
    def __init__(self, module_type: str, start_layer_id: int, end_layer_id: int):
        super().__init__()
        self.module_type = module_type
        self.start_layer_id = start_layer_id
        self.end_layer_id = end_layer_id
        self.saved_samples: List[Sample] = []

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        raise RuntimeError

    def disable_monitors(self):
        pass


class LlamaAttentionWithGateProj(ExportableModule, KVCache):
    def __init__(
        self,
        layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        has_qkv_bias: bool,
        rms_norm_eps: float,
        ffn_hidden_dim: int,
        cache_size: int,
        attention_linear_class: Type[nn.Linear],
        ffn_linear_class: Type[nn.Linear],
    ):
        super().__init__("attn_gate", layer_id, layer_id + 1)
        self.layer_id = layer_id
        self.n_kv_heads = n_kv_heads

        self.attn = LlamaAttention(
            layer_id=layer_id,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            has_qkv_bias=has_qkv_bias,
            rms_norm_eps=rms_norm_eps,
            linear_class=attention_linear_class,
        )
        self.ffn_norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps)
        self.gate_proj = skip_init(
            ffn_linear_class,
            in_features=embed_dim,
            out_features=ffn_hidden_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.relu = nn.ReLU()

        self.init_kv_caches(
            start_layer_id=layer_id,
            end_layer_id=(layer_id + 1),
            n_kv_heads=n_kv_heads,
            head_dim=(embed_dim // n_heads),
            cache_size=cache_size,
        )

    @property
    def input_names(self) -> List[str]:
        return ["x", "attn_bias", "rope_embed_cos", "rope_embed_sin", *self.kv_cache_names]

    @property
    def output_names(self) -> List[str]:
        return ["attn_out", "ffn_input", "gate_out", *self.kv_names]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "attn_out", "ffn_input", "gate_out", "rope_embed_cos", "rope_embed_sin"]

    def load_weights(self, loader: ModelLoader):
        self.attn.load_weights(loader)
        loader.load(self.ffn_norm, f"model.layers.{self.layer_id}.post_attention_layernorm.weight")
        loader.load(self.gate_proj, f"model.layers.{self.layer_id}.mlp.gate_proj.weight")

    def disable_monitors(self):
        self.attn.disable_monitors()

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        return (last_outputs[0], attn_bias, rope_embeds[0], rope_embeds[1], *self.kv_caches)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        rope_embed_cos: torch.Tensor,
        rope_embed_sin: torch.Tensor,
        *caches: torch.Tensor,
    ) -> torch.Tensor:
        key_t_caches = caches[: self.n_kv_heads]
        value_caches = caches[self.n_kv_heads :]

        attn_out, keys, values = self.attn(x, key_t_caches, value_caches, attn_bias, (rope_embed_cos, rope_embed_sin))

        ffn_input = self.ffn_norm(attn_out)
        gate_out = self.relu(self.gate_proj(ffn_input))

        return attn_out, ffn_input, gate_out, *keys, *values


class LlamaUpDownProj(ExportableModule):
    def __init__(self, layer_id: int, embed_dim: int, ffn_hidden_dim: int, linear_class: Type[nn.Linear]):
        super().__init__("up_down", layer_id, layer_id + 1)
        self.layer_id = layer_id

        self.up_proj = skip_init(
            linear_class,
            in_features=embed_dim,
            out_features=ffn_hidden_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.down_proj = skip_init(
            linear_class,
            in_features=ffn_hidden_dim,
            out_features=embed_dim,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.relu = nn.ReLU()

    @property
    def input_names(self) -> List[str]:
        return ["attn_out", "ffn_input", "gate_out"]

    @property
    def output_names(self) -> List[str]:
        return ["ffn_out"]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["attn_out", "ffn_input", "gate_out", "ffn_out"]

    def load_weights(self, loader: ModelLoader):
        loader.load(self.up_proj, f"model.layers.{self.layer_id}.mlp.up_proj.weight")
        loader.load(self.down_proj, f"model.layers.{self.layer_id}.mlp.down_proj.weight")

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        return last_outputs[:3]

    def forward(self, attn_out: torch.Tensor, ffn_input: torch.Tensor, gate_out: torch.Tensor) -> torch.Tensor:
        up_out = self.relu(self.up_proj(ffn_input))
        out = gate_out * up_out
        out = self.down_proj(out)
        return attn_out + out


class LlamaModelChunk(ExportableModule, KVCache):
    """A model chunk consists of consecutive transformer layers"""

    def __init__(
        self,
        start_layer_id: int,
        end_layer_id: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        context_size: int,
        ffn_hidden_dim: int,
        rms_norm_eps: float,
        has_qkv_bias: bool,
        use_drelu: bool,
        cache_size: int,
        fp16_attention_layers: List[int],
        fp16_ffn_layers: List[int],
    ):
        super().__init__("transformers", start_layer_id, end_layer_id)
        self.n_kv_heads = n_kv_heads

        self.layers = nn.ModuleList([
            LlamaTransformer(
                layer_id=layer_id,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                context_size=context_size,
                ffn_hidden_dim=ffn_hidden_dim,
                rms_norm_eps=rms_norm_eps,
                has_qkv_bias=has_qkv_bias,
                use_drelu=use_drelu,
                # attention_linear_class=(nn.Linear if layer_id in fp16_attention_layers else GPTQLinear),
                # ffn_linear_class=(nn.Linear if layer_id in fp16_ffn_layers else GPTQLinear),
                attention_linear_class=nn.Linear,
                ffn_linear_class=nn.Linear,
            )
            for layer_id in range(start_layer_id, end_layer_id)
        ])

        self.init_kv_caches(
            start_layer_id=start_layer_id,
            end_layer_id=end_layer_id,
            n_kv_heads=n_kv_heads,
            head_dim=(embed_dim // n_heads),
            cache_size=cache_size,
        )

    @property
    def n_layers(self) -> int:
        return self.end_layer_id - self.start_layer_id

    @property
    def layer_ids(self) -> range:
        return range(self.start_layer_id, self.end_layer_id)

    @property
    def input_names(self) -> List[str]:
        return ["x", "attn_bias", "rope_embed_cos", "rope_embed_sin", *self.kv_cache_names]

    @property
    def output_names(self) -> List[str]:
        return ["out", *self.kv_names]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "out", "rope_embed_cos", "rope_embed_sin"]

    def load_weights(self, loader: ModelLoader):
        for layer in self.layers:
            layer.load_weights(loader)

    def disable_monitors(self):
        for layer in self.layers:
            layer.attn.disable_monitors()
            layer.ffn.output_monitor.disable_and_print()

    def enable_quantization(self):
        for module in self.modules():
            if isinstance(module, QLinear):
                module.enable()

    def compute_quantization(self):
        modules = list(item for item in self.named_modules() if isinstance(item[1], QLinear))
        for name, module in (bar := tqdm(modules)):
            bar.set_description(f'{module.__class__.__name__} "{name}"')
            module.compute()

    def finalize_quantization(self):
        for module in self.modules():
            if isinstance(module, QLinear):
                module.finalize()

    def get_inputs(
        self, last_outputs: Tuple[torch.Tensor], attn_bias: torch.Tensor, rope_embeds: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        return (last_outputs[0], attn_bias, rope_embeds[0], rope_embeds[1], *self.kv_caches)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        rope_embed_cos: torch.Tensor,
        rope_embed_sin: torch.Tensor,
        *caches: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        assert len(caches) == 2 * self.n_kv_heads * self.n_layers

        new_keys = []
        new_values = []
        for i, layer in enumerate(self.layers):
            key_t_caches = caches[: self.n_kv_caches][i * self.n_kv_heads : (i + 1) * self.n_kv_heads]
            value_caches = caches[self.n_kv_caches :][i * self.n_kv_heads : (i + 1) * self.n_kv_heads]

            x, keys, values = layer(x, key_t_caches, value_caches, attn_bias, (rope_embed_cos, rope_embed_sin))

            new_keys.extend(keys)
            new_values.extend(values)

        return x, *new_keys, *new_values


class LlamaInputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        self.embedding = skip_init(
            nn.Embedding, num_embeddings=vocab_size, embedding_dim=embed_dim, dtype=torch.float32, device=device
        )

    def load_weights(self, loader: ModelLoader):
        loader.load(self.embedding, "model.embed_tokens.weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LlamaOutputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, rms_norm_eps: float):
        super().__init__()

        self.norm = LlamaRMSNorm(embed_dim=embed_dim, eps=rms_norm_eps)
        self.output_proj = skip_init(
            nn.Linear, in_features=embed_dim, out_features=vocab_size, bias=False, dtype=torch.float32, device=device
        )

        self.saved_samples: List[Sample] = []

    def load_weights(self, loader: ModelLoader):
        loader.load(self.norm, "model.norm.weight")

        # Models use tie embedding do not have lm_head.weight
        if loader.contain("lm_head.weight"):
            loader.load(self.output_proj, "lm_head.weight")
        else:
            loader.load(self.output_proj, "model.embed_tokens.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    @property
    def input_names(self) -> List[str]:
        return ["x"]

    @property
    def output_names(self) -> List[str]:
        return ["logits"]

    @property
    def dtype_preserved_io_names(self) -> List[str]:
        return ["x", "logits"]


class LlamaModel(nn.Module):
    """Wrapper for Llama model"""

    def __init__(
        self,
        model_folder: Path,
        model_params: ModelParams,
        graph_params: GraphParams,
        model_chunks: List[ExportableModule],
    ) -> None:
        super().__init__()
        self.model_folder = model_folder
        self.model_params = model_params
        self.graph_params = graph_params

        self.loader = ModelLoader(model_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)

        self.input_embedding = LlamaInputEmbedding(vocab_size=model_params.vocab_size, embed_dim=model_params.embed_dim)

        self.output_embedding = LlamaOutputEmbedding(
            vocab_size=model_params.vocab_size, embed_dim=model_params.embed_dim, rms_norm_eps=model_params.rms_norm_eps
        )

        self.model_chunks: List[LlamaModelChunk] = nn.ModuleList(model_chunks)

        self.system_prompt_length = 0
        self.first_prompt = True
        self.last_logits: Optional[torch.Tensor] = None
        self.logits: List[float] = []

    def load_weights(self):
        self.input_embedding.load_weights(self.loader)
        self.output_embedding.load_weights(self.loader)
        for model_chunk in self.model_chunks:
            model_chunk.load_weights(self.loader)

        for param in self.parameters():
            param.requires_grad = False

    @property
    def kv_cache_position(self) -> int:
        return self.model_chunks[0].kv_cache_position

    def tokenize(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt").flatten()

    def get_attention_bias(self, batch_size: int) -> torch.Tensor:
        # Unmask tokens in KV caches
        attn_bias = torch.full(
            size=(batch_size, self.graph_params.context_size),
            fill_value=self.model_params.attention_mask_value,
            dtype=torch.float32,
            device=device,
        )
        attn_bias[:, : self.kv_cache_position] = 0

        # Causal mask
        cache_size = self.graph_params.cache_size
        attn_bias[:, cache_size : cache_size + batch_size] = torch.full(
            size=(batch_size, batch_size),
            fill_value=self.model_params.attention_mask_value,
            dtype=torch.float32,
            device=device,
        ).triu(diagonal=1)

        return attn_bias

    def update_logits(self, input_ids: torch.Tensor, logits: torch.Tensor):
        logits = logits.log_softmax(dim=-1)

        if self.last_logits is not None:
            self.logits.append(self.last_logits[input_ids[0]].item())

        self.last_logits = logits[-1]

        logits = logits[:-1]
        input_ids = input_ids[1:].reshape(-1, 1)
        self.logits.extend(logits.gather(dim=1, index=input_ids).flatten().tolist())

    @property
    def perplexity(self) -> float:
        return torch.exp(-torch.tensor(self.logits).mean()).item()

    @torch.no_grad()
    def eval_batch(
        self, input_ids: torch.Tensor, save_samples: bool = False, save_kv: bool = False, update_logits: bool = False
    ):
        x = self.input_embedding(input_ids)

        (batch_size,) = input_ids.shape
        last_outputs = (x,)
        attn_bias = self.get_attention_bias(batch_size)
        rope_embeds = LlamaRoPE.compute_embeds(
            dim=self.model_params.head_dim,
            start_position=self.kv_cache_position,
            end_position=self.kv_cache_position + batch_size,
            theta=self.model_params.rope_theta,
        )

        for model_chunk in self.model_chunks:
            inputs = model_chunk.get_inputs(last_outputs, attn_bias, rope_embeds)
            outputs = model_chunk(*inputs)

            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            if save_samples:
                model_chunk.saved_samples.append(Sample(inputs, outputs))

            if isinstance(model_chunk, KVCache):
                kv = outputs[-2 * model_chunk.n_kv_caches :]
                model_chunk.update_kv_caches(kv)

                if save_kv:
                    assert model_chunk.saved_kv is None
                    model_chunk.saved_kv = kv

            last_outputs = outputs

        x = last_outputs[0]
        logits = self.output_embedding(x)
        if save_samples:
            self.output_embedding.saved_samples.append(Sample((x,), (logits,)))

        if update_logits:
            # x = last_outputs[0]
            # logits = self.output_embedding(x)
            self.update_logits(input_ids, logits)

    def eval_system_prompt(self, prompt: str):
        assert self.first_prompt
        self.first_prompt = False

        input_ids = self.tokenize(prompt)
        self.system_prompt_length = len(input_ids)
        print(f"System prompt: {len(input_ids)} tokens: {input_ids.tolist()} {repr(self.tokenizer.decode(input_ids))}")

        self.eval_batch(input_ids, save_kv=True)

    def eval_prompt(self, prompt: str, batch_size: int, save_samples: bool = True, max_n_tokens: Optional[int] = None):
        input_ids = self.tokenize(prompt, add_special_tokens=self.first_prompt)

        self.first_prompt = False

        n_tokens = min(input_ids.nelement(), self.graph_params.cache_size - self.kv_cache_position, max_n_tokens)
        n_tokens -= n_tokens % batch_size
        print(f"Prompt: {len(input_ids)} tokens, truncated to {n_tokens} tokens")

        for i in tqdm(range(0, n_tokens, batch_size)):
            self.eval_batch(input_ids[i : i + batch_size], save_samples=save_samples, update_logits=True)

    def reset(self):
        self.first_prompt = self.system_prompt_length == 0
        self.last_logits = None
        self.logits = []

        for model_chunk in self.model_chunks:
            model_chunk.reset_kv_cache_position(self.system_prompt_length)

    def disable_monitors(self):
        for model_chunk in self.model_chunks:
            model_chunk.disable_monitors()

    def dump_config_template(self) -> dict:
        return {
            "model_parameters": {
                "n_layers": self.model_params.n_layers,
                "vocab_size": self.model_params.vocab_size,
                "embed_dim": self.model_params.embed_dim,
                "ffn_hidden_dim": self.model_params.ffn_hidden_dim,
                "head_dim": self.model_params.head_dim,
                "n_kv_heads": self.model_params.n_kv_heads,
                "rope_theta": self.model_params.rope_theta,
                "rms_norm_eps": self.model_params.rms_norm_eps,
                "attention_mask_value": self.model_params.attention_mask_value,
                "tie_embedding": self.model_params.tie_embedding,
            },
            "qnn_parameters": {"n_hvx_threads": 4},
            "graphs": [
                {
                    "type": model_chunk.module_type,
                    "start_layer_id": model_chunk.start_layer_id,
                    "end_layer_id": model_chunk.end_layer_id,
                    "batch_size": self.graph_params.batch_size,
                    "cache_size": self.graph_params.cache_size,
                    "context_size": self.graph_params.context_size,
                    # To be filled later
                    "graph_name": "",
                    "model_path": f"{args.model_name}_{i}.bin",
                    "kv_path_format": f"kv/layer_{{layer_id}}_{{kv_type}}_{{head_id}}.raw",
                    "kv_size": self.system_prompt_length,
                    "x_name": "x",
                    "out_name": "out",
                }
                for i, model_chunk in enumerate(self.model_chunks)
            ],
            "embeddings": [{
                "graph_name": "",
                "model_path": "lm_head.bin",
                "batch_size": self.graph_params.batch_size,
                "x_name": "x",
                "out_name": "logits",
            }],
        }


class OutputEmbeddingExporter:
    """Export a model chunk to ONNX model, quantization calibration data and configurations"""

    def __init__(
        self,
        graph_name: str,
        model_chunk: Union[LlamaOutputEmbedding, LlamaInputEmbedding],
        use_fp16: bool,
        output_folder: Path,
    ):
        self.graph_name = graph_name
        self.model_chunk = model_chunk
        self.use_fp16 = use_fp16
        self.output_folder = output_folder

    @torch.no_grad()
    def export_onnx_model(self):
        onnx_model_folder = self.output_folder / "onnx_model"
        onnx_model_folder.mkdir(parents=True, exist_ok=True)

        onnx_model_path = onnx_model_folder / f"{self.graph_name}.onnx"
        torch.onnx.export(
            model=self.model_chunk,
            args=self.model_chunk.saved_samples[0].inputs,
            f=str(onnx_model_path),
            input_names=self.model_chunk.input_names,
            output_names=self.model_chunk.output_names,
        )

        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
        self.onnx_model = shape_inference.infer_shapes(onnx_model)

    def export_io_spec(self):
        def dump_info_list(io_type: Literal["in", "out"], names: List[str], tensors: List[torch.Tensor]) -> List[dict]:
            return [
                {
                    "name": name,
                    "type": io_type,
                    "dtype": "int64" if name == "input_ids" else "float32",
                    "preserve_dtype": name in self.model_chunk.dtype_preserved_io_names,
                    "shape": list(tensor.shape),
                }
                for name, tensor in zip(names, tensors)
            ]

        io_spec = [
            *dump_info_list("in", self.model_chunk.input_names, self.model_chunk.saved_samples[0].inputs),
            *dump_info_list("out", self.model_chunk.output_names, self.model_chunk.saved_samples[0].outputs),
        ]

        export_json(io_spec, self.output_folder / f"{self.graph_name}.io.json")

    def export_quantization_config(self):
        class Encoding(NamedTuple):
            category: Literal["activation", "param"]
            bitwidth: int
            dtype: Literal["float", "int"]

        encoding_map: Dict[str, Encoding] = {}

        def update_encoding(name: str, encoding: Encoding):
            """Set encoding for name. Only update if the bitwidth is larger than the previous setting."""

            if name not in encoding_map or encoding.bitwidth > encoding_map[name].bitwidth:
                encoding_map[name] = encoding

        def encode_activation(node: str | onnx.ValueInfoProto, bitwidth: int):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("activation", bitwidth, "float"))

        def encode_output(node: onnx.NodeProto, bitwidth: int):
            for name in node.output:
                update_encoding(name, Encoding("activation", bitwidth, "float"))

        def encode_param(node: str | onnx.NodeProto, bitwidth: Encoding, dtype: Literal["float", "int"]):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("param", bitwidth, dtype))

        def match(target: str | onnx.NodeProto | onnx.TensorProto | onnx.ValueInfoProto, pattern: str):
            if not isinstance(target, str):
                target = target.name
            return re.fullmatch(pattern, target) is not None

        graph = self.onnx_model.graph

        # Inputs
        encode_activation("x", 32)
        encode_activation("attn_bias", 16)

        # KV cache: FP16
        for node in graph.input:
            if match(node, "layer_[0-9]+_(key_t|value)_cache_[0-9]+"):
                encode_activation(node, 16)
        for node in graph.output:
            if match(node, "layer_[0-9]+_(key|value)_[0-9]+"):
                encode_activation(node, 16)

        # Attention core: FP16
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/attn/core.*"):
                encode_output(node, 16)

        # Residual connection: FP32
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn|ffn/down_proj)/Add(_[0-9]+|)"):
                encode_output(node, 32)

        # RMSNorm: FP32
        for node in graph.initializer:
            if match(node, "layers\.[0-9]+\.(attn|ffn)\.norm\.weight"):
                encode_param(node, 32, "float")
        for node in graph.node:
            # print(node)
            if match(node, "/layers\.[0-9]+/(attn|ffn)/norm.*"):
                encode_output(node, 32)
            elif match(node, "/norm.*"):
                encode_output(node, 32)

        if self.use_fp16:
            print("NOTE: Use FP16 for output embedding.")
            for node in graph.initializer:
                encode_param(node, 16, "float")
            for node in graph.node:
                encode_output(node, 16)

        # Generate config
        config = {
            "version": "0.6.1",
            "quantizer_args": {
                "activation_bitwidth": 16,
                "param_bitwidth": 4,
                "dtype": "int",
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf",
            },
            "activation_encodings": {},
            "param_encodings": {},
        }

        for name, encoding in sorted(encoding_map.items(), key=lambda item: item[0]):
            config[f"{encoding.category}_encodings"][name] = [{
                "bitwidth": encoding.bitwidth,
                "dtype": encoding.dtype,
                "is_symmetric": str(encoding.category == "param"),
            }]

        export_json(config, self.output_folder / f"{self.graph_name}.encodings")

    def export_sample_inputs(self):
        input_list = []
        for i, samples in enumerate(self.model_chunk.saved_samples):
            data_folder = self.output_folder / "data" / str(i)
            data_folder.mkdir(parents=True, exist_ok=True)

            tensor_paths = []
            for name, tensor in zip(self.model_chunk.input_names, samples.inputs, strict=True):
                output_path = data_folder / f"{name}.raw"
                tensor.cpu().numpy().tofile(output_path)
                tensor_paths.append(f"{name}:={output_path}")

            input_list.append(" ".join(tensor_paths))

        with open(self.output_folder / "input_list.txt", "w") as f:
            f.write("\n".join(input_list))

    def export_saved_kv(self):
        kv_folder = self.output_folder / "kv"
        kv_folder.mkdir(parents=True, exist_ok=True)

        for name, tensor in zip(self.model_chunk.kv_names, self.model_chunk.saved_kv, strict=True):
            tensor.cpu().numpy().tofile(kv_folder / f"{name}.raw")

    def export(self):
        self.export_onnx_model()
        self.export_io_spec()
        self.export_quantization_config()
        self.export_sample_inputs()
        if isinstance(self.model_chunk, KVCache) and self.model_chunk.saved_kv is not None:
            self.export_saved_kv()


class ModelChunkExporter:
    """Export a model chunk to ONNX model, quantization calibration data and configurations"""

    def __init__(
        self, graph_name: str, model_chunk: LlamaModelChunk, fp16_overrides: Dict[str, List[int]], output_folder: Path
    ):
        self.graph_name = graph_name
        self.model_chunk = model_chunk
        self.fp16_overrides = fp16_overrides
        self.output_folder = output_folder

    @torch.no_grad()
    def export_onnx_model(self):
        onnx_model_folder = self.output_folder / "onnx_model"
        onnx_model_folder.mkdir(parents=True, exist_ok=True)

        onnx_model_path = onnx_model_folder / f"{self.graph_name}.onnx"
        torch.onnx.export(
            model=self.model_chunk,
            args=self.model_chunk.saved_samples[0].inputs,
            f=str(onnx_model_path),
            input_names=self.model_chunk.input_names,
            output_names=self.model_chunk.output_names,
        )

        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
        self.onnx_model = shape_inference.infer_shapes(onnx_model)

    def export_io_spec(self):
        def dump_info_list(io_type: Literal["in", "out"], names: List[str], tensors: List[torch.Tensor]) -> List[dict]:
            return [
                {
                    "name": name,
                    "type": io_type,
                    "dtype": "float32",
                    "preserve_dtype": name in self.model_chunk.dtype_preserved_io_names,
                    "shape": list(tensor.shape),
                }
                for name, tensor in zip(names, tensors)
            ]

        io_spec = [
            *dump_info_list("in", self.model_chunk.input_names, self.model_chunk.saved_samples[0].inputs),
            *dump_info_list("out", self.model_chunk.output_names, self.model_chunk.saved_samples[0].outputs),
        ]

        export_json(io_spec, self.output_folder / f"{self.graph_name}.io.json")

    def export_quantization_config(self):
        class Encoding(NamedTuple):
            category: Literal["activation", "param"]
            bitwidth: int
            dtype: Literal["float", "int"]

        encoding_map: Dict[str, Encoding] = {}

        def update_encoding(name: str, encoding: Encoding):
            """Set encoding for name. Only update if the bitwidth is larger than the previous setting."""

            if name not in encoding_map or encoding.bitwidth > encoding_map[name].bitwidth:
                encoding_map[name] = encoding

        def encode_activation(node: Union[str, onnx.ValueInfoProto], bitwidth: int):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("activation", bitwidth, "float"))

        def encode_output(node: onnx.NodeProto, bitwidth: int):
            for name in node.output:
                update_encoding(name, Encoding("activation", bitwidth, "float"))

        def encode_param(node: Union[str, onnx.NodeProto], bitwidth: Encoding, dtype: Literal["float", "int"]):
            if not isinstance(node, str):
                node = node.name
            update_encoding(node, Encoding("param", bitwidth, dtype))

        def match(target: Union[str, onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto], pattern: str):
            if not isinstance(target, str):
                target = target.name
            return re.fullmatch(pattern, target) is not None

        graph = self.onnx_model.graph

        # Inputs
        encode_activation("x", 32)
        encode_activation("attn_bias", 16)

        # KV cache: FP16
        for node in graph.input:
            if match(node, "layer_[0-9]+_(key_t|value)_cache_[0-9]+"):
                encode_activation(node, 16)
        for node in graph.output:
            if match(node, "layer_[0-9]+_(key|value)_[0-9]+"):
                encode_activation(node, 16)

        # Attention core: FP16
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/attn/core.*"):
                encode_output(node, 16)

        # Manually specified FP16 attention/FFN layers
        for layer_type, layer_id_list in self.fp16_overrides.items():
            for layer_id in layer_id_list:
                if not (self.model_chunk.start_layer_id <= layer_id < self.model_chunk.end_layer_id):
                    continue

                # NOTE: Layer ids in an ONNX model are always started from 0
                count = 0
                index = layer_id - self.model_chunk.start_layer_id
                for node in graph.initializer:
                    if match(node, f"layers\.{index}\.{layer_type}.*"):
                        count += 1
                        encode_param(node, 16, "float")
                for node in graph.node:
                    if match(node, f"/layers\.{index}/{layer_type}.*"):
                        count += 1
                        encode_output(node, 16)
                print(f'Override {count} nodes in layer "{layer_type}_{layer_id}" to FP16')

        # Residual connection: FP32
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn|ffn/down_proj)/Add(_[0-9]+|)"):
                encode_output(node, 32)

        # RMSNorm: FP32
        for node in graph.initializer:
            if match(node, "layers\.[0-9]+\.(attn|ffn)\.norm\.weight"):
                encode_param(node, 32, "float")
        for node in graph.node:
            if match(node, "/layers\.[0-9]+/(attn|ffn)/norm.*"):
                encode_output(node, 32)

        # Generate config
        config = {
            "version": "0.6.1",
            "quantizer_args": {
                "activation_bitwidth": 16,
                "param_bitwidth": 4,
                "dtype": "int",
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf",
            },
            "activation_encodings": {},
            "param_encodings": {},
        }

        for name, encoding in sorted(encoding_map.items(), key=lambda item: item[0]):
            config[f"{encoding.category}_encodings"][name] = [{
                "bitwidth": encoding.bitwidth,
                "dtype": encoding.dtype,
                "is_symmetric": str(encoding.category == "param"),
            }]

        export_json(config, self.output_folder / f"{self.graph_name}.encodings")

    def export_sample_inputs(self):
        input_list = []
        for i, samples in enumerate(self.model_chunk.saved_samples):
            data_folder = self.output_folder / "data" / str(i)
            data_folder.mkdir(parents=True, exist_ok=True)

            tensor_paths = []
            for name, tensor in zip(self.model_chunk.input_names, samples.inputs):
                output_path = data_folder / f"{name}.raw"
                tensor.cpu().numpy().tofile(output_path)
                tensor_paths.append(f"{name}:={output_path}")

            input_list.append(" ".join(tensor_paths))

        with open(self.output_folder / "input_list.txt", "w") as f:
            f.write("\n".join(input_list))

    def export_saved_kv(self):
        kv_folder = self.output_folder / "kv"
        kv_folder.mkdir(parents=True, exist_ok=True)

        for name, tensor in zip(self.model_chunk.kv_names, self.model_chunk.saved_kv):
            tensor.cpu().numpy().tofile(kv_folder / f"{name}.raw")

    def export(self):
        self.export_onnx_model()
        self.export_io_spec()
        self.export_quantization_config()
        self.export_sample_inputs()
        if isinstance(self.model_chunk, KVCache) and self.model_chunk.saved_kv is not None:
            self.export_saved_kv()


print("Creating model...")

model_params: ModelParams = model_map[args.model_name]()
graph_params: GraphParams = graph_map[args.graph_name]()

assert model_params.n_layers % args.n_model_chunks == 0
n_layers_per_model_chunk = model_params.n_layers // args.n_model_chunks

model_chunks = [
    LlamaModelChunk(
        start_layer_id=i,
        end_layer_id=(i + n_layers_per_model_chunk),
        embed_dim=model_params.embed_dim,
        n_heads=model_params.n_heads,
        n_kv_heads=model_params.n_kv_heads,
        context_size=graph_params.context_size,
        ffn_hidden_dim=model_params.ffn_hidden_dim,
        rms_norm_eps=model_params.rms_norm_eps,
        has_qkv_bias=model_params.has_qkv_bias,
        use_drelu=model_params.use_drelu,
        cache_size=graph_params.cache_size,
        fp16_attention_layers=model_params.fp16_attention_layers,
        fp16_ffn_layers=model_params.fp16_ffn_layers,
    )
    for i in range(0, model_params.n_layers, n_layers_per_model_chunk)
]


model = LlamaModel(
    model_folder=args.model_folder, model_params=model_params, graph_params=graph_params, model_chunks=model_chunks
)

print("Loading model weights...")
model.load_weights()

if args.system_prompt_file is not None:
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()
    model.eval_system_prompt(system_prompt)

with open(args.prompt_file, "r") as f:
    prompt = f.read()


def eval_prompt(save_samples: bool = False):
    model.eval_prompt(
        prompt=prompt,
        batch_size=model.graph_params.batch_size,
        save_samples=save_samples,
        max_n_tokens=args.max_n_tokens,
    )
    print(f"Perplexity: {model.perplexity:.4f}")


eval_prompt(save_samples=True)

if profile_prune_ffn:
    for name, module in model.named_modules():
        if isinstance(module, LlamaFeedForward):
            zero_count = (module.activation_count == 0).sum().item()
            dead_ratio = zero_count / module.activation_count.shape[0]
            sparsity = 1 - module.activation_count.sum().item() / (module.ffn_hidden_dim * module.n_activation_samples)

            print(
                name, f"dead_ratio={100 * dead_ratio:.1f}%", f"sparsity={100 * sparsity:.1f}%", module.activation_count
            )

            n_save = zero_count % 256
            mask = (module.activation_count == 0).tolist()
            for i in range(len(mask)):
                mask[i] = not mask[i]
                if not mask[i] and n_save > 0:
                    mask[i] = True
                    n_save -= 1

            layer_id = module.layer_id
            text = "".join(map(str, map(int, mask)))
            with open(f"prune/ffn_{layer_id}.txt", "w") as f:
                f.write(text)

model.disable_monitors()


if args.output_folder is None:
    exit(0)

args.output_folder.mkdir(parents=True, exist_ok=True)

config_template = model.dump_config_template()
for graph_info in config_template["graphs"]:
    graph_info["graph_name"] = args.graph_name
for embedding_info in config_template["embeddings"]:
    embedding_info["graph_name"] = args.graph_name
export_json(config_template, args.output_folder / f"config_{args.graph_name}.json")

for i, model_chunk in enumerate(model.model_chunks):
    print(f'Exporting "model_chunk_{i}"...')

    output_folder = args.output_folder / f"model_chunk_{i}" / args.graph_name
    output_folder.mkdir(parents=True, exist_ok=True)

    exporter = ModelChunkExporter(
        graph_name=args.graph_name,
        model_chunk=model_chunk,
        fp16_overrides={"attn": model_params.fp16_attention_layers, "ffn": model_params.fp16_ffn_layers},
        output_folder=output_folder,
    )
    exporter.export()

print("Exporting output embedding...")
output_folder = args.output_folder / "output_embedding" / args.graph_name
output_folder.mkdir(parents=True, exist_ok=True)

exporter = OutputEmbeddingExporter(
    graph_name=args.graph_name,
    model_chunk=model.output_embedding,
    use_fp16=args.fp16_lm_head,
    output_folder=output_folder,
)
exporter.export()
