"""
Connector diagnostics for debugging embedding pipeline issues.

Last Updated: 2026-01-20

Implements Gemini's recommended diagnostic checks for the LTX-2
Embeddings1DConnector explosion issue:
1. Weight magnitudes - std > 1.0 indicates massive weights bug
2. RoPE values - should be in [-1, 1] range
3. Token segmentation - text vs register tokens analyzed separately
4. Per-stage per-dim range tracking

Reference: Gemini's analysis of the 3000x Block 0 explosion
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _compute_tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """Compute basic tensor statistics."""
    t_float = t.float()
    return {
        "mean": t_float.mean().item(),
        "std": t_float.std().item(),
        "min": t_float.min().item(),
        "max": t_float.max().item(),
        "abs_max": t_float.abs().max().item(),
    }


def _compute_per_dim_range(t: torch.Tensor) -> float:
    """
    Compute per-dimension range metric.

    For tensor [B, T, D], compute:
    - Mean per dimension: [D]
    - Range of means: max - min

    This is the key metric that explodes from 0.52 to 780.
    """
    if t.ndim == 3:  # [B, T, D]
        per_dim_mean = t.float().mean(dim=(0, 1))  # [D]
    elif t.ndim == 2:  # [T, D]
        per_dim_mean = t.float().mean(dim=0)  # [D]
    else:
        return 0.0

    return (per_dim_mean.max() - per_dim_mean.min()).item()


@dataclass
class ConnectorDiagnostics:
    """Structured diagnostics from connector forward pass.

    Implements Gemini's recommended checks for Block 0 explosion debugging.
    """

    # GEMINI CHECK 1: Weight magnitudes (std > 1.0 is suspicious)
    weight_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # GEMINI CHECK 2: RoPE values (should be in [-1, 1])
    rope_cos_range: tuple[float, float] = (0.0, 0.0)
    rope_sin_range: tuple[float, float] = (0.0, 0.0)

    # GEMINI CHECK 3: Token segmentation (text vs registers separately)
    text_tokens_stats: dict[str, float] = field(default_factory=dict)
    register_tokens_stats: dict[str, float] = field(default_factory=dict)
    num_text_tokens: int = 0
    num_register_tokens: int = 0

    # Per-stage per-dim range tracking (the explosion metric)
    per_dim_range_by_stage: dict[str, float] = field(default_factory=dict)

    # Additional useful stats
    input_stats: dict[str, float] = field(default_factory=dict)
    output_stats: dict[str, float] = field(default_factory=dict)

    # Block-by-block breakdown
    block_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def save(self, output_path: Path) -> None:
        """Save diagnostics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = asdict(self)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved connector diagnostics to {output_path}")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CONNECTOR DIAGNOSTICS SUMMARY",
            "=" * 60,
            "",
            "INTERFACE STATISTICS (tensor entering caption_projection):",
        ]

        # Show output stats first - these are critical for centering investigation
        if self.output_stats:
            out_mean = self.output_stats.get("mean", 0)
            out_std = self.output_stats.get("std", 0)
            out_range = self.output_stats.get("per_dim_range", 0)
            mean_flag = " ⚠️  BIASED!" if abs(out_mean) > 1.0 else ""
            lines.extend([
                f"  Output mean: {out_mean:.4f}{mean_flag}",
                f"  Output std: {out_std:.4f}",
                f"  Output per-dim range: {out_range:.2f}",
            ])
        if self.input_stats:
            in_mean = self.input_stats.get("mean", 0)
            in_std = self.input_stats.get("std", 0)
            lines.extend([
                f"  Input mean: {in_mean:.4f}",
                f"  Input std: {in_std:.4f}",
            ])

        lines.extend([
            "",
            "PER-DIM RANGE BY STAGE (key explosion metric):",
        ])

        for stage, value in sorted(self.per_dim_range_by_stage.items()):
            flag = " ⚠️  EXPLODED!" if value > 100 else ""
            lines.append(f"  {stage}: {value:.2f}{flag}")

        lines.extend([
            "",
            "TOKEN SEGMENTATION:",
            f"  Text tokens: {self.num_text_tokens}",
            f"  Register tokens: {self.num_register_tokens}",
        ])

        if self.text_tokens_stats:
            lines.append(f"  Text per-dim range: {self.text_tokens_stats.get('per_dim_range', 0):.4f}")
        if self.register_tokens_stats:
            lines.append(f"  Register per-dim range: {self.register_tokens_stats.get('per_dim_range', 0):.4f}")

        lines.extend([
            "",
            "ROPE VALUES:",
            f"  cos range: [{self.rope_cos_range[0]:.4f}, {self.rope_cos_range[1]:.4f}]",
            f"  sin range: [{self.rope_sin_range[0]:.4f}, {self.rope_sin_range[1]:.4f}]",
        ])

        if self.weight_stats:
            lines.extend(["", "WEIGHT MAGNITUDES (std > 1.0 suspicious):"])
            for name, stats in sorted(self.weight_stats.items()):
                flag = " ⚠️  HIGH STD!" if stats.get("std", 0) > 1.0 else ""
                lines.append(f"  {name}: std={stats.get('std', 0):.4f}{flag}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def check_for_anomalies(self) -> list[str]:
        """
        Run Gemini's recommended checks and return list of warnings.

        Returns:
            List of warning messages for detected anomalies
        """
        warnings = []

        # Gemini Check 1: Weight magnitudes
        for name, stats in self.weight_stats.items():
            if stats.get("std", 0) > 1.0:
                warnings.append(
                    f"MASSIVE WEIGHT: {name} has std={stats['std']:.4f} (expected < 1.0)"
                )
            if stats.get("abs_max", 0) > 10.0:
                warnings.append(
                    f"LARGE WEIGHT: {name} has abs_max={stats['abs_max']:.4f} (may cause explosion)"
                )

        # Gemini Check 2: RoPE values
        cos_min, cos_max = self.rope_cos_range
        sin_min, sin_max = self.rope_sin_range

        if abs(cos_min) > 1.5 or abs(cos_max) > 1.5:
            warnings.append(
                f"BROKEN ROPE: cos range [{cos_min:.4f}, {cos_max:.4f}] (expected [-1, 1])"
            )
        if abs(sin_min) > 1.5 or abs(sin_max) > 1.5:
            warnings.append(
                f"BROKEN ROPE: sin range [{sin_min:.4f}, {sin_max:.4f}] (expected [-1, 1])"
            )

        # Gemini Check 3: Token segmentation
        if self.register_tokens_stats:
            reg_mean = abs(self.register_tokens_stats.get("mean", 0))
            if reg_mean > 5.0:
                warnings.append(
                    f"REGISTER BIAS: registers have mean={reg_mean:.4f} (unexpectedly large)"
                )

            reg_per_dim = self.register_tokens_stats.get("per_dim_range", 0)
            if reg_per_dim > 10.0:
                warnings.append(
                    f"REGISTER EXPLOSION: registers have per_dim_range={reg_per_dim:.4f}"
                )

        if self.text_tokens_stats:
            text_per_dim = self.text_tokens_stats.get("per_dim_range", 0)
            if text_per_dim > 5.0:
                warnings.append(
                    f"TEXT TOKEN ISSUE: text tokens have per_dim_range={text_per_dim:.4f}"
                )

        # Check per-stage explosion
        for stage, value in self.per_dim_range_by_stage.items():
            if value > 100:
                warnings.append(
                    f"STAGE EXPLOSION: {stage} has per_dim_range={value:.2f} (expected < 10)"
                )

        return warnings


class ConnectorDiagnosticsCollector:
    """
    Collects diagnostics from connector forward pass using hooks.

    Usage:
        collector = ConnectorDiagnosticsCollector()
        collector.attach_hooks(connector)

        # Run forward pass
        output = connector(hidden_states, attention_mask)

        # Collect diagnostics
        diagnostics = collector.collect()
        diagnostics.save(output_dir / "connector_diagnostics.json")
    """

    def __init__(self):
        self._hooks: list[Any] = []
        self._activations: dict[str, torch.Tensor] = {}
        self._rope_values: dict[str, torch.Tensor] = {}
        self._connector: Optional[nn.Module] = None
        self._input_tensor: Optional[torch.Tensor] = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._num_text_tokens: int = 0
        self._num_register_tokens: int = 0

    def attach_hooks(self, connector: nn.Module) -> None:
        """Attach forward hooks to connector for diagnostics collection."""
        self._connector = connector

        # Hook to capture input to connector
        def input_hook(module, args, kwargs):
            if args:
                self._input_tensor = args[0].detach().clone()
                self._activations["input"] = self._input_tensor
            if len(args) > 1 and args[1] is not None:
                mask = args[1].detach().clone()
                self._attention_mask = mask
                # Count text tokens from attention mask
                # Mask format: 0=valid, -10000=padding
                if mask.ndim == 4:
                    mask_2d = mask.squeeze(1).squeeze(1)  # [B, T]
                else:
                    mask_2d = mask
                self._num_text_tokens = int((mask_2d >= -9000.0).sum().item())

        # Hook to capture output from connector
        def output_hook(module, args, kwargs, output):
            if isinstance(output, tuple):
                self._output_tensor = output[0].detach().clone()
            else:
                self._output_tensor = output.detach().clone()
            self._activations["final_output"] = self._output_tensor

        # Register hooks on connector
        handle1 = connector.register_forward_pre_hook(input_hook, with_kwargs=True)
        handle2 = connector.register_forward_hook(output_hook, with_kwargs=True)
        self._hooks.extend([handle1, handle2])

        # Hook each transformer block
        if hasattr(connector, "transformer_blocks"):
            blocks = connector.transformer_blocks
            if isinstance(blocks, nn.ModuleList):
                for i, block in enumerate(blocks):
                    self._attach_block_hooks(block, f"block_{i}")

    def _attach_block_hooks(self, block: nn.Module, block_name: str) -> None:
        """Attach hooks to a single transformer block."""

        # Capture output after attention
        attn1 = getattr(block, "attn1", None)
        if attn1 is not None and isinstance(attn1, nn.Module):
            def attn_hook(module, args, kwargs, output, name=block_name):
                self._activations[f"{name}_attn_output"] = output.detach().clone()

            handle = attn1.register_forward_hook(attn_hook, with_kwargs=True)
            self._hooks.append(handle)

        # Capture output after feed-forward AND internals
        ff = getattr(block, "ff", None)
        if ff is not None and isinstance(ff, nn.Module):
            # Hook FF input
            def ff_input_hook(module, args, kwargs, name=block_name):
                if args:
                    self._activations[f"{name}_ff_input"] = args[0].detach().clone()

            handle = ff.register_forward_pre_hook(ff_input_hook, with_kwargs=True)
            self._hooks.append(handle)

            # Hook FF output
            def ff_hook(module, args, kwargs, output, name=block_name):
                self._activations[f"{name}_ff_output"] = output.detach().clone()

            handle = ff.register_forward_hook(ff_hook, with_kwargs=True)
            self._hooks.append(handle)

            # Hook internals of FF (GELUApprox and final linear)
            self._attach_ff_internal_hooks(ff, block_name)

        # Capture block output (after residuals)
        def block_output_hook(module, args, kwargs, output, name=block_name):
            self._activations[f"{name}_output"] = output.detach().clone()

        handle = block.register_forward_hook(block_output_hook, with_kwargs=True)
        self._hooks.append(handle)

    def _attach_ff_internal_hooks(self, ff: nn.Module, block_name: str) -> None:
        """Attach hooks to FeedForward internal components."""
        # FF structure: Sequential(GELUApprox, Identity, Linear)
        net = getattr(ff, "net", None)
        if net is None or not isinstance(net, nn.Sequential):
            return

        # Hook GELUApprox output (after GELU, before final linear)
        if len(net) >= 1:
            gelu_approx = net[0]
            if isinstance(gelu_approx, nn.Module):
                # Hook GELUApprox input (before projection)
                def gelu_input_hook(module, args, kwargs, name=block_name):
                    if args:
                        self._activations[f"{name}_gelu_input"] = args[0].detach().clone()

                handle = gelu_approx.register_forward_pre_hook(gelu_input_hook, with_kwargs=True)
                self._hooks.append(handle)

                # Hook GELUApprox output (after GELU)
                def gelu_output_hook(module, args, kwargs, output, name=block_name):
                    self._activations[f"{name}_gelu_output"] = output.detach().clone()

                handle = gelu_approx.register_forward_hook(gelu_output_hook, with_kwargs=True)
                self._hooks.append(handle)

                # Also hook the linear projection inside GELUApprox
                proj = getattr(gelu_approx, "proj", None)
                if proj is not None and isinstance(proj, nn.Module):
                    def proj_output_hook(module, args, kwargs, output, name=block_name):
                        self._activations[f"{name}_proj_output_preGELU"] = output.detach().clone()

                    handle = proj.register_forward_hook(proj_output_hook, with_kwargs=True)
                    self._hooks.append(handle)

        # Hook final linear output
        if len(net) >= 3:
            final_linear = net[2]
            if isinstance(final_linear, nn.Linear):
                def final_linear_hook(module, args, kwargs, output, name=block_name):
                    self._activations[f"{name}_ff_final_linear_output"] = output.detach().clone()

                handle = final_linear.register_forward_hook(final_linear_hook, with_kwargs=True)
                self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def collect(self) -> ConnectorDiagnostics:
        """Collect all diagnostics into a ConnectorDiagnostics object."""
        diag = ConnectorDiagnostics()

        # Collect weight statistics
        if self._connector is not None:
            diag.weight_stats = self._collect_weight_stats()

            # Collect RoPE statistics if available
            diag.rope_cos_range, diag.rope_sin_range = self._collect_rope_stats()

            # Count registers
            num_registers = getattr(self._connector, "num_learnable_registers", None)
            if isinstance(num_registers, int):
                diag.num_register_tokens = num_registers
                # Compute text tokens from input sequence length minus registers
                if self._input_tensor is not None:
                    total_tokens = self._input_tensor.shape[1]
                    # Note: after register insertion, all tokens become "valid"
                    diag.num_text_tokens = total_tokens - diag.num_register_tokens

        # Per-stage per-dim range (the key explosion metric)
        diag.per_dim_range_by_stage = self._collect_per_dim_ranges()

        # Token segmentation stats
        diag.text_tokens_stats, diag.register_tokens_stats = self._collect_token_segment_stats()

        # Input/output stats
        if self._input_tensor is not None:
            diag.input_stats = _compute_tensor_stats(self._input_tensor)
            diag.input_stats["per_dim_range"] = _compute_per_dim_range(self._input_tensor)

        if self._output_tensor is not None:
            diag.output_stats = _compute_tensor_stats(self._output_tensor)
            diag.output_stats["per_dim_range"] = _compute_per_dim_range(self._output_tensor)

        # Block-by-block breakdown
        diag.block_stats = self._collect_block_stats()

        return diag

    def _collect_weight_stats(self) -> dict[str, dict[str, float]]:
        """Collect weight statistics from all connector layers."""
        stats = {}

        if self._connector is None:
            return stats

        for name, param in self._connector.named_parameters():
            if "weight" in name:
                stats[name] = _compute_tensor_stats(param.data)

        return stats

    def _collect_rope_stats(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Collect RoPE cos/sin range statistics."""
        cos_range = (0.0, 0.0)
        sin_range = (0.0, 0.0)

        # Try to get RoPE values from connector's last forward pass
        # This is tricky since RoPE is computed inside the forward pass
        # We'll need to capture it via hooks or recompute

        if self._connector is not None and self._input_tensor is not None:
            # Recompute RoPE to get the values
            try:
                from llm_dit.encoders.embeddings_connector import precompute_freqs_cis

                connector = self._connector
                input_tensor = self._input_tensor

                # Get connector attributes with type guards
                inner_dim = getattr(connector, "inner_dim", None)
                theta = getattr(connector, "positional_embedding_theta", None)
                max_pos = getattr(connector, "positional_embedding_max_pos", None)
                num_heads = getattr(connector, "num_attention_heads", None)
                rope_type = getattr(connector, "rope_type", None)
                use_double = getattr(connector, "use_double_precision_rope", None)

                # Check all required attributes exist and have correct types
                if not all([
                    isinstance(inner_dim, int),
                    isinstance(theta, (int, float)),
                    isinstance(max_pos, list),
                    isinstance(num_heads, int),
                    rope_type is not None,
                    isinstance(use_double, bool),
                ]):
                    logger.warning("Connector missing RoPE attributes")
                    return cos_range, sin_range

                # Cast to correct types (already validated above)
                from llm_dit.encoders.embeddings_connector import RopeType

                inner_dim_int: int = int(inner_dim)  # type: ignore[arg-type]
                theta_float: float = float(theta)  # type: ignore[arg-type]
                max_pos_list: list[int] = list(max_pos)  # type: ignore[arg-type]
                num_heads_int: int = int(num_heads)  # type: ignore[arg-type]
                use_double_bool: bool = bool(use_double)  # type: ignore[arg-type]
                rope_type_cast: RopeType = rope_type  # type: ignore[assignment]

                seq_len = input_tensor.shape[1]
                indices_grid = torch.arange(
                    seq_len,
                    dtype=torch.float32,
                    device=input_tensor.device,
                )
                indices_grid = indices_grid[None, None, :]

                freqs_cis = precompute_freqs_cis(
                    indices_grid=indices_grid,
                    dim=inner_dim_int,
                    out_dtype=input_tensor.dtype,
                    theta=theta_float,
                    max_pos=max_pos_list,
                    num_attention_heads=num_heads_int,
                    rope_type=rope_type_cast,
                    use_double_precision=use_double_bool,
                )

                cos_freq, sin_freq = freqs_cis
                cos_range = (cos_freq.min().item(), cos_freq.max().item())
                sin_range = (sin_freq.min().item(), sin_freq.max().item())

            except Exception as e:
                logger.warning(f"Failed to compute RoPE stats: {e}")

        return cos_range, sin_range

    def _collect_per_dim_ranges(self) -> dict[str, float]:
        """Collect per-dimension range for each activation stage."""
        ranges = {}

        for stage_name, tensor in self._activations.items():
            ranges[stage_name] = _compute_per_dim_range(tensor)

        return ranges

    def _collect_token_segment_stats(
        self,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Collect separate statistics for text tokens vs register tokens."""
        text_stats = {}
        register_stats = {}

        # After connector, sequence is: [registers..., text_tokens...]
        # Registers fill in the padding positions at the start
        if self._output_tensor is not None and self._connector is not None:
            num_registers = getattr(self._connector, "num_learnable_registers", 0)
            seq_len = self._output_tensor.shape[1]
            num_text = seq_len - num_registers

            if num_registers > 0 and num_text > 0:
                # Registers are at the start (replacing padding)
                register_tensor = self._output_tensor[:, :num_registers, :]
                text_tensor = self._output_tensor[:, num_registers:, :]

                register_stats = _compute_tensor_stats(register_tensor)
                register_stats["per_dim_range"] = _compute_per_dim_range(register_tensor)

                text_stats = _compute_tensor_stats(text_tensor)
                text_stats["per_dim_range"] = _compute_per_dim_range(text_tensor)

        return text_stats, register_stats

    def _collect_block_stats(self) -> dict[str, dict[str, float]]:
        """Collect per-block statistics."""
        block_stats = {}

        for stage_name, tensor in self._activations.items():
            if "block_" in stage_name:
                block_stats[stage_name] = _compute_tensor_stats(tensor)
                block_stats[stage_name]["per_dim_range"] = _compute_per_dim_range(tensor)

        return block_stats


def load_diagnostics(path: Path) -> ConnectorDiagnostics:
    """Load diagnostics from JSON file."""
    with open(path) as f:
        data = json.load(f)

    diag = ConnectorDiagnostics(
        weight_stats=data.get("weight_stats", {}),
        rope_cos_range=tuple(data.get("rope_cos_range", (0.0, 0.0))),
        rope_sin_range=tuple(data.get("rope_sin_range", (0.0, 0.0))),
        text_tokens_stats=data.get("text_tokens_stats", {}),
        register_tokens_stats=data.get("register_tokens_stats", {}),
        num_text_tokens=data.get("num_text_tokens", 0),
        num_register_tokens=data.get("num_register_tokens", 0),
        per_dim_range_by_stage=data.get("per_dim_range_by_stage", {}),
        input_stats=data.get("input_stats", {}),
        output_stats=data.get("output_stats", {}),
        block_stats=data.get("block_stats", {}),
    )

    return diag
