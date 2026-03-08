"""
DRY Configuration Consistency Tests.

Last Updated: 2026-03-06

Ensures parameters flow correctly through all configuration layers:
    config.toml -> Config dataclasses -> RuntimeConfig -> Backend configs

This prevents "disconnected settings" where a parameter exists in one layer
but is never wired through to actual usage.

Run with: uv run pytest tests/unit/test_dry_config.py -v
"""

import ast
import dataclasses
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# Paths to key files
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_EXAMPLE = PROJECT_ROOT / "config.toml.example"
CONFIG_PY = PROJECT_ROOT / "src" / "llm_dit" / "config.py"
CLI_PY = PROJECT_ROOT / "src" / "llm_dit" / "cli.py"
STARTUP_PY = PROJECT_ROOT / "src" / "llm_dit" / "startup.py"
MODEL_MANAGER_PY = PROJECT_ROOT / "src" / "llm_dit" / "model_manager.py"
API_BACKEND_PY = PROJECT_ROOT / "src" / "llm_dit" / "backends" / "api.py"


def parse_toml_keys(toml_path: Path) -> dict[str, set[str]]:
    """
    Parse TOML file and return keys grouped by section.

    Returns dict like:
        {
            "encoder": {"device", "dtype", "quantization", ...},
            "pipeline": {"device", "dtype", ...},
            ...
        }
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    sections = {}

    # Get default profile sections
    if "default" in data:
        profile = data["default"]
        for key, value in profile.items():
            if isinstance(value, dict):
                sections[key] = set(value.keys())
            else:
                sections.setdefault("_root", set()).add(key)

    return sections


def get_dataclass_fields(cls) -> set[str]:
    """Get field names from a dataclass."""
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set()


def extract_argparse_dests(filepath: Path) -> set[str]:
    """
    Extract argparse argument destinations from a Python file.

    Parses the AST to find add_argument calls and extracts the dest
    (either explicit or derived from the flag name).
    """
    dests = set()

    with open(filepath) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Look for .add_argument(...) calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                dest = None

                # Check for explicit dest= keyword
                for kw in node.keywords:
                    if kw.arg == "dest":
                        if isinstance(kw.value, ast.Constant):
                            dest = kw.value.value
                        break

                # If no explicit dest, derive from first positional arg
                if dest is None and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant):
                        flag = first_arg.value
                        if flag.startswith("--"):
                            # --my-flag -> my_flag
                            dest = flag[2:].replace("-", "_")

                if dest:
                    dests.add(dest)

    return dests


def check_string_in_file(filepath: Path, string: str) -> bool:
    """Check if a string appears in a file."""
    with open(filepath) as f:
        return string in f.read()


class TestTOMLToConfigDataclass:
    """Ensure config.toml.example parameters exist in Config dataclasses."""

    def test_encoder_params_exist_in_encoder_config(self):
        """All TOML [encoder] params should exist in EncoderConfig."""
        from llm_dit.config import EncoderConfig

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)
        toml_encoder_keys = toml_sections.get("encoder", set())
        dataclass_fields = get_dataclass_fields(EncoderConfig)

        missing = toml_encoder_keys - dataclass_fields
        assert not missing, (
            f"TOML encoder params not in EncoderConfig: {missing}. "
            f"Add these fields to EncoderConfig in config.py"
        )

    def test_generation_params_exist_in_generation_config(self):
        """All TOML [generation] params should exist in GenerationConfig."""
        from llm_dit.config import GenerationConfig

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)
        toml_keys = toml_sections.get("generation", set())
        dataclass_fields = get_dataclass_fields(GenerationConfig)

        missing = toml_keys - dataclass_fields
        assert not missing, (
            f"TOML generation params not in GenerationConfig: {missing}. "
            f"Add these fields to GenerationConfig in config.py"
        )

    def test_scheduler_params_exist_in_scheduler_config(self):
        """All TOML [scheduler] params should exist in SchedulerConfig."""
        from llm_dit.config import SchedulerConfig

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)
        toml_keys = toml_sections.get("scheduler", set())
        dataclass_fields = get_dataclass_fields(SchedulerConfig)

        missing = toml_keys - dataclass_fields
        assert not missing, (
            f"TOML scheduler params not in SchedulerConfig: {missing}. "
            f"Add these fields to SchedulerConfig in config.py"
        )

    def test_optimization_params_exist_in_optimization_config(self):
        """All TOML [optimization] params should exist in OptimizationConfig."""
        from llm_dit.config import OptimizationConfig

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)
        toml_keys = toml_sections.get("optimization", set())
        dataclass_fields = get_dataclass_fields(OptimizationConfig)

        missing = toml_keys - dataclass_fields
        assert not missing, (
            f"TOML optimization params not in OptimizationConfig: {missing}. "
            f"Add these fields to OptimizationConfig in config.py"
        )

    def test_rewriter_params_exist_in_rewriter_config(self):
        """All TOML [rewriter] params should exist in RewriterConfig."""
        from llm_dit.config import RewriterConfig

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)
        toml_keys = toml_sections.get("rewriter", set())
        dataclass_fields = get_dataclass_fields(RewriterConfig)

        missing = toml_keys - dataclass_fields
        assert not missing, (
            f"TOML rewriter params not in RewriterConfig: {missing}. "
            f"Add these fields to RewriterConfig in config.py"
        )


class TestCLIToRuntimeConfig:
    """Ensure CLI arguments are wired through _apply_cli_overrides."""

    def test_cli_args_have_runtime_config_fields(self):
        """CLI argument dests should be reachable on RuntimeConfig (via sub-configs or properties)."""
        from llm_dit.cli import RuntimeConfig

        cli_dests = extract_argparse_dests(CLI_PY)

        # These CLI args don't need RuntimeConfig fields (action-only or script-specific)
        excluded = {
            "config",  # Used to load config, not stored
            "config_name",  # Used to select modular config, not stored
            "profile",  # Used to select profile, not stored
            "lora",  # Parsed specially into lora_paths/lora_scales
            "loras",  # Alias for lora, same special handling
            "output",  # Script-specific, not in RuntimeConfig
            "prompts",  # Script-specific positional
            "version",  # Just prints version
            "embeddings_file",  # Script-specific (generate.py)
            "ltx2_output",  # CLI maps to config.ltx2.output_path
            "flux2_output",  # CLI maps to config.flux2.output_path
            "flux2_input_image",  # CLI maps to config.flux2.input_images
            "flux2_offload",  # Action flag (sets offload_between_stages=true)
            "flux2_no_offload",  # Action flag (sets offload_between_stages=false)
            "wan_output",  # CLI maps to config.wan.output_path
            # WAN CLI args access sub-config directly (backward-compat properties removed)
            "wan_humo_path",
            "wan_base_path",
            "wan_whisper_path",
            "wan_humo_variant",
            "wan_num_frames",
            "wan_fps",
            "wan_height",
            "wan_width",
            "wan_guidance_scale",
            "wan_audio_scale",
            "wan_steps",
            "wan_offload_mode",
            "torch_dtype",  # Legacy dtype flag, maps to config.encoder.dtype
            # CLI names that intentionally differ from sub-config field names
            # (wired correctly in _apply_cli_overrides)
            "template",  # -> config.generation.default_template
            "text_encoder_device",  # -> config.encoder.device
        }

        cli_dests_filtered = cli_dests - excluded

        # With composed RuntimeConfig, CLI args may exist as:
        # 1. Direct dataclass fields (e.g., host, port, debug)
        # 2. Backward-compat @property (e.g., attention_backend -> pytorch.attention_backend)
        # 3. Sub-config fields wired in _apply_cli_overrides
        # We verify by checking that each CLI arg is accessible as an attribute
        rc = RuntimeConfig()
        missing = []
        for dest in cli_dests_filtered:
            if not hasattr(rc, dest):
                missing.append(dest)

        assert not missing, (
            f"CLI args not accessible on RuntimeConfig: {missing}. "
            f"Either add a backward-compat @property or sub-config field."
        )


class TestKeyParameterWiring:
    """Ensure critical parameters are wired through to backend usage."""

    def test_hidden_layer_wired_to_api_backend_config(self):
        """hidden_layer must be passed to APIBackendConfig in model_manager.py."""
        # Check that model_manager.py passes hidden_layer to APIBackendConfig
        assert check_string_in_file(MODEL_MANAGER_PY, "hidden_layer=config.hidden_layer"), (
            "model_manager.py must pass hidden_layer to APIBackendConfig. "
            "Add: hidden_layer=config.hidden_layer"
        )

    def test_hidden_layer_in_api_backend_config(self):
        """APIBackendConfig must have hidden_layer field."""
        from llm_dit.backends.api import APIBackendConfig

        fields = get_dataclass_fields(APIBackendConfig)
        assert "hidden_layer" in fields, (
            "APIBackendConfig must have hidden_layer field. Add: hidden_layer: int = -2"
        )

    def test_rewriter_params_in_runtime_config(self):
        """All rewriter params should be accessible on RuntimeConfig (via sub-config)."""
        from llm_dit.config import RuntimeConfig

        rc = RuntimeConfig()

        # These are accessed via config.rewriter.* sub-config
        rewriter_params = [
            "use_api",
            "api_url",
            "api_model",
            "temperature",
            "top_p",
            "min_p",
            "max_tokens",
        ]

        missing = [p for p in rewriter_params if not hasattr(rc.rewriter, p)]
        assert not missing, (
            f"Rewriter params not accessible on config.rewriter: {missing}. "
            f"Add these fields to RewriterConfig in config.py"
        )


class TestStripUnknownKeys:
    """Config.from_dict() must tolerate stale TOML keys."""

    def test_unknown_keys_stripped_not_crash(self):
        """A stale TOML key in one section must not crash the entire config."""
        from llm_dit.config import Config

        data = {
            "default_pipeline": "none",
            "encoder": {"device": "cpu"},
            "ltx2": {"model_path": "/tmp/ltx2", "bogus_key": True},
            "flux2": {"model_path": "/tmp/flux2"},
        }
        config = Config.from_dict(data)
        assert config.ltx2.model_path == "/tmp/ltx2"
        assert config.flux2.model_path == "/tmp/flux2"
        assert not hasattr(config.ltx2, "bogus_key")

    def test_unknown_keys_logged(self, caplog):
        """Stripped keys should produce a warning."""
        import logging
        from llm_dit.config import Config

        data = {
            "encoder": {},
            "flux2": {"model_path": "/tmp/flux2", "nonexistent_field": 42},
        }
        with caplog.at_level(logging.WARNING, logger="llm_dit.config"):
            Config.from_dict(data)
        assert "nonexistent_field" in caplog.text

    def test_valid_keys_pass_through(self):
        """Valid keys are not stripped."""
        from llm_dit.config import Config

        data = {
            "encoder": {"device": "cpu", "hidden_layer": -3},
            "flux2": {"model_path": "/tmp/flux2", "encoder_path": "/tmp/enc"},
        }
        config = Config.from_dict(data)
        assert config.encoder.hidden_layer == -3
        assert config.flux2.encoder_path == "/tmp/enc"


class TestConfigSerialization:
    """Ensure Config.to_dict() includes all fields."""

    def test_encoder_config_to_dict_includes_hidden_layer(self):
        """Config.to_dict() must include encoder.hidden_layer."""
        from llm_dit.config import Config, EncoderConfig

        config = Config(encoder=EncoderConfig(hidden_layer=-3))
        data = config.to_dict()

        assert "hidden_layer" in data["encoder"], (
            "Config.to_dict() must include encoder.hidden_layer. Update to_dict() in config.py"
        )
        assert data["encoder"]["hidden_layer"] == -3


class TestDocumentation:
    """Ensure key parameters are documented."""

    def test_hidden_layer_in_claude_md(self):
        """hidden_layer should be documented in CLAUDE.md."""
        claude_md = PROJECT_ROOT / "CLAUDE.md"
        assert check_string_in_file(claude_md, "--hidden-layer"), (
            "--hidden-layer should be documented in CLAUDE.md CLI flags table"
        )

    def test_dry_principles_documented(self):
        """DRY configuration rules should be in CLAUDE.md."""
        claude_md = PROJECT_ROOT / "CLAUDE.md"
        assert check_string_in_file(claude_md, "configuration rules"), (
            "CLAUDE.md should have configuration rules section"
        )


# Convenience function for running from command line
def run_consistency_check():
    """
    Run all consistency checks and print results.

    Usage: uv run python -c "from tests.unit.test_dry_config import run_consistency_check; run_consistency_check()"
    """
    import sys

    print("=" * 60)
    print("DRY Configuration Consistency Check")
    print("=" * 60)

    errors = []

    # Check TOML -> Config dataclass
    print("\n[1/4] Checking TOML -> Config dataclasses...")
    try:
        from llm_dit.config import (
            EncoderConfig,
            GenerationConfig,
            OptimizationConfig,
            PipelineConfig,
            RewriterConfig,
            SchedulerConfig,
        )

        toml_sections = parse_toml_keys(CONFIG_EXAMPLE)

        checks = [
            ("encoder", EncoderConfig),
            ("pipeline", PipelineConfig),
            ("generation", GenerationConfig),
            ("scheduler", SchedulerConfig),
            ("optimization", OptimizationConfig),
            ("rewriter", RewriterConfig),
        ]

        for section, cls in checks:
            toml_keys = toml_sections.get(section, set())
            dataclass_fields = get_dataclass_fields(cls)
            missing = toml_keys - dataclass_fields
            if missing:
                errors.append(f"TOML [{section}] has params not in {cls.__name__}: {missing}")
                print(f"  FAIL: {section} -> {cls.__name__} (missing: {missing})")
            else:
                print(f"  OK: {section} -> {cls.__name__}")
    except Exception as e:
        errors.append(f"TOML check failed: {e}")
        print(f"  ERROR: {e}")

    # Check CLI -> RuntimeConfig
    print("\n[2/4] Checking CLI -> RuntimeConfig...")
    try:
        from llm_dit.config import RuntimeConfig

        cli_dests = extract_argparse_dests(CLI_PY)
        excluded = {
            "config", "config_name", "profile", "lora", "loras", "output",
            "prompts", "version", "embeddings_file", "ltx2_output",
            "flux2_output", "flux2_input_image", "flux2_offload",
            "flux2_no_offload", "wan_output", "torch_dtype",
        }

        rc = RuntimeConfig()
        cli_filtered = cli_dests - excluded
        missing = [d for d in cli_filtered if not hasattr(rc, d)]

        if missing:
            errors.append(f"CLI args not accessible on RuntimeConfig: {missing}")
            print(f"  FAIL: Not accessible on RuntimeConfig: {missing}")
        else:
            print(f"  OK: All CLI args accessible ({len(cli_filtered)} args)")
    except Exception as e:
        errors.append(f"CLI check failed: {e}")
        print(f"  ERROR: {e}")

    # Check key parameter wiring
    print("\n[3/4] Checking key parameter wiring...")
    try:
        if check_string_in_file(MODEL_MANAGER_PY, "hidden_layer=config.hidden_layer"):
            print("  OK: hidden_layer wired to APIBackendConfig")
        else:
            errors.append("hidden_layer not wired to APIBackendConfig in model_manager.py")
            print("  FAIL: hidden_layer not wired to APIBackendConfig")
    except Exception as e:
        errors.append(f"Wiring check failed: {e}")
        print(f"  ERROR: {e}")

    # Check documentation
    print("\n[4/4] Checking documentation...")
    try:
        claude_md = PROJECT_ROOT / "CLAUDE.md"
        if check_string_in_file(claude_md, "--hidden-layer"):
            print("  OK: --hidden-layer documented in CLAUDE.md")
        else:
            errors.append("--hidden-layer not documented in CLAUDE.md")
            print("  FAIL: --hidden-layer not in CLAUDE.md")

        if check_string_in_file(claude_md, "DRY Configuration Principles"):
            print("  OK: DRY principles documented")
        else:
            errors.append("DRY Configuration Principles not in CLAUDE.md")
            print("  FAIL: DRY principles not documented")
    except Exception as e:
        errors.append(f"Doc check failed: {e}")
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} issues found")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("PASSED: All consistency checks passed")
        sys.exit(0)


class TestResolveParamFieldConsistency:
    """Ensure all resolve_param() calls reference real fields on Pydantic schemas and config dataclasses.

    This test parses router source files to extract resolve_param() calls and verifies:
    1. The field name exists on the corresponding Pydantic request model
    2. The config dataclass has a corresponding field (with known name mappings)

    Prevents typos in resolve_param() calls from silently failing at runtime.
    """

    # Router -> Pydantic schema class mapping
    ROUTER_SCHEMAS = {
        "ltx2.py": [("LTX2GenerateRequest", "LTX2Config")],
        "flux2.py": [("Flux2GenerateRequest", "Flux2Config")],
        "qwen_image.py": [
            ("QwenImageEditLayerRequest", "QwenImageConfig"),
            ("QwenImageEditMultiRequest", "QwenImageConfig"),
            ("QwenImage2512GenerateRequest", "QwenImageConfig"),
        ],
    }

    # Known schema field -> config field name differences
    # Format: (schema_field, config_field)
    KNOWN_NAME_MAPPINGS = {
        # LTX-2 mappings
        ("stage1_steps", "LTX2Config"): "stage1_num_inference_steps",
        ("stage2_steps", "LTX2Config"): "stage2_num_inference_steps",
        ("enable_audio", "LTX2Config"): "audio_enabled",
        # FLUX.2 mappings
        ("num_steps", "Flux2Config"): "default_steps",
        ("guidance", "Flux2Config"): "default_guidance",
        # Qwen-Image mappings
        ("steps", "QwenImageConfig"): "num_inference_steps",
    }

    def _extract_resolve_param_calls(self, router_file: Path) -> list[tuple[str, str]]:
        """Extract all resolve_param() calls from a router file.

        Returns list of (field_name, line_number) tuples.
        """
        with open(router_file) as f:
            tree = ast.parse(f.read(), filename=str(router_file))

        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a resolve_param() call
                if isinstance(node.func, ast.Name) and node.func.id == "resolve_param":
                    # Second positional arg is the field name (as a string literal)
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        field_name = node.args[1].value
                        if isinstance(field_name, str):
                            calls.append((field_name, node.lineno))

        return calls

    def test_ltx2_resolve_param_fields(self):
        """All resolve_param() calls in ltx2.py reference real LTX2GenerateRequest and LTX2Config fields."""
        from llm_dit.config import LTX2Config
        from web.schemas import LTX2GenerateRequest

        router_file = PROJECT_ROOT / "web" / "routers" / "ltx2.py"
        calls = self._extract_resolve_param_calls(router_file)

        assert len(calls) > 0, "Expected to find resolve_param() calls in ltx2.py"

        # Get schema fields
        schema_fields = set(LTX2GenerateRequest.model_fields.keys())
        # Get config fields
        config_fields = get_dataclass_fields(LTX2Config)

        errors = []
        for field_name, line_no in calls:
            # Check schema has the field
            if field_name not in schema_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in LTX2GenerateRequest.model_fields. "
                    f"Available fields: {sorted(schema_fields)}"
                )

            # Check config has the field (exact match or known mapping)
            mapped_name = self.KNOWN_NAME_MAPPINGS.get((field_name, "LTX2Config"))
            if mapped_name:
                # Use the mapped name
                if mapped_name not in config_fields:
                    errors.append(
                        f"Line {line_no}: resolve_param uses '{field_name}' which maps to "
                        f"'{mapped_name}', but '{mapped_name}' not in LTX2Config. "
                        f"Update the mapping or add the field to LTX2Config."
                    )
            elif field_name not in config_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in LTX2Config and no known mapping exists. "
                    f"Add field to LTX2Config or update KNOWN_NAME_MAPPINGS."
                )

        assert not errors, "\n".join(errors)

    def test_flux2_resolve_param_fields(self):
        """All resolve_param() calls in flux2.py reference real Flux2GenerateRequest and Flux2Config fields."""
        from llm_dit.config import Flux2Config
        from web.schemas import Flux2GenerateRequest

        router_file = PROJECT_ROOT / "web" / "routers" / "flux2.py"
        calls = self._extract_resolve_param_calls(router_file)

        assert len(calls) > 0, "Expected to find resolve_param() calls in flux2.py"

        schema_fields = set(Flux2GenerateRequest.model_fields.keys())
        config_fields = get_dataclass_fields(Flux2Config)

        errors = []
        for field_name, line_no in calls:
            if field_name not in schema_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in Flux2GenerateRequest.model_fields. "
                    f"Available fields: {sorted(schema_fields)}"
                )

            mapped_name = self.KNOWN_NAME_MAPPINGS.get((field_name, "Flux2Config"))
            if mapped_name:
                if mapped_name not in config_fields:
                    errors.append(
                        f"Line {line_no}: resolve_param uses '{field_name}' which maps to "
                        f"'{mapped_name}', but '{mapped_name}' not in Flux2Config. "
                        f"Update the mapping or add the field to Flux2Config."
                    )
            elif field_name not in config_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in Flux2Config and no known mapping exists. "
                    f"Add field to Flux2Config or update KNOWN_NAME_MAPPINGS."
                )

        assert not errors, "\n".join(errors)

    def test_qwen_image_resolve_param_fields(self):
        """All resolve_param() calls in qwen_image.py reference real schema and QwenImageConfig fields."""
        from llm_dit.config import QwenImageConfig
        from web.schemas import (
            QwenImage2512GenerateRequest,
            QwenImageEditLayerRequest,
            QwenImageEditMultiRequest,
        )

        router_file = PROJECT_ROOT / "web" / "routers" / "qwen_image.py"
        calls = self._extract_resolve_param_calls(router_file)

        assert len(calls) > 0, "Expected to find resolve_param() calls in qwen_image.py"

        # Aggregate schema fields from all Qwen-Image request types
        all_schema_fields = set()
        all_schema_fields.update(QwenImageEditLayerRequest.model_fields.keys())
        all_schema_fields.update(QwenImageEditMultiRequest.model_fields.keys())
        all_schema_fields.update(QwenImage2512GenerateRequest.model_fields.keys())

        config_fields = get_dataclass_fields(QwenImageConfig)

        errors = []
        for field_name, line_no in calls:
            if field_name not in all_schema_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in any Qwen-Image request schema. "
                    f"Available fields: {sorted(all_schema_fields)}"
                )

            mapped_name = self.KNOWN_NAME_MAPPINGS.get((field_name, "QwenImageConfig"))
            if mapped_name:
                if mapped_name not in config_fields:
                    errors.append(
                        f"Line {line_no}: resolve_param uses '{field_name}' which maps to "
                        f"'{mapped_name}', but '{mapped_name}' not in QwenImageConfig. "
                        f"Update the mapping or add the field to QwenImageConfig."
                    )
            elif field_name not in config_fields:
                errors.append(
                    f"Line {line_no}: resolve_param(request, '{field_name}', ...) "
                    f"but '{field_name}' not in QwenImageConfig and no known mapping exists. "
                    f"Add field to QwenImageConfig or update KNOWN_NAME_MAPPINGS."
                )

        assert not errors, "\n".join(errors)


class TestDefaultsEndpointNameMappings:
    """Ensure _PARAM_NAME_MAPS and _PIPELINE_CONFIG_KEYS in config_mgmt.py
    reference real schema param IDs and config dataclass fields.

    Prevents name drift between the defaults endpoint mapping tables and
    the actual pipeline schemas / config dataclasses.
    """

    def test_pipeline_config_keys_point_to_real_sub_configs(self):
        """Every _PIPELINE_CONFIG_KEYS value must be a field on RuntimeConfig."""
        from llm_dit.config import RuntimeConfig
        from web.routers.config_mgmt import _PIPELINE_CONFIG_KEYS

        rc = RuntimeConfig()
        config_dict = rc.to_dict()

        errors = []
        for schema_id, config_key in _PIPELINE_CONFIG_KEYS.items():
            if config_key not in config_dict:
                errors.append(
                    f"_PIPELINE_CONFIG_KEYS['{schema_id}'] = '{config_key}', "
                    f"but '{config_key}' not in RuntimeConfig.to_dict(). "
                    f"Available keys: {sorted(config_dict.keys())}"
                )

        assert not errors, "\n".join(errors)

    def test_param_name_maps_reference_real_schema_params(self):
        """Every key in _PARAM_NAME_MAPS[pipeline] must be a real schema param ID."""
        from llm_dit.pipelines.schemas import get_pipeline
        from web.routers.config_mgmt import _PARAM_NAME_MAPS

        errors = []
        for pipeline_id, param_map in _PARAM_NAME_MAPS.items():
            schema = get_pipeline(pipeline_id)
            assert schema is not None, f"Pipeline '{pipeline_id}' not found in schema registry"

            schema_param_ids = {p.id for p in schema.params}
            for schema_name in param_map:
                if schema_name not in schema_param_ids:
                    errors.append(
                        f"_PARAM_NAME_MAPS['{pipeline_id}']['{schema_name}'] "
                        f"but '{schema_name}' not in schema params. "
                        f"Available: {sorted(schema_param_ids)}"
                    )

        assert not errors, "\n".join(errors)

    def test_param_name_maps_reference_real_config_fields(self):
        """Every value in _PARAM_NAME_MAPS[pipeline] must be a real config dataclass field."""
        from dataclasses import fields as dc_fields

        from llm_dit.config import Flux2Config, LTX2Config, QwenImageConfig, ZImageConfig
        from web.routers.config_mgmt import _PARAM_NAME_MAPS, _PIPELINE_CONFIG_KEYS

        # Map pipeline schema IDs to their config dataclass
        config_classes = {
            "ltx2": LTX2Config,
            "flux2": Flux2Config,
            "zimage": ZImageConfig,
            "qwenimage-t2i": QwenImageConfig,
            "qwenimage-edit": QwenImageConfig,
        }

        errors = []
        for pipeline_id, param_map in _PARAM_NAME_MAPS.items():
            config_cls = config_classes.get(pipeline_id)
            if config_cls is None:
                errors.append(f"No config class registered for pipeline '{pipeline_id}'")
                continue

            config_field_names = {f.name for f in dc_fields(config_cls)}
            for schema_name, config_name in param_map.items():
                if config_name not in config_field_names:
                    errors.append(
                        f"_PARAM_NAME_MAPS['{pipeline_id}']['{schema_name}'] = '{config_name}', "
                        f"but '{config_name}' not in {config_cls.__name__}. "
                        f"Available: {sorted(config_field_names)}"
                    )

        assert not errors, "\n".join(errors)

    def test_mismatched_params_are_all_mapped(self):
        """Schema params that don't match config field names must be in _PARAM_NAME_MAPS.

        This catches cases where a new schema param is added with a name that
        differs from the config field, but no mapping entry is created.
        """
        from dataclasses import fields as dc_fields

        from llm_dit.config import Flux2Config, LTX2Config, QwenImageConfig, ZImageConfig
        from llm_dit.pipelines.schemas import get_pipeline
        from web.routers.config_mgmt import _PARAM_NAME_MAPS, _PIPELINE_CONFIG_KEYS

        config_classes = {
            "ltx2": LTX2Config,
            "flux2": Flux2Config,
            "zimage": ZImageConfig,
            "qwenimage-t2i": QwenImageConfig,
            "qwenimage-edit": QwenImageConfig,
        }

        warnings = []
        for pipeline_id, config_cls in config_classes.items():
            schema = get_pipeline(pipeline_id)
            if schema is None:
                continue

            config_field_names = {f.name for f in dc_fields(config_cls)}
            param_map = _PARAM_NAME_MAPS.get(pipeline_id, {})

            for param in schema.params:
                if not param.config_mapped:
                    continue
                if param.id in param_map:
                    continue  # Already mapped
                if param.id in config_field_names:
                    continue  # Direct match, no mapping needed

                warnings.append(
                    f"Pipeline '{pipeline_id}': schema param '{param.id}' has no direct "
                    f"match in {config_cls.__name__} and no entry in _PARAM_NAME_MAPS. "
                    f"Either add a mapping or set config_mapped=False on the ParamSchema."
                )

        assert not warnings, "\n".join(warnings)


class TestLTX2ConfigFields:
    """LTX-2.3 specific config field validation."""

    def test_connectors_file_in_config(self):
        """connectors_file must exist on LTX2Config."""
        from llm_dit.config import LTX2Config

        fields = get_dataclass_fields(LTX2Config)
        assert "connectors_file" in fields

    def test_model_version_removed_from_config(self):
        """model_version must NOT be in LTX2Config (V2.3-only, no version dispatch)."""
        from llm_dit.config import LTX2Config

        fields = get_dataclass_fields(LTX2Config)
        assert "model_version" not in fields

    def test_stale_model_version_toml_stripped(self):
        """TOML with model_version in [ltx2] should be silently stripped without crash."""
        from llm_dit.config import Config

        data = {
            "ltx2": {"model_path": "/tmp/ltx2", "model_version": "2.3"},
        }
        config = Config.from_dict(data)
        assert config.ltx2.model_path == "/tmp/ltx2"
        assert not hasattr(config.ltx2, "model_version")


if __name__ == "__main__":
    run_consistency_check()
