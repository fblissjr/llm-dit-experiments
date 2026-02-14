# documentation checklist

*last updated: 2026-02-14*

## documentation structure

Documentation is organized in tiers for efficient Claude agent consumption:

### tier 1: CLAUDE.md (~300 lines)
Read at every cold start. Contains:
- Critical rules (no emojis, use uv, etc.)
- Architecture overview
- Key parameters (tables only)
- Directory structure
- Documentation map (pointers to detailed docs)

**Keep CLAUDE.md lean.** If adding content, consider if it belongs in a reference doc or guide instead.

### tier 2: docs/reference/
Specs and facts - load when working on specific areas:

| File | Content |
|------|---------|
| `cli_flags.md` | All CLI flags by category |
| `api_endpoints.md` | REST API reference |
| `configuration.md` | DRY config principles, TOML wiring |
| `resolution.md` | VAE constraints, presets |
| `dype.md` | High-resolution generation |
| `long_prompts.md` | 1504 token limit, compression |
| `chat_templates.md` | Qwen3/VL template formats |
| `documentation_checklist.md` | This file |

### tier 3: docs/guides/
How-to docs - load when working on features:

| File | Content |
|------|---------|
| `config_management.md` | Edit config from UI, profiles, server control |
| `flux2_klein.md` | FLUX.2 Klein generation guide |
| `prompt_rewriting.md` | Qwen3 prompt expansion |
| `lora.md` | LoRA loading and fusion |
| `distributed.md` | Mac encode, CUDA generate |
| `profiler.md` | Performance testing |

### tier 4: internal/
Research and development - load for context on past decisions:

| Path | Content |
|------|---------|
| `index.md` | Map of all internal docs |
| `state/current.md` | Current state, blockers, next steps |
| `principles/architectural_decisions.md` | Architectural north star |
| `research/` | Hypotheses, experiments, findings |
| `log/` | Session logs (log_YYYY-MM-DD.md) |

### where to put new content

| Content Type | Location |
|--------------|----------|
| New CLI flag | `docs/reference/cli_flags.md` |
| New API endpoint | `docs/reference/api_endpoints.md` |
| New feature guide | `docs/guides/<feature>.md` |
| Research notes | `internal/research/<topic>.md` |
| Session work | `internal/log/log_YYYY-MM-DD.md` |
| Architecture change | CLAUDE.md (brief) + relevant docs |

### avoiding duplication

- Each topic should have ONE source of truth
- Use pointers (links) instead of copying content
- When consolidating, update all references to point to the canonical location
- Each topic should have a single canonical location, with links from other docs

---

## after implementing any feature or significant change

| File | When to Update |
|------|----------------|
| `CHANGELOG.md` | Every feature, fix, or breaking change (semantic versioning) |
| `README.md` | New user-facing features, quick start examples |
| `CLAUDE.md` | New directories, architecture changes |
| `pyproject.toml` | New dependencies, version bumps |
| `config.toml.example` | New configurable parameters (with comments) |
| `experiments/README.md` | Experiment-related features or tools |
| `internal/log/log_YYYY-MM-DD.md` | Every session (create dated file) |
| `docs/*.md` | Feature-specific documentation |

## checklist for new features

1. Add entry to `CHANGELOG.md` under appropriate version
2. Update `README.md` if user-facing
3. Update `CLAUDE.md` Directory Structure if new files/directories
4. Update relevant `docs/*.md` or `experiments/README.md`
5. Create/update `internal/log/log_YYYY-MM-DD.md` with session details
6. Update `internal/state/current.md` with current state

## checklist for new experiments

1. Output directory: `experiments/results/<experiment_name>/`
2. Use shared utilities from `experiments/utils.py`:
   - `save_image_grid()` for comparison grids
   - `save_metadata()` for JSON metadata with timestamps
   - `create_comparison_grid()` for grids without saving
3. Do NOT create custom grid or metadata functions
4. Add argparse `--output-dir` with default to `experiments/results/`

## for new configurable parameters

Follow DRY Configuration Principles (see `docs/reference/configuration.md`). Only **2 touchpoints** required:

1. Add field to the Config dataclass in `src/llm_dit/config.py` (e.g., `Flux2Config`, `LTX2Config`)
2. Add TOML section in `config.toml`

`RuntimeConfig.from_toml_config()` picks it up automatically.

Optional additional steps:
3. Add CLI flag in `src/llm_dit/cli.py` if command-line override is needed
4. Run DRY config test: `uv run pytest tests/unit/test_dry_config.py -v`
5. Update `docs/reference/cli_flags.md` with new flag if added

## end of session

1. Create/update `internal/log/log_YYYY-MM-DD.md` with session summary
2. Update `internal/state/current.md` with:
   - Current focus
   - Recent decisions
   - Known blockers
   - Next steps
3. If significant changes: Update CHANGELOG.md
