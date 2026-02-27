"""FastAPI routers for the Z-Image generation server.

Each module handles a domain of endpoints:
- system: health, history, server info, rewriter
- vram: model load/unload/status
- config_mgmt: config CRUD, presets, generation-config
- core: Z-Image default generation pipeline
- flux2: FLUX.2 generation + config
- ltx2: LTX-2 video generation
- qwen_image: Qwen-Image all variants
"""
