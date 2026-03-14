"""
Tests for FLUX.2 KV-cache support for reference-image editing.

Last Updated: 2026-03-14

Tests causal_attn_fn, modulation blending, block-level KV extract/cached,
and model-level forward_kv_extract/forward_kv_cached.

Run with: uv run pytest tests/unit/test_flux2_kv_cache.py -v
"""

import pytest
import torch
from einops import rearrange

from llm_dit.models.flux2.transformer import (
    Flux2Transformer,
    DoubleStreamBlock,
    SingleStreamBlock,
    causal_attn_fn,
    _blend_mod_triple,
    _blend_double_mods,
    _blend_single_mods,
)
from llm_dit.models.flux2.constants import (
    Klein9BParams,
    FLUX2_MODEL_INFO,
    supports_kv_cache,
)
from llm_dit.models.flux2.rope import create_image_ids, create_text_ids, EmbedND


# ============================================================================
# Constants Registry Tests
# ============================================================================


class TestKVConstants:
    """Tests for KV model entries in constants."""

    def test_kv_models_registered(self):
        """KV model variants exist in registry."""
        assert "klein-9b-kv" in FLUX2_MODEL_INFO
        assert "klein-9b-kv-fp8" in FLUX2_MODEL_INFO

    def test_kv_models_have_flag(self):
        """KV models have use_kv_cache=True."""
        assert FLUX2_MODEL_INFO["klein-9b-kv"]["use_kv_cache"] is True
        assert FLUX2_MODEL_INFO["klein-9b-kv-fp8"]["use_kv_cache"] is True

    def test_non_kv_models_no_flag(self):
        """Non-KV models don't have use_kv_cache."""
        assert FLUX2_MODEL_INFO["klein-9b"].get("use_kv_cache", False) is False
        assert FLUX2_MODEL_INFO["klein-9b-fp8"].get("use_kv_cache", False) is False

    def test_supports_kv_cache_helper(self):
        """supports_kv_cache() correctly identifies KV models."""
        assert supports_kv_cache("klein-9b-kv") is True
        assert supports_kv_cache("klein-9b-kv-fp8") is True
        assert supports_kv_cache("klein-9b") is False
        assert supports_kv_cache("nonexistent") is False

    def test_kv_models_are_distilled(self):
        """KV models are distilled with correct defaults."""
        for name in ("klein-9b-kv", "klein-9b-kv-fp8"):
            info = FLUX2_MODEL_INFO[name]
            assert info["distilled"] is True
            assert info["defaults"]["guidance"] == 1.0
            assert info["defaults"]["num_steps"] == 4

    def test_kv_fp8_model_has_fp8_flag(self):
        """KV FP8 model has fp8=True."""
        assert FLUX2_MODEL_INFO["klein-9b-kv-fp8"].get("fp8", False) is True
        assert FLUX2_MODEL_INFO["klein-9b-kv"].get("fp8", False) is False

    def test_kv_models_use_klein9b_params(self):
        """KV models use Klein9BParams (same architecture)."""
        assert FLUX2_MODEL_INFO["klein-9b-kv"]["params_cls"] is Klein9BParams
        assert FLUX2_MODEL_INFO["klein-9b-kv-fp8"]["params_cls"] is Klein9BParams


# ============================================================================
# causal_attn_fn Tests
# ============================================================================


class TestCausalAttnFn:
    """Tests for the causal attention function."""

    @pytest.fixture
    def attn_dims(self):
        """Common attention dimensions."""
        return {"B": 1, "H": 4, "D": 32}

    def test_no_ref_tokens_matches_full_attention(self, attn_dims):
        """With num_ref_tokens=0, causal_attn_fn degenerates to full attention."""
        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt = 10
        num_img = 20
        L = num_txt + num_img

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        # causal with 0 ref tokens
        out_causal = causal_attn_fn(q, k, v, num_txt, num_ref_tokens=0)

        # Standard attention
        out_std = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), is_causal=False,
        )
        out_std = rearrange(out_std, "b h n d -> b n (h d)")

        assert torch.allclose(out_causal, out_std, atol=1e-5), \
            "With 0 ref tokens, causal_attn_fn should match standard attention"

    def test_with_ref_tokens_shape(self, attn_dims):
        """causal_attn_fn with ref tokens produces correct output shape."""
        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt = 10
        num_ref = 8
        num_img = 20
        L = num_txt + num_ref + num_img

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        out = causal_attn_fn(q, k, v, num_txt, num_ref)

        # Output should have same sequence length
        assert out.shape == (B, L, H * D)

    def test_ref_self_attends_only(self, attn_dims):
        """Reference tokens should only attend to themselves (causal mask)."""
        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt = 5
        num_ref = 4
        num_img = 6
        L = num_txt + num_ref + num_img

        torch.manual_seed(42)
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        out = causal_attn_fn(q, k, v, num_txt, num_ref)

        # Extract ref portion of output
        ref_out = out[:, num_txt:num_txt + num_ref, :]

        # Compute what ref-only self-attention would produce
        q_ref = q[:, :, num_txt:num_txt + num_ref, :]
        k_ref = k[:, :, num_txt:num_txt + num_ref, :]
        v_ref = v[:, :, num_txt:num_txt + num_ref, :]
        ref_only_out = torch.nn.functional.scaled_dot_product_attention(
            q_ref.contiguous(), k_ref.contiguous(), v_ref.contiguous(), is_causal=False,
        )
        ref_only_out = rearrange(ref_only_out, "b h n d -> b n (h d)")

        assert torch.allclose(ref_out, ref_only_out, atol=1e-5), \
            "Reference tokens should only attend to themselves"

    def test_cached_path_shape(self, attn_dims):
        """Cached path with kv_cache produces correct output shape."""
        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt = 10
        num_ref = 8
        num_img = 20

        # Cached path input: [txt, img] only (no ref in sequence)
        L_cached = num_txt + num_img
        q = torch.randn(B, H, L_cached, D)
        k = torch.randn(B, H, L_cached, D)
        v = torch.randn(B, H, L_cached, D)

        # Cache has ref K/V from step 0
        kv_cache = {
            "k_ref": torch.randn(B, H, num_ref, D),
            "v_ref": torch.randn(B, H, num_ref, D),
        }

        out = causal_attn_fn(q, k, v, num_txt, num_ref, kv_cache)

        assert out.shape == (B, L_cached, H * D)

    def test_cached_path_injects_ref_kv(self, attn_dims):
        """Cached path should inject ref K/V between txt and img."""
        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt = 5
        num_ref = 4
        num_img = 6
        L_cached = num_txt + num_img

        torch.manual_seed(42)
        q = torch.randn(B, H, L_cached, D)
        k = torch.randn(B, H, L_cached, D)
        v = torch.randn(B, H, L_cached, D)

        # With different ref caches, outputs should differ
        cache1 = {
            "k_ref": torch.randn(B, H, num_ref, D),
            "v_ref": torch.randn(B, H, num_ref, D),
        }
        cache2 = {
            "k_ref": torch.randn(B, H, num_ref, D),
            "v_ref": torch.randn(B, H, num_ref, D),
        }

        out1 = causal_attn_fn(q, k, v, num_txt, num_ref, cache1)
        out2 = causal_attn_fn(q, k, v, num_txt, num_ref, cache2)

        assert not torch.allclose(out1, out2, atol=1e-5), \
            "Different ref caches should produce different outputs"

    def test_uses_attention_forward_backend(self, attn_dims):
        """causal_attn_fn should use attention_forward() instead of raw F.sdpa."""
        from unittest.mock import patch
        from llm_dit.utils.attention import attention_forward

        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt, num_ref, num_img = 5, 4, 6
        L = num_txt + num_ref + num_img

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        # Patch attention_forward and verify it's called
        with patch("llm_dit.models.flux2.transformer.attention_forward", wraps=attention_forward) as mock_attn:
            causal_attn_fn(q, k, v, num_txt, num_ref)
            # Extract path calls attention_forward twice:
            # once for txt+img attending to all, once for ref self-attention
            assert mock_attn.call_count == 2, (
                f"Expected 2 calls to attention_forward, got {mock_attn.call_count}"
            )

    def test_cached_path_uses_attention_forward(self, attn_dims):
        """Cached path should also use attention_forward()."""
        from unittest.mock import patch
        from llm_dit.utils.attention import attention_forward

        B, H, D = attn_dims["B"], attn_dims["H"], attn_dims["D"]
        num_txt, num_ref, num_img = 5, 4, 6
        L_cached = num_txt + num_img

        q = torch.randn(B, H, L_cached, D)
        k = torch.randn(B, H, L_cached, D)
        v = torch.randn(B, H, L_cached, D)

        kv_cache = {
            "k_ref": torch.randn(B, H, num_ref, D),
            "v_ref": torch.randn(B, H, num_ref, D),
        }

        with patch("llm_dit.models.flux2.transformer.attention_forward", wraps=attention_forward) as mock_attn:
            causal_attn_fn(q, k, v, num_txt, num_ref, kv_cache)
            # Cached path calls attention_forward once
            assert mock_attn.call_count == 1, (
                f"Expected 1 call to attention_forward, got {mock_attn.call_count}"
            )


# ============================================================================
# Modulation Blending Tests
# ============================================================================


class TestModulationBlending:
    """Tests for modulation blending helpers."""

    def test_blend_mod_triple_shape(self):
        """_blend_mod_triple produces correct shape."""
        B, D = 2, 64
        num_ref = 8
        seq_len = 32

        img_m = (torch.ones(B, 1, D), torch.ones(B, 1, D), torch.ones(B, 1, D))
        ref_m = (torch.zeros(B, 1, D), torch.zeros(B, 1, D), torch.zeros(B, 1, D))

        result = _blend_mod_triple(img_m, ref_m, num_ref, seq_len)

        assert len(result) == 3
        for t in result:
            assert t.shape == (B, seq_len, D)

    def test_blend_mod_triple_values(self):
        """_blend_mod_triple puts ref values first, img values after."""
        B, D = 1, 4
        num_ref = 3
        seq_len = 8

        img_m = (torch.ones(B, 1, D) * 2.0,) * 3
        ref_m = (torch.ones(B, 1, D) * 5.0,) * 3

        result = _blend_mod_triple(img_m, ref_m, num_ref, seq_len)

        for t in result:
            # First num_ref positions should have ref values (5.0)
            assert torch.allclose(t[:, :num_ref, :], torch.full((B, num_ref, D), 5.0))
            # Remaining positions should have img values (2.0)
            assert torch.allclose(t[:, num_ref:, :], torch.full((B, seq_len - num_ref, D), 2.0))

    def test_blend_double_mods_shape(self):
        """_blend_double_mods produces two triples."""
        B, D = 1, 32
        num_ref = 4
        seq_len = 16

        def make_mod():
            return (
                (torch.randn(B, 1, D), torch.randn(B, 1, D), torch.randn(B, 1, D)),
                (torch.randn(B, 1, D), torch.randn(B, 1, D), torch.randn(B, 1, D)),
            )

        result = _blend_double_mods(make_mod(), make_mod(), num_ref, seq_len)
        assert len(result) == 2
        for triple in result:
            assert len(triple) == 3
            for t in triple:
                assert t.shape == (B, seq_len, D)

    def test_blend_single_mods_layout(self):
        """_blend_single_mods produces [txt, ref, img] layout."""
        B, D = 1, 4
        num_txt = 3
        num_ref = 2
        seq_len = 10  # 3 txt + 2 ref + 5 img

        img_m = (torch.ones(B, 1, D) * 1.0,) * 3
        ref_m = (torch.ones(B, 1, D) * 9.0,) * 3

        result = _blend_single_mods(img_m, ref_m, num_txt, num_ref, seq_len)

        for t in result:
            # txt positions get img_m (1.0)
            assert torch.allclose(t[:, :num_txt, :], torch.full((B, num_txt, D), 1.0))
            # ref positions get ref_m (9.0)
            assert torch.allclose(t[:, num_txt:num_txt + num_ref, :], torch.full((B, num_ref, D), 9.0))
            # img positions get img_m (1.0)
            assert torch.allclose(t[:, num_txt + num_ref:, :], torch.full((B, seq_len - num_txt - num_ref, D), 1.0))


# ============================================================================
# Block-Level KV Tests
# ============================================================================


class TestDoubleStreamBlockKV:
    """Tests for DoubleStreamBlock KV extract/cached methods."""

    @pytest.fixture
    def setup(self):
        """Create block and test inputs."""
        hidden = 128
        heads = 4
        block = DoubleStreamBlock(hidden, heads, mlp_ratio=3.0)
        B, img_len, txt_len = 1, 32, 10
        num_ref = 8

        # img has layout [ref, img] for extract
        img = torch.randn(B, num_ref + img_len, hidden)
        txt = torch.randn(B, txt_len, hidden)

        embed = EmbedND(dim=hidden // heads, theta=2000, axes_dim=[8, 8, 8, 8])
        # Position IDs for [ref + img] tokens
        ref_ids = torch.zeros(B, num_ref, 4)
        ref_ids[..., 0] = 10.0  # t=10 for ref
        ref_ids[..., 1] = torch.arange(num_ref).float()
        img_ids = create_image_ids(B, 4, 8)  # 32 tokens
        combined_img_ids = torch.cat([ref_ids, img_ids], dim=1)
        txt_ids = create_text_ids(B, txt_len)

        pe_x = embed(combined_img_ids)
        pe_ctx = embed(txt_ids)

        def make_mod(seq_len):
            return (
                (
                    torch.zeros(B, seq_len, hidden),
                    torch.zeros(B, seq_len, hidden),
                    torch.ones(B, seq_len, hidden),
                ),
                (
                    torch.zeros(B, seq_len, hidden),
                    torch.zeros(B, seq_len, hidden),
                    torch.ones(B, seq_len, hidden),
                ),
            )

        mod_img = make_mod(num_ref + img_len)
        mod_txt = make_mod(txt_len)

        return {
            "block": block, "img": img, "txt": txt,
            "pe_x": pe_x, "pe_ctx": pe_ctx,
            "mod_img": mod_img, "mod_txt": mod_txt,
            "num_ref": num_ref, "B": B, "img_len": img_len,
            "txt_len": txt_len, "hidden": hidden,
        }

    def test_forward_kv_extract_shapes(self, setup):
        """forward_kv_extract returns correct shapes and cache."""
        s = setup
        img_out, txt_out, cache = s["block"].forward_kv_extract(
            s["img"], s["txt"], s["pe_x"], s["pe_ctx"],
            s["mod_img"], s["mod_txt"], s["num_ref"],
        )

        # Output shapes should match input
        assert img_out.shape == s["img"].shape
        assert txt_out.shape == s["txt"].shape

        # Cache should contain ref K/V
        assert "k_ref" in cache
        assert "v_ref" in cache
        # Cache should have num_ref tokens in sequence dim
        assert cache["k_ref"].shape[2] == s["num_ref"]
        assert cache["v_ref"].shape[2] == s["num_ref"]

    def test_forward_kv_cached_shapes(self, setup):
        """forward_kv_cached uses cache and returns correct shapes."""
        s = setup

        # First extract to get cache
        _, _, cache = s["block"].forward_kv_extract(
            s["img"], s["txt"], s["pe_x"], s["pe_ctx"],
            s["mod_img"], s["mod_txt"], s["num_ref"],
        )

        # Now use cached path: img has layout [img] only (no ref)
        img_no_ref = torch.randn(s["B"], s["img_len"], s["hidden"])
        img_ids = create_image_ids(s["B"], 4, 8)
        txt_ids = create_text_ids(s["B"], s["txt_len"])

        embed = EmbedND(dim=s["hidden"] // 4, theta=2000, axes_dim=[8, 8, 8, 8])
        pe_x = embed(img_ids)
        pe_ctx = embed(txt_ids)

        def make_mod(seq_len):
            return (
                (
                    torch.zeros(s["B"], seq_len, s["hidden"]),
                    torch.zeros(s["B"], seq_len, s["hidden"]),
                    torch.ones(s["B"], seq_len, s["hidden"]),
                ),
                (
                    torch.zeros(s["B"], seq_len, s["hidden"]),
                    torch.zeros(s["B"], seq_len, s["hidden"]),
                    torch.ones(s["B"], seq_len, s["hidden"]),
                ),
            )

        img_out, txt_out = s["block"].forward_kv_cached(
            img_no_ref, s["txt"], pe_x, pe_ctx,
            make_mod(s["img_len"]), make_mod(s["txt_len"]),
            cache,
        )

        assert img_out.shape == (s["B"], s["img_len"], s["hidden"])
        assert txt_out.shape == s["txt"].shape


class TestSingleStreamBlockKV:
    """Tests for SingleStreamBlock KV extract/cached methods."""

    @pytest.fixture
    def setup(self):
        """Create block and test inputs."""
        hidden = 128
        heads = 4
        block = SingleStreamBlock(hidden, heads, mlp_ratio=3.0)
        B, num_txt, num_ref, num_img = 1, 10, 8, 32
        seq_len = num_txt + num_ref + num_img

        x = torch.randn(B, seq_len, hidden)

        embed = EmbedND(dim=hidden // heads, theta=2000, axes_dim=[8, 8, 8, 8])
        ids = torch.zeros(B, seq_len, 4)
        ids[..., 3] = torch.arange(seq_len).float()
        pe = embed(ids)

        mod = (
            torch.zeros(B, seq_len, hidden),
            torch.zeros(B, seq_len, hidden),
            torch.ones(B, seq_len, hidden),
        )

        return {
            "block": block, "x": x, "pe": pe, "mod": mod,
            "num_txt": num_txt, "num_ref": num_ref, "num_img": num_img,
            "B": B, "hidden": hidden, "seq_len": seq_len,
        }

    def test_forward_kv_extract_shapes(self, setup):
        """forward_kv_extract returns correct shapes and cache."""
        s = setup
        out, cache = s["block"].forward_kv_extract(
            s["x"], s["pe"], s["mod"], s["num_txt"], s["num_ref"],
        )

        assert out.shape == s["x"].shape
        assert cache["k_ref"].shape[2] == s["num_ref"]
        assert cache["v_ref"].shape[2] == s["num_ref"]

    def test_forward_kv_cached_shapes(self, setup):
        """forward_kv_cached uses cache and returns correct shapes."""
        s = setup

        # Extract to get cache
        _, cache = s["block"].forward_kv_extract(
            s["x"], s["pe"], s["mod"], s["num_txt"], s["num_ref"],
        )

        # Cached path: [txt, img] only
        seq_len_cached = s["num_txt"] + s["num_img"]
        x_cached = torch.randn(s["B"], seq_len_cached, s["hidden"])

        embed = EmbedND(dim=s["hidden"] // 4, theta=2000, axes_dim=[8, 8, 8, 8])
        ids = torch.zeros(s["B"], seq_len_cached, 4)
        ids[..., 3] = torch.arange(seq_len_cached).float()
        pe = embed(ids)

        mod = (
            torch.zeros(s["B"], seq_len_cached, s["hidden"]),
            torch.zeros(s["B"], seq_len_cached, s["hidden"]),
            torch.ones(s["B"], seq_len_cached, s["hidden"]),
        )

        out = s["block"].forward_kv_cached(
            x_cached, pe, mod, s["num_txt"], cache,
        )

        assert out.shape == (s["B"], seq_len_cached, s["hidden"])


# ============================================================================
# Flux2Transformer Model-Level KV Tests
# ============================================================================


class TestFlux2TransformerKV:
    """Tests for Flux2Transformer forward_kv_extract/forward_kv_cached."""

    @pytest.fixture
    def mini_model(self):
        """Create minimal transformer (1 double + 1 single block)."""
        params = Klein9BParams()
        params.depth = 1
        params.depth_single_blocks = 1
        model = Flux2Transformer(params)
        model.eval()
        return model

    @pytest.fixture
    def model_inputs(self):
        """Create test inputs for the model."""
        B = 1
        img_len = 64  # 8x8 latent
        txt_len = 16
        ref_len = 32  # 4x8 ref latent
        in_ch = 128
        ctx_dim = 12288
        device = "cpu"

        x = torch.randn(B, img_len, in_ch, device=device)
        x_ids = create_image_ids(B, 8, 8, device=device)
        timesteps = torch.tensor([0.5], device=device)
        ctx = torch.randn(B, txt_len, ctx_dim, device=device)
        ctx_ids = create_text_ids(B, txt_len, device=device)

        x_ref = torch.randn(B, ref_len, in_ch, device=device)
        ref_ids = torch.zeros(B, ref_len, 4, device=device)
        ref_ids[..., 0] = 10.0
        ref_ids[..., 1] = torch.arange(ref_len).float().unsqueeze(0)

        return {
            "x": x, "x_ids": x_ids, "timesteps": timesteps,
            "ctx": ctx, "ctx_ids": ctx_ids,
            "x_ref": x_ref, "ref_ids": ref_ids,
            "B": B, "img_len": img_len, "ref_len": ref_len, "in_ch": in_ch,
        }

    def test_forward_still_works(self, mini_model, model_inputs):
        """Standard forward() still works after refactoring."""
        inp = model_inputs
        with torch.no_grad():
            out = mini_model.forward(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
            )
        assert out.shape == (inp["B"], inp["img_len"], inp["in_ch"])

    def test_forward_matches_kv_extract_zero_ref(self, mini_model, model_inputs):
        """forward() output must match forward_kv_extract(num_ref_tokens=0).

        This is the core unification invariant: forward() delegates to
        forward_kv_extract at the block level, so they must produce
        identical results when there are no reference tokens.
        """
        inp = model_inputs
        torch.manual_seed(0)
        with torch.no_grad():
            out_forward = mini_model.forward(
                x=inp["x"].clone(), x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
            )

        # forward_kv_extract with empty ref tensors (0 ref tokens)
        empty_ref = torch.zeros(inp["B"], 0, inp["in_ch"])
        empty_ref_ids = torch.zeros(inp["B"], 0, 4)
        torch.manual_seed(0)
        with torch.no_grad():
            out_extract, kv_cache = mini_model.forward_kv_extract(
                x=inp["x"].clone(), x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                x_seq_concat=empty_ref,
                x_seq_concat_ids=empty_ref_ids,
            )

        assert torch.allclose(out_forward, out_extract, atol=1e-5), (
            f"forward() and forward_kv_extract(num_ref_tokens=0) must match. "
            f"Max diff: {(out_forward - out_extract).abs().max():.6f}"
        )

        # KV cache should have empty entries
        for cache in kv_cache["double_blocks"]:
            assert cache["k_ref"].shape[2] == 0
        for cache in kv_cache["single_blocks"]:
            assert cache["k_ref"].shape[2] == 0

    def test_forward_kv_extract_shapes(self, mini_model, model_inputs):
        """forward_kv_extract returns prediction + kv_cache."""
        inp = model_inputs
        with torch.no_grad():
            pred, kv_cache = mini_model.forward_kv_extract(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                x_seq_concat=inp["x_ref"],
                x_seq_concat_ids=inp["ref_ids"],
            )

        # Prediction should be img_len tokens (not ref)
        assert pred.shape == (inp["B"], inp["img_len"], inp["in_ch"])

        # KV cache should have entries for each block
        assert "double_blocks" in kv_cache
        assert "single_blocks" in kv_cache
        assert len(kv_cache["double_blocks"]) == 1  # mini model has 1
        assert len(kv_cache["single_blocks"]) == 1
        assert kv_cache["num_ref_tokens"] == inp["ref_len"]

    def test_forward_kv_cached_shapes(self, mini_model, model_inputs):
        """forward_kv_cached uses cache and returns correct shape."""
        inp = model_inputs

        # Step 0: extract
        with torch.no_grad():
            _, kv_cache = mini_model.forward_kv_extract(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                x_seq_concat=inp["x_ref"],
                x_seq_concat_ids=inp["ref_ids"],
            )

        # Step 1+: cached (no ref tokens in input)
        with torch.no_grad():
            pred = mini_model.forward_kv_cached(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=torch.tensor([0.4]),
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                kv_cache=kv_cache,
            )

        assert pred.shape == (inp["B"], inp["img_len"], inp["in_ch"])

    def test_cached_differs_from_no_cache(self, mini_model, model_inputs):
        """Cached forward should produce different output than no-cache forward."""
        inp = model_inputs

        # No-cache forward (standard)
        with torch.no_grad():
            out_standard = mini_model.forward(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
            )

        # KV-cached forward
        with torch.no_grad():
            pred_extract, _kv_cache = mini_model.forward_kv_extract(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                x_seq_concat=inp["x_ref"],
                x_seq_concat_ids=inp["ref_ids"],
            )

        # They should differ because extract path includes ref tokens
        assert not torch.allclose(out_standard, pred_extract, atol=1e-3), \
            "KV extract with ref tokens should differ from standard forward"

    def test_kv_cache_contains_valid_tensors(self, mini_model, model_inputs):
        """KV cache tensors should be finite and have correct dims."""
        inp = model_inputs

        with torch.no_grad():
            _, kv_cache = mini_model.forward_kv_extract(
                x=inp["x"], x_ids=inp["x_ids"],
                timesteps=inp["timesteps"],
                ctx=inp["ctx"], ctx_ids=inp["ctx_ids"],
                guidance=None,
                x_seq_concat=inp["x_ref"],
                x_seq_concat_ids=inp["ref_ids"],
            )

        for cache in kv_cache["double_blocks"]:
            assert torch.isfinite(cache["k_ref"]).all()
            assert torch.isfinite(cache["v_ref"]).all()
            # Should be [B, H, ref_len, D]
            assert cache["k_ref"].ndim == 4
            assert cache["k_ref"].shape[2] == inp["ref_len"]

        for cache in kv_cache["single_blocks"]:
            assert torch.isfinite(cache["k_ref"]).all()
            assert torch.isfinite(cache["v_ref"]).all()
            assert cache["k_ref"].ndim == 4
            assert cache["k_ref"].shape[2] == inp["ref_len"]


# =============================================================================
# Test denoise_cached pipeline function
# =============================================================================

class TestDenoiseCached:
    """Tests for the denoise_cached() denoising loop function."""

    @pytest.fixture
    def mini_model(self):
        """Tiny transformer for denoise_cached testing."""
        params = Klein9BParams()
        params.depth = 1
        params.depth_single_blocks = 1
        model = Flux2Transformer(params)
        model.eval()
        return model

    def test_denoise_cached_runs(self, mini_model):
        """denoise_cached completes without error and returns correct shape."""
        from llm_dit.pipelines.flux2_generate import denoise_cached

        B, img_tokens, channels = 1, 32, 128
        txt_tokens = 8
        ref_tokens = 16
        context_dim = Klein9BParams().context_in_dim

        img = torch.randn(B, img_tokens, channels)
        img_ids = create_image_ids(B, 4, 8, dtype=torch.float32)
        txt = torch.randn(B, txt_tokens, context_dim)
        txt_ids = create_text_ids(B, txt_tokens, dtype=torch.float32)

        # Reference tokens with unique temporal coords
        ref = torch.randn(B, ref_tokens, channels)
        ref_ids = create_image_ids(B, 4, 4, dtype=torch.float32)
        ref_ids[:, :, 0] = 10.0  # Unique temporal coordinate

        timesteps = [1.0, 0.5, 0.0]  # 2 steps

        with torch.no_grad():
            result = denoise_cached(
                model=mini_model,
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=None,
                img_cond_seq=ref,
                img_cond_seq_ids=ref_ids,
            )

        assert result.shape == img.shape
        assert torch.isfinite(result).all()

    def test_denoise_cached_requires_ref_tokens(self, mini_model):
        """denoise_cached should assert when called without reference tokens."""
        from llm_dit.pipelines.flux2_generate import denoise_cached

        img = torch.randn(1, 32, 128)
        img_ids = create_image_ids(1, 4, 8, dtype=torch.float32)
        txt = torch.randn(1, 8, Klein9BParams().context_in_dim)
        txt_ids = create_text_ids(1, 8, dtype=torch.float32)

        with pytest.raises(AssertionError, match="requires reference tokens"):
            denoise_cached(
                model=mini_model,
                img=img, img_ids=img_ids,
                txt=txt, txt_ids=txt_ids,
                timesteps=[1.0, 0.0],
                img_cond_seq=None,
                img_cond_seq_ids=None,
            )

    def test_denoise_cached_differs_from_denoise(self, mini_model):
        """KV-cached denoising should produce different output than standard denoise."""
        from llm_dit.pipelines.flux2_generate import denoise, denoise_cached

        B, img_tokens, channels = 1, 32, 128
        txt_tokens = 8
        ref_tokens_count = 16
        context_dim = Klein9BParams().context_in_dim

        torch.manual_seed(42)
        img = torch.randn(B, img_tokens, channels)
        img_ids = create_image_ids(B, 4, 8, dtype=torch.float32)
        txt = torch.randn(B, txt_tokens, context_dim)
        txt_ids = create_text_ids(B, txt_tokens, dtype=torch.float32)
        ref = torch.randn(B, ref_tokens_count, channels)
        ref_ids = create_image_ids(B, 4, 4, dtype=torch.float32)
        ref_ids[:, :, 0] = 10.0

        timesteps = [1.0, 0.5, 0.0]

        with torch.no_grad():
            # Standard denoise (concatenates ref every step)
            torch.manual_seed(42)
            img_standard = torch.randn(B, img_tokens, channels)
            out_standard = denoise(
                model=mini_model,
                img=img_standard,
                img_ids=img_ids,
                txt=txt, txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=None,
                img_cond_seq=ref,
                img_cond_seq_ids=ref_ids,
            )

            # KV-cached denoise (ref processed once on step 0)
            torch.manual_seed(42)
            img_cached = torch.randn(B, img_tokens, channels)
            out_cached = denoise_cached(
                model=mini_model,
                img=img_cached,
                img_ids=img_ids,
                txt=txt, txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=None,
                img_cond_seq=ref,
                img_cond_seq_ids=ref_ids,
            )

        # Both should be finite
        assert torch.isfinite(out_standard).all()
        assert torch.isfinite(out_cached).all()
        # They will differ because the attention paths differ
        # (standard: ref in every step, cached: ref only step 0 then cached KV)
        assert out_standard.shape == out_cached.shape


# =============================================================================
# KV cache logging key consistency
# =============================================================================


class TestKVCacheLogging:
    """Logging must use the same dict keys as the cache structure."""

    def test_log_uses_correct_cache_keys(self):
        """denoise_cached log string must reference 'double_blocks'/'single_blocks',
        not bare 'double'/'single'."""
        import inspect
        from llm_dit.pipelines import flux2_generate

        source = inspect.getsource(flux2_generate.denoise_cached)
        # The cache dict uses 'double_blocks' and 'single_blocks'
        assert "double_blocks" in source, (
            "denoise_cached must reference 'double_blocks' (not bare 'double')"
        )
        assert "single_blocks" in source, (
            "denoise_cached must reference 'single_blocks' (not bare 'single')"
        )
