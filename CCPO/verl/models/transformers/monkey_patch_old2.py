# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Apply monkey-patch function to models
"""

import importlib.metadata
import sys
from functools import lru_cache
from typing import Optional

import torch
from packaging import version
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.import_utils import is_trl_available
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = inputs_embeds is not None and current_ulysses_sp_size > 1 and getattr(self, "_needs_initial_slice", True)
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """
    Choose the forward function based on the model and backend.
    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.
    """
    if not use_fused_kernels or fused_kernels_backend not in ["triton", "torch"]:
        print(f"Skipping monkey patch for {model.__class__.__name__} as use_fused_kernels is {use_fused_kernels} or fused_kernels_backend is {fused_kernels_backend}")
        return

    forward_with_torch_backend_function = model.__class__.forward
    forward_with_triton_backend_function = model.__class__.forward
    if model.config.model_type == "qwen2_5_vl":
        from verl.models.transformers.qwen2_5_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    elif model.config.model_type == "qwen2_vl":
        from verl.models.transformers.qwen2_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    else:
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend

    if fused_kernels_backend == "triton":
        model.__class__.forward = forward_with_triton_backend_function
        print(f"Using Triton backend for fused kernels in {model.__class__.__name__}")
    elif fused_kernels_backend == "torch":
        model.__class__.forward = forward_with_torch_backend_function
        print(f"Using Torch backend for fused kernels in {model.__class__.__name__}")
    else:
        raise ValueError(f"Unsupported fused_kernels_backend: {fused_kernels_backend}. Choose 'triton' or 'torch'.")

# monkey_patch.py (or a small util imported from there)
# from transformers import AttentionInterface
# from transformers.integrations.flash_attention_2 import flash_attention_2_forward as fa2_forward

# def ulysses_fa2_wrapper(module, *args, **kwargs):
#     # Expected order: (query, key, value, [attention_mask])
#     # Some models also send attention_mask as a kwarg; prefer kwarg and avoid duplication.
#     q, k, v = args[:3]
#     attn_mask_pos = args[3] if len(args) >= 4 else None
#     attn_mask_kw = kwargs.pop("attention_mask", None)
#     attention_mask = attn_mask_kw if attn_mask_kw is not None else attn_mask_pos

#     # (Optional) read extra flags safely
#     # position_ids = kwargs.get("position_ids", None)
#     # use_cache = kwargs.get("use_cache", False)
#     # output_attentions = kwargs.get("output_attentions", False)
#     # top-left mask flag moved in recent TF; guard with getattr
#     kwargs["use_top_left_mask"] = getattr(module, "_flash_attn_uses_top_left_mask", False)

#     # TODO: if you have a custom varlen/remove-padding kernel, call it here.
#     # For a drop-in equivalent to your old FA2 path, delegate to HF's FA2:
#     return fa2_forward(module, q, k, v, attention_mask, **kwargs)

# # Make it available as an attention implementation:
# # AttentionInterface.register("ulysses_fa2", ulysses_fa2_wrapper)

# --- begin robust global FA2 hook for HF >= 4.53 ----------------------------
import transformers
from packaging import version

def _install_global_fa2_hook_for_hf(model):
    """
    Register a global flash_attention_2 hook that calls our Ulysses kernel
    using HF's AttentionInterface signature, avoiding arg collisions.
    """
    # import here to avoid import-time errors on older HF
    from transformers import AttentionInterface, AttentionMaskInterface
    # Mask builder name is stable across 4.52+ but guard anyway.
    try:
        from transformers.masking_utils import flash_attention_mask as _fa2_mask
    except Exception:
        _fa2_mask = None

    # 1) Get our Ulysses kernel (yours from verl)
    #    If you already have _ulysses_flash_attention_forward in this file,
    #    you can use that directly; otherwise import from verl.
    try:
        from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward as _ulysses_fa2_core
    except Exception:
        # Fallback to a local implementation if you have it defined
        # with a permissive signature (q,k,v,*args, position_ids=None, **kwargs)
        _ulysses_fa2_core = _ulysses_flash_attention_forward  # defined above in your file

    # 2) Build a shim that EXACTLY matches HF's attention signature:
    #    (module, query, key, value, attention_mask, **kwargs) -> (attn, weights|None)

        # ---- HF>=4.53 global FA2 adapter (do not edit your existing ulysses_flash_attn_forward) ----
    # def _ulysses_fa2_adapter(module, query, key, value, attention_mask, **kwargs):
    #     """
    #     Adapter with HF AttentionInterface signature:
    #       (module, query, key, value, attention_mask, **kwargs) -> (attn_out, attn_weights_or_None)
    #     Calls our local flash_attention_forward with the right keyword args,
    #     avoiding positional/duplication errors.
    #     """
    #     # 1) avoid "multiple values for argument 'attention_mask'"
    #     kw_mask = kwargs.pop("attention_mask", None)
    #     if kw_mask is not None:
    #         attention_mask = kw_mask

    #     # 2) normalize options
    #     q_len = query.size(-2)
    #     dropout = kwargs.pop("dropout", 0.0)
    #     sliding_window = kwargs.pop("sliding_window", None)
    #     is_causal = kwargs.pop("is_causal", True)
    #     deterministic = kwargs.pop("deterministic", None)

    #     # some models pass different names; normalize to 'position_ids'
    #     position_ids = kwargs.pop("position_ids", None) or kwargs.pop("norm_position_ids", None) or kwargs.pop("mm_positions", None)

    #     # Qwen variants flag this on the module; default False if missing
    #     use_top_left_mask = getattr(module, "_flash_attn_uses_top_left_mask", False)

    #     # 3) call the local FA2 entry (keywords only — no positional arity issues)
    #     attn_out = flash_attention_forward(
    #         query_states=query,
    #         key_states=key,
    #         value_states=value,
    #         attention_mask=attention_mask,
    #         query_length=q_len,
    #         is_causal=is_causal,
    #         position_ids=position_ids,
    #         sliding_window=sliding_window,
    #         use_top_left_mask=use_top_left_mask,
    #         deterministic=deterministic,
    #         dropout=dropout,
    #         **kwargs,
    #     )
    #     # HF AttentionInterface requires a 2-tuple
    #     return attn_out, None


    def _ulysses_fa2_adapter(module, query, key, value, attention_mask, **kwargs):
        # Avoid "multiple values for argument 'attention_mask'":
        # prefer kwarg if provided, but DO NOT pass both.
        kw_mask = kwargs.pop("attention_mask", None)
        if kw_mask is not None:
            attention_mask = kw_mask

        # Common extras HF may pass; provide sane defaults
        # (We do not forward them positionally; only as kwargs)
        kwargs.setdefault("dropout", 0.0)
        kwargs.setdefault("sliding_window", None)
        kwargs.setdefault("is_causal", True)

        # Position IDs show up under various names across Qwen variants
        # (e.g., position_ids, norm_position_ids, mm_positions). Forward any we see.
        posids = (
            kwargs.get("position_ids", None)
            or kwargs.get("norm_position_ids", None)
            or kwargs.get("mm_positions", None)
        )
        if posids is not None:
            kwargs["position_ids"] = posids  # normalize key

        # Call your kernel STRICTLY with keywords to avoid positional-arity errors.
        # Your core must accept (query_states, key_states, value_states, attention_mask=..., **kwargs)
        out = _ulysses_fa2_core(
            query, key, value,
            attention_mask=attention_mask,
            **kwargs,
        )
        # HF expects a 2-tuple (attn_output, attn_weights_or_None)
        return out, None

    # 3) Register the mask (optional but recommended) and the attention function
    try:
        if _fa2_mask is not None:
            AttentionMaskInterface.register("flash_attention_2", _fa2_mask)
    except Exception:
        # already registered in this process
        pass

    # Register our adapter under the SAME key so it's the global FA2 entrypoint
    AttentionInterface.register("flash_attention_2", _ulysses_fa2_adapter)

    # If a model instance is already loaded, make sure it points to FA2 now.
    try:
        # multimodal models can accept dict; "" key applies globally per HF docs
        model.set_attn_implementation({"": "flash_attention_2"})
    except Exception:
        # some older builds only accept a string
        try:
            model.set_attn_implementation("flash_attention_2")
        except Exception:
            pass

# --- in your apply_monkey_patch(...) body ---





def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """
    Apply monkey patch to the models for ulysses sequence parallel and fused kernel.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.
    """

    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = model.config.text_config.num_attention_heads, model.config.text_config.num_key_value_heads

    assert num_attention_heads % ulysses_sp_size == 0, f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,kv heads are repeated to ensure correctness."
    )

    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    import transformers
    from packaging import version


    if model.config.model_type == "qwen2_5_vl":
        if use_remove_padding or ulysses_sp_size > 1:
            if version.parse(transformers.__version__) >= version.parse("4.53.0"):
                _install_global_fa2_hook_for_hf(model)
                print("Monkey patched global FA2 via AttentionInterface (HF>=4.53)")
            else:
                # HF <= 4.52 path: keep the original class forward override
                try:
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
                    # Import your core
                    from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward as _ulysses_fa2_core
                    # Override with a minimal wrapper that matches the class forward
                    def _class_forward_wrapper(self, query, key, value, attention_mask=None, **kwargs):
                        return _ulysses_fa2_core(query, key, value, attention_mask=attention_mask, **kwargs)
                    Qwen2_5_VLFlashAttention2.forward = _class_forward_wrapper
                    print("Monkey patched Qwen2_5_VLFlashAttention2.forward (HF<=4.52)")
                except Exception as e:
                    # Some 4.52 builds backport the interface; fall back to the new path.
                    _install_global_fa2_hook_for_hf(model)
                    print(f"Class-level FA2 not found; fell back to AttentionInterface: {e}")
    
            # Keep your Ulysses input-slicing patch as before
            if ulysses_sp_size > 1:
                if version.parse(transformers.__version__) >= version.parse("4.52.0"):
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
                    patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
                else:
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
                    patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)


    
    # if model.config.model_type == "qwen2_5_vl":
        # if use_remove_padding or ulysses_sp_size > 1:
        #     # your existing kernel
        #     from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward
    
        #     if version.parse(transformers.__version__) >= version.parse("4.53.0"):
        #         # HF ≥ 4.53: use the new AttentionInterface
        #         from transformers import AttentionInterface, AttentionMaskInterface
        #         # use the built-in FA2 mask factory to preserve correct masking behavior
        #         from transformers.masking_utils import flash_attention_mask
    
        #         # Adapter so our old function matches the new interface signature
        #         def _ulysses_fa2_adapter(module, query, key, value, attention_mask, **kwargs):
        #             # kwargs we may care about (provide sane defaults)
        #             dropout = kwargs.get("dropout", 0.0)
        #             sliding_window = kwargs.get("sliding_window", None)
        #             is_causal = kwargs.get("is_causal", True)
        #             q_len = query.size(-2)
        #             # call your original FA2 path; return (out, None) per interface contract
        #             out = ulysses_flash_attn_forward(
        #                 query, key, value, attention_mask, q_len, dropout, sliding_window, is_causal
        #             )
        #             return out, None
    
        #         # Register under the existing key so callers can keep attn_implementation="flash_attention_2"
        #         AttentionMaskInterface.register("flash_attention_2", flash_attention_mask)
        #         AttentionInterface.register("flash_attention_2", _ulysses_fa2_adapter)
        #         print("Monkey patched FA2 via AttentionInterface for Qwen2.5-VL")
    
        #     else:
        #         # HF ≤ 4.52: original class exists, keep old monkey-patch
        #         from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
        #         Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
        #         print("Monkey patched Qwen2_5_VLFlashAttention2.forward")
    
        #     # your Ulysses input slicing patch stays as before
        #     if ulysses_sp_size > 1:
        #         if is_transformers_version_in_range(min_version="4.52.0"):
        #             from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
        #             patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
        #         else:
        #             from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
        #             patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)
                
    # # TODO: VLM models only, unify monkey patch to LLM models.
    # if model.config.model_type == "qwen2_5_vl":
    #     import transformers
    #     from packaging import version

    #     # before: from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
    #     # after: patch the attention function used when attn_implementation="flash_attention_2"
    
    #     if use_remove_padding or ulysses_sp_size > 1:
    #         # your existing FA2 override
    #         from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward
    
    #         # Transformers 4.53+ (incl. 4.55.x): no Qwen2_5_VLFlashAttention2 class; use the new attention interface.
    #         if version.parse(transformers.__version__) >= version.parse("4.53.0"):
    #             try:
    #                 # Prefer model-local map if available
    #                 from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention as _QwenAttn
    #                 if hasattr(_QwenAttn, "SUPPORTED_ATTENTION_IMPLEMENTATIONS"):
    #                     _QwenAttn.SUPPORTED_ATTENTION_IMPLEMENTATIONS["flash_attention_2"] = ulysses_flash_attn_forward
    #                 else:
    #                     # Fallback: global registration (affects all models using this key)
    #                     from transformers import AttentionInterface
    #                     # AttentionInterface.register("flash_attention_2", ulysses_flash_attn_forward)
    #                     AttentionInterface.register("flash_attention_2", ulysses_fa2_wrapper)
    #             except Exception:
    #                 # Final fallback: global registration
    #                 from transformers import AttentionInterface
    #                 AttentionInterface.register("flash_attention_2", ulysses_fa2_wrapper)
    #             print("Monkey patched FA2 for Qwen2.5-VL via AttentionInterface/mapping")
    
    #         else:
    #             # <= 4.52.x: original class exists, keep old monkey-patch
    #             from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
    #             Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
    #             print("Monkey patched Qwen2_5_VLFlashAttention2.forward")

    #     # your existing input slicing patch still applies
    #     if ulysses_sp_size > 1:
    #         if is_transformers_version_in_range(min_version="4.52.0"):
    #             from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    #             patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
    #         else:
    #             from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    #             patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)


    
        
        # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        #     Qwen2_5_VLFlashAttention2,
        # )

        # if use_remove_padding or ulysses_sp_size > 1:
        #     from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

        #     Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
        #     print("Monkey patch FlashAttention2.forward in Qwen2.5VL")

        # if ulysses_sp_size > 1:
        #     if is_transformers_version_in_range(min_version="4.52.0"):
        #         from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

        #         patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
        #     else:
        #         from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        #         patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)

    elif model.config.model_type == "qwen2_vl":
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLFlashAttention2,
        )

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

            Qwen2VLFlashAttention2.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2VL")

        if ulysses_sp_size > 1:
            if is_transformers_version_in_range(min_version="4.52.0"):
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel

                patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)
            else:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

                patch_vlm_for_ulysses_input_slicing(Qwen2VLModel)

    elif model.config.model_type == "kimi_vl":
        if use_remove_padding or ulysses_sp_size > 1:
            # TODO: Changes need to be made when transformers are adapted.
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward

            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in KimiVL")

        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)

        if use_fused_kernels:
            print("Not support fused kernels for KimiVL")

        return

    # transformers<=4.47.1
    if use_remove_padding or ulysses_sp_size > 1:
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            # transformers>=4.48.0
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

    patch_forward_with_backends(model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend)


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check
