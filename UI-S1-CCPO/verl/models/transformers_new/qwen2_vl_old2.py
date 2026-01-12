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

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLAttention,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from verl.utils.device import is_npu_available
from verl.utils.transformers_compat import is_transformers_version_in_range
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

# NOTE: is_npu_available can be either a function or boolean in different VERL builds.
try:
    _is_npu = is_npu_available()  # type: ignore[operator]
except TypeError:
    _is_npu = bool(is_npu_available)

if _is_npu:
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = flash_attn_supports_top_left_mask()

_flash_deterministic_enabled = os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1405
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    # position_ids: (batch_size, seq_length)
    assert position_ids.ndim == 2
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.view(-1)
    cu_seqlens = torch.cat(
        (
            (position_ids == 0).nonzero().view(-1).to(torch.int32),
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))


def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if _flash_supports_deterministic:
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    if kwargs.get("softcap") is not None:
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    # --- UI-S1 compat: tolerate None / 1D / 2D here; normalize to (B, S) ---
    if position_ids is not None:
        if position_ids.ndim == 1:
            # treat as single example
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.ndim == 3:
            # already handled in the call-site, but keep an extra guard
            position_ids = position_ids[0]
        elif position_ids.ndim != 2:
            position_ids = None

    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size > 1 and position_ids is not None:
        # qkv: (batch_size, seq_length / sp_size, num_head, head_size)
        validate_ulysses_config(query_states.size(2), sp_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (batch_size, seq_length)

    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        batch_size = query_states.size(0)
        q, k, v, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )  # do not pass position_ids to old flash_attention_forward

    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


def qwen2_vl_attn_forward(
    self: "Qwen2VLAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> tuple[torch.Tensor, None, None]:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    # This is before the transpose
    q_len = query_states.shape[2]

    # FA2 uses non-transposed inputs
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # --- UI-S1 compat: normalize to (B, S) or None for the FA2 varlen path ---
    pos_2d = None
    if position_ids is not None:
        if position_ids.ndim == 3:
            # could be (3, B, S) or (4, B, S) — use channel 0 (text) for FA2 grouping
            pos_2d = position_ids[0]
        elif position_ids.ndim == 2:
            # already (B, S)
            pos_2d = position_ids
        elif position_ids.ndim == 1:
            # (S,) -> assume batch=1
            pos_2d = position_ids.unsqueeze(0)

    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        sliding_window=sliding_window,
        use_top_left_mask=_flash_use_top_left_mask,
        position_ids=pos_2d,  # important: pass normalized position ids
    )  # (batch_size, seq_length / sp_size, num_head, head_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    if is_transformers_version_in_range(min_version="4.54.0"):
        return attn_output, None
    else:
        return attn_output, None, None


def _get_input_embeds(
    model: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:  # handle mixed text-image data
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


# --- UI-S1 compat helpers -----------------------------------------------------

def _make_text_positions_from_mask(attn_mask: Optional[torch.Tensor], length: int, device) -> torch.Tensor:
    """
    Produce a (B, S) text position index from an attention mask when available,
    otherwise fall back to arange(0..S-1) per batch.
    """
    if attn_mask is not None and attn_mask.ndim == 2:
        pos = attn_mask.long().cumsum(-1) - 1
        pos = pos.clamp_min(0)
        return pos.to(device)
    else:
        return torch.arange(length, device=device).unsqueeze(0)


def _expand_batch_if_needed(t: torch.Tensor, batch: int) -> torch.Tensor:
    """
    If you pass a single-example position id (S,) or (C, S), expand it to (B, S) or (C, B, S).
    """
    if t.ndim == 1:
        return t.unsqueeze(0).expand(batch, -1)  # (B, S)
    if t.ndim == 2:
        # Could be (C, S) or (B, S). If the first dim is small (<=4), treat as channels.
        if t.size(0) <= 4 and t.size(1) > 4:  # (C, S)
            return t.unsqueeze(1).expand(-1, batch, -1)  # (C, B, S)
        # else assume (B, S)
    return t


def process_position_ids(
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Backward-compatible position_ids normalizer.

    Accepts legacy UI-S1 shapes:
      - (3, S)      : vision-only for a single example
      - (3, B, S)   : vision-only
      - (B, S)      : text-only
      - (S,)        : text-only single example
      - (4, B, S)   : text+vision (new HF mRoPE)
    and returns what the current Transformers expects:
      - >= 4.54.0 : (4, B, S)
      - <= 4.53.3 : (3, B, S)  (vision-only; HF builds text internally)

    If some channels are missing, we synthesize them conservatively:
      - missing text channel  -> built from attention_mask (or arange)
      - missing vision chans  -> copy text positions into 3 vision channels
    """
    if position_ids is None:
        return None

    # Infer batch/seq
    # Prefer attention_mask for batch & seq length if available.
    B = attention_mask.size(0) if (attention_mask is not None and attention_mask.ndim == 2) else None
    S = attention_mask.size(1) if (attention_mask is not None and attention_mask.ndim == 2) else None

    device = position_ids.device
    pid = position_ids

    # Add batch dimension to legacy shapes if needed
    if B is not None:
        pid = _expand_batch_if_needed(pid, B)

    # Now branch by ndim/shape
    if pid.ndim == 1:
        # (S,) -> (B=1, S)
        pid_text = pid.unsqueeze(0)  # (1, S)
        B_eff = 1
        S_eff = pid_text.size(1)
        text_2d = pid_text
        vision_3d = text_2d.unsqueeze(0).expand(3, B_eff, S_eff)  # synthesize 3 chans
    elif pid.ndim == 2:
        # Could be (B, S) text-only OR (C, S) already handled above.
        B_eff, S_eff = pid.size(0), pid.size(1)
        text_2d = pid
        vision_3d = text_2d.unsqueeze(0).expand(3, B_eff, S_eff)  # synthesize vision from text
    elif pid.ndim == 3:
        C, B_eff, S_eff = pid.size(0), pid.size(1), pid.size(2)
        if C == 4:
            # Already (text + 3 vision)
            text_2d = pid[0]
            vision_3d = pid[1:]
        elif C == 3:
            # Vision-only (legacy); synthesize text
            vision_3d = pid
            # Use attention_mask if available, else arange
            if S is None:
                S = S_eff
            text_2d = _make_text_positions_from_mask(attention_mask, S, device)
            if text_2d.size(0) != B_eff:
                text_2d = text_2d.expand(B_eff, -1)
        else:
            raise ValueError(f"Unsupported position_ids channels: expected 3 or 4, got {C}.")
    else:
        raise ValueError("position_ids must have ndim in {1,2,3}.")

    # Ensure shapes align with batch inferred from mask when present
    if attention_mask is not None and attention_mask.ndim == 2:
        B_mask, S_mask = attention_mask.shape
        if text_2d.size(0) != B_mask or text_2d.size(1) != S_mask:
            # center crop / pad to match attention_mask, conservative route:
            S_min = min(S_mask, text_2d.size(1))
            text_2d = text_2d[:, :S_min]
            vision_3d = vision_3d[:, :, :S_min]

    # Compose output according to Transformers version expectation
    if is_transformers_version_in_range(min_version="4.54.0"):
        # Need (4, B, S): [text, time, height, width]
        out = torch.cat([text_2d.unsqueeze(0), vision_3d], dim=0)
        return out
    else:
        # Need (3, B, S): only the 3D vision channels
        return vision_3d


@dataclass
class Qwen2VLCausalLMOutputForPPO(Qwen2VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen2_vl_base_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    return self.language_model(input_ids=None, **kwargs)


def qwen2_vl_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # --- UI-S1 compat: normalize here so both old/new callers work
    norm_position_ids = process_position_ids(position_ids, attention_mask=attention_mask)

    if is_transformers_version_in_range(min_version="4.52.0"):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=norm_position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
    else:
        inputs_embeds, attention_mask = _get_input_embeds(
            self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
        )
        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=norm_position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def forward_with_normal_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen2VLCausalLMOutputWithPast":
    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen2VLCausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: Qwen2VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> tuple | Qwen2VLCausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = qwen2_vl_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen2VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
