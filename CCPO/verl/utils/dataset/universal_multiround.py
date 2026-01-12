import copy
import logging
import os
import random
import re
import time
import traceback
import uuid
from collections import defaultdict
from typing import Any, List, Optional, Union

from datetime import datetime
import json

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from x.data.agent.json import JsonFormat
from x.io import JsonWrap
from x.parallel.parallel_task import ParallelTask
from x.qwen.data_format import slim_messages

from PIL import Image, ImageOps
import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import copy
import time

class QwenMessages2Inputs():
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig, processor: Any | None = None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
      
        self.max_pixels = 12800*28*28
        self.min_pixels = 4*28*28
        self.num_image_limit = config.get("num_image_limit", 4)

        self.max_prompt_length = config.get("max_prompt_length", 32768)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)

    
    def _apply_bboxes_to_images(self, messages, images, bboxes):
        """
        images : List[PIL.Image.Image]
        bboxes : List[Optional[List[float]]]  # [[x1,y1,x2,y2], ...], one per image (or None to drop)
        returns: List[PIL.Image.Image] (cropped, with any None-box images omitted)
        Side effect: removes from `messages` the image items whose bbox is None.
        """
        if bboxes is None:
            return images
    
        if len(bboxes) != len(images):
            raise ValueError(f"len(bboxes)={len(bboxes)} must equal len(images)={len(images)}")
    
        # --- Find each image position in messages (msg_index, content_index) ---
        image_positions = []
        for mi, msg in enumerate(messages):
            parts = msg.get("content") or []
            for ci, part in enumerate(parts):
                if isinstance(part, dict) and "image" in part:
                    image_positions.append((mi, ci))
    
        if len(image_positions) != len(images):
            raise ValueError(
                f"Found {len(image_positions)} image(s) in messages, but got {len(images)} image(s). "
                "These must be equal so we can align bboxes to message images."
            )
    
        cropped: list = []
        to_remove: list[tuple[int, int]] = []  # (msg_index, content_index) to delete
    
        # --- Apply crops (or mark for removal if bbox is None) ---
        for (img, box, (mi, ci)) in zip(images, bboxes, image_positions):
            # If bbox is None -> remove the image from messages and skip from output
            if box is None:
                to_remove.append((mi, ci))
                continue
    
            # Respect EXIF orientation so coords match what you see
            img = ImageOps.exif_transpose(img)
    
            # If bbox is malformed, keep the original image (no removal)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                cropped.append(img)
                continue
    
            w, h = img.size
            x1, y1, x2, y2 = box
    
            # Auto-detect normalized boxes (all in [0,1])
            if 0.0 <= min(x1, y1, x2, y2) and max(x1, y1, x2, y2) <= 1.0:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
    
            # Ensure integers and correct ordering
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
    
            # Clamp to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
    
            # Skip degenerate crops: fall back to original
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                cropped.append(img)
            else:
                cropped.append(img.crop((x1, y1, x2, y2)))
    
        # --- Remove images with None bboxes from messages (do this after looping) ---
        if to_remove:
            # group removals per message and delete in reverse index order to avoid reindexing issues
            per_msg: dict[int, list[int]] = {}
            for mi, ci in to_remove:
                per_msg.setdefault(mi, []).append(ci)
            for mi, idxs in per_msg.items():
                idxs.sort(reverse=True)
                content_list = messages[mi].get("content") or []
                for ci in idxs:
                    if 0 <= ci < len(content_list):
                        del content_list[ci]
    
        return cropped


    
    def __call__(self, state):
        messages = state['messages']
        check_options = state['check_options']
        # Optional: bboxes comes from state (one per image in messages)
        bboxes = state.get('previous_img_bbox', None)
        
        row_dict = {}
        messages = slim_messages(messages, num_image_limit=self.num_image_limit)
        last_image_ele = None
        for msg in messages:
            for content in msg['content']:
                if 'image' in content:
                    if 'min_pixels' not in content:
                        content['min_pixels'] = self.min_pixels
                    if 'max_pixels' not in content:
                        content['max_pixels'] = self.max_pixels
                    last_image_ele = content
        assert messages[-1]['role'] == 'user'
        assert self.processor is not None

        from verl.utils.dataset.vision_utils import (process_image, process_video)

        multi_modal_data = {}
        image_inputs, video_inputs = process_vision_info(messages)
        
        assert 0 < len(image_inputs) <= self.num_image_limit
        
        if len(bboxes) >= len(image_inputs):
            bboxes = bboxes[-len(image_inputs):]       
        else:
            print(f"bboxes {bboxes}")
            print(f"image {len(image_inputs)}")
            bboxes = [None] * (len(image_inputs)-len(bboxes)) + bboxes
        
        # Apply bboxes
        if bboxes is not None:
            # print(f"image length inputs {len(image_inputs)}")
            image_inputs = self._apply_bboxes_to_images(messages, image_inputs, bboxes)

        # (optional) keep per-image crop sizes for debugging/analysis
        cropped_sizes = [im.size for im in image_inputs]

        width, height = last_image_ele['width'], last_image_ele['height']
        resized_width, resized_height = image_inputs[-1].size
        
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False

        
        model_inputs = self.processor(
            text=[raw_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt"
        )

        # If your tokenizer exposes an image token id, use it; otherwise your magic number
        image_token_id = getattr(getattr(self.processor, "tokenizer", None), "image_token_id", 151655)

        if image_inputs is not None:
            # This assert assumes the processor respects min/max pixels when resizing.
            # It will continue to hold after cropping because we pass PILs post-crop.
            expected = sum(round((w*h) / (28*28)) for (w, h) in (im.size for im in image_inputs))
            actual = (model_inputs['input_ids'] == image_token_id).sum()
            # assert expected == actual, f"Image token mismatch: expected {expected} got {int(actual)}"

        multi_modal_data = {'image': image_inputs}
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        model_inputs.pop("second_per_grid_ts", None)
        
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        assert self.processor.image_processor.__class__.__name__ != "Qwen2_5VLImageProcessor"
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
            
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict['reward_model'] = {"style": "rule", "ground_truth": check_options}

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = message_translate(messages, to_format="openai")

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        if 'extra_info' not in row_dict:
            row_dict['extra_info'] = {}
        row_dict['extra_info']['resized_width'] = resized_width
        row_dict['extra_info']['resized_height'] = resized_height
        row_dict['extra_info']['width'] = width
        row_dict['extra_info']['height'] = height
        # NEW: record all cropped sizes (useful if you later want per-image stats)
        row_dict['extra_info']['cropped_sizes'] = cropped_sizes
        
        # print(f"Cropped images size: {cropped_sizes}")

        return row_dict



class StdTrajectory():
    def __init__(self, line,actions_only,hint) -> None:
        self.line = line[()]
        self.num_steps = len(self.line['steps'])
        from x.data.agent.space.std_space import RAW_SPACE
        # self.fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True,actions_only=actions_only,hint=hint)
        self.fm = JsonFormat(RAW_SPACE, add_thought=False, force_add_thought=False,actions_only=actions_only,hint=hint)

        self.state = None

    def get_next(self, model_response):
        state = self.fm.gen_next_round(self.line, self.state, previous_model_response=model_response)
        if state is None:
            return "Finished"
        return state

class MultiRoundGenerator():
    def __init__(self, batch: DataProto, rollout_n, msg_man, patch_threshold=0,actions_only=None,hint=False) -> None:
        self.rollout_n = rollout_n
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.non_tensor_batch["line"]))], dtype=object)
        
        repeat_batch = batch.repeat(repeat_times=self.rollout_n, interleave=True) # need set rollout kwargs to 1
        self.batch = repeat_batch
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(len(self.batch))], dtype=object)
        self.batch.non_tensor_batch["traj_uid"] = traj_uid
        self.task_queue = [StdTrajectory(line,actions_only,hint) for line in self.batch.non_tensor_batch["line"]]
        self.finished = [False for i in range(len(self.task_queue))]
        self.current_response = [None for i in range(len(self.task_queue))]
        self.previous_response = [[] for i in range(len(self.task_queue))]  # History of all previous responses, if we need nAO
        self.previous_ref_response = [[] for i in range(len(self.task_queue))]
        self.previous_img_max_bbox = []
        # Calculate number of unique examples (before rollouts)
        self.num_unique_examples = len(batch.non_tensor_batch["line"])
        # Initialize per-example coordinate tracking
        self.previous_aggregate_coordinates = [[] for _ in range(self.num_unique_examples)]
        self._previous_aggregate_keys = [set() for _ in range(self.num_unique_examples)]
        self.error_num = [0 for i in range(len(self.task_queue))]
        self.msg_man = msg_man
        from x.data.agent.space.std_space import RAW_SPACE
        # self.fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
        self.fm = JsonFormat(RAW_SPACE, add_thought=False, force_add_thought=False)
        self.patch_threshold = patch_threshold
        self.hint = hint

        # Message collection for JSON logging
        self.all_messages = []  # Simple list to store messages with episode IDs
        self.json_log_file = None
        
        print('Finish generator init')


    def _fetch_next(self, ptr):
        if self.finished[ptr]:
            return True, (None, None)
        current_gen = self.task_queue[ptr]
        current_response = self.current_response[ptr]
        state = current_gen.get_next(current_response)

        if state == "Finished":
            return True, ("Finished", state)

        try:
            step = {}
            step['action_content'] = copy.deepcopy(state['check_options'])
            self.previous_ref_response[ptr].append(copy.deepcopy(state['check_options']))
        except Exception:
            traceback.print_exc() 

            
        # print("@@@"*40)
        # Calculate which example group this ptr belongs to
        example_group = ptr // self.rollout_n
        rollout_id = ptr % self.rollout_n
        example_id = ptr // self.rollout_n
        
        # Add debugging info
        state['example_id'] = example_group
        state['rollout_id'] = rollout_id
        state['ptr'] = ptr
        state['rollout_n'] = self.rollout_n
        
        # print(f"DEBUG: ptr={ptr}, example_id={example_id}, rollout_id={rollout_id}, example_group={example_group}")
        
        self.aggregate_previous_coordinates(ptr)
        state['previous_img_bbox'] = [self.coords_to_bbox(coords) for coords in self.previous_aggregate_coordinates[example_group]].copy()
        state['previous_img_bbox'].append([])  # Current Image        
        row_dict = self.msg_man(state)
        row_dict['ptr'] = ptr
        return True, (row_dict, state)
        

    def fetch_batch(self):
        while True:
            batch = []

            tasks = list(range(len(self.task_queue)))
            # input()
            mid_result = ParallelTask((tasks,), self._fetch_next, total=len(tasks), num_process=len(tasks), passing_indices=False, return_list=True).run_and_collect(tqdm_args={"disable": False})
            assert len(mid_result) == len(self.task_queue)
            
            for ptr, res in enumerate(mid_result):
                row_dict, state = res
                if row_dict == None:
                    continue
                    
                self.current_response[ptr]= None
                
                if row_dict == "Finished":
                    self.finished[ptr] = True
                else:
                    self.task_queue[ptr].state = state
                    row_dict['uid'] = self.batch.non_tensor_batch['uid'][ptr]
                    row_dict['traj_uid'] = self.batch.non_tensor_batch['traj_uid'][ptr]
                    row_dict['step_id'] = state['step_id']
                    row_dict['data_source'] = self.batch.non_tensor_batch['data_source'][ptr] if 'data_source' in self.batch.non_tensor_batch else "gui_traj_action_match"
                    row_dict['reward_model'] = {
                        "style": "rule",
                        "ground_truth": {
                            "check_options": state['check_options'],
                            'num_steps': self.task_queue[ptr].num_steps,
                            'thought': state['thought'],
                            }
                    }

                    batch.append(row_dict)
                    
            if len(batch) == 0:
                break
                
            yield collate_fn(batch)



    def apply_response(self, batch):
        failed_num = 0
        for ptr, response, extract_match, reward_model,extra_info in zip(batch.non_tensor_batch['ptr'], batch.batch['responses'], batch.non_tensor_batch['extract_match'], batch.non_tensor_batch['reward_model'], batch.non_tensor_batch['extra_info']):
            response_text = self.msg_man.tokenizer.decode(response, skip_special_tokens=True) 
            
            self.current_response[ptr] = response_text

            model_response = self.fm.parse_response(self.current_response[ptr])
            # Add current response to history before updating current_response
            if self.current_response[ptr] is not None and 'action_content' in model_response:
                self.previous_response[ptr].append(model_response['action_content'])
            else:
                self.previous_response[ptr].append(None)
            
            if not extract_match:
                failed_num += 1
                if self.patch_threshold > self.error_num[ptr] or self.patch_threshold == -1:
                    step = {}
                    step['action_content'] = reward_model['ground_truth']['check_options']
                    keys_to_remove = ['bbox', 'candidate_bbox','annotation','thought']
                    for key in keys_to_remove:
                        step['action_content'].pop(key, None)
                    # print("reward_model['ground_truth']",reward_model['ground_truth'])
                    step['thought'] = reward_model['ground_truth']['thought']
                    ground_truth_response = self.fm.format_response(step,extra_info) # resize coordinate
                    # ground_truth = reward_model['ground_truth']['check_options']
                
                    self.current_response[ptr] = ground_truth_response

                    model_response = self.fm.parse_response(self.current_response[ptr])
                    self.previous_response[ptr][-1] = model_response['action_content']
                    
                    self.error_num[ptr] += 1
                else:
                    self.finished[ptr] = True
                    
        return failed_num

        
    def get_previous_responses(self, ptr):
        """Get the history of previous responses for a specific pointer"""
        return self.previous_response[ptr].copy()
    
    def get_all_previous_responses(self):
        """Get all previous responses for all pointers"""
        return [responses.copy() for responses in self.previous_response]           
                
    def reset_previous_responses(self, ptr):
        """Reset previous responses for a specific pointer when trajectory is finished"""
        # print(f"{ptr} |{ptr // self.rollout_n} has been reset_previous_responses")
        self.previous_response[ptr] = []

    def get_previous_ref_responses(self, ptr):
        """Get the history of reference responses for a specific pointer"""
        return self.previous_ref_response[ptr].copy()
    
    def get_all_previous_ref_responses(self):
        """Get all reference responses for all pointers"""
        return [responses.copy() for responses in self.previous_ref_response]
    
    def reset_previous_ref_responses(self, ptr):
        """Reset reference responses for a specific pointer when trajectory is finished"""
        # print(f"{ptr} |{ptr // self.rollout_n} has been reset_previous_responses")
        self.previous_ref_response[ptr] = []
        
    def reset_all_previous_ref_responses(self):
        """Reset reference responses for all pointers"""
        self.previous_ref_response = [[] for i in range(len(self.task_queue))]

    def reset_all_previous_responses(self):
        """Reset reference responses for all pointers"""
        self.previous_response = [[] for i in range(len(self.task_queue))]
    
    def reset_previous_aggregate_coordinates(self, ptr=None):
        """
        Reset previous aggregate coordinates.
        If ptr is provided, resets only for that example group.
        Otherwise resets all.
        """
        if ptr is not None:
            example_group = ptr // self.rollout_n
            self.previous_aggregate_coordinates[example_group] = []
            self._previous_aggregate_keys[example_group] = set()
        else:
            # Reset all
            self.previous_aggregate_coordinates = [[] for _ in range(self.num_unique_examples)]
            self._previous_aggregate_keys = [set() for _ in range(self.num_unique_examples)]
         
    def add_coordinates(self, response):
        """Collect unique [x, y] pairs from a single response."""
        coords, seen = [], set()
    
        def _add_pair(xy):
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                return
            if not isinstance(xy[0], (int, float)) or not isinstance(xy[1], (int, float)):
                return
            t = (xy[0], xy[1])
            if t not in seen:
                seen.add(t)
                # keep as lists for downstream compatibility
                coords.append([xy[0], xy[1]])
    
        if isinstance(response, dict) and response.get("action") in {"click", "long_press", "swipe"}:
            _add_pair(response.get("coordinate"))
            _add_pair(response.get("coordinate2"))
    
            bbox = response.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                _add_pair(bbox[:2])
                _add_pair(bbox[2:])
    
            candidate_bbox = response.get("candidate_bbox")
            if isinstance(candidate_bbox, (list, tuple)):
                for bbox in candidate_bbox:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        _add_pair(bbox[:2])
                        _add_pair(bbox[2:])
    
        return coords
    
    
    def aggregate_previous_coordinates(self, ptr):
        """
        Aggregate unique coordinates from previous responses and reference responses.
        Only aggregates coordinates from rollouts of the same example.
        Ensures:
          - No duplicate [x, y] in aggregate_coords
          - No duplicate aggregate entries in self.previous_aggregate_coordinates
        """
        # Calculate which example group this ptr belongs to
        example_group = ptr // self.rollout_n
        
        pr  = self.get_all_previous_responses()        or []
        prr = self.get_all_previous_ref_responses()    or []
    
        aggregate_coords, seen = [], set()
    
        def _extend_unique(pairs):
            for xy in pairs:
                if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                    continue
                t = (xy[0], xy[1])
                if t not in seen:
                    seen.add(t)
                    aggregate_coords.append([xy[0], xy[1]])
    
        # Only aggregate from trajectories in the same example group
        # Example group has rollout_n trajectories starting at example_group * rollout_
        index = [example_group * self.rollout_n + i for i in range(self.rollout_n)]
        
        # Important! prev_reference added at runtime
        prr_len = [len(prr[i]) for i in index]
        prr_idx = prr_len[0] - 1 if all(x == prr_len[0] for x in prr_len) else min(prr_len)
        prr_idx = 0 if prr_idx < 0 else prr_idx - 1
        cur_round = len(pr[index[0]])
                
        for i in range(self.rollout_n):
            idx = example_group * self.rollout_n + i
            # Reference Check    
            if idx < len(prr) and len(prr[idx]) > 0:
                _extend_unique(self.add_coordinates(prr[idx][prr_idx]))

        for i in range(self.rollout_n):
            # Response Check  
            idx = example_group * self.rollout_n + i
            if idx < len(pr) and len(pr[idx]) > 0:
                _extend_unique(self.add_coordinates(pr[idx][-1]))

        # only store if this aggregate hasn't been seen before for this example group
        # build a canonical key for this aggregate (order-insensitive)
        
        key = tuple(sorted(seen))  # e.g., ((10,20),(30,40),...)
    
        if len(self.previous_aggregate_coordinates[example_group]) < cur_round:
            self.previous_aggregate_coordinates[example_group].append(aggregate_coords)
            
        return aggregate_coords

    
    def coords_to_bbox(self, coords):
        """
        coords: iterable of [x, y]
        returns: [min_x, min_y, max_y, max_x] or None if invalid/degenerate / without validation
        """
        if coords == []:
            return None
                    
        xs, ys = [], []
        for p in coords or []:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                x, y = p
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    xs.append(float(x)); ys.append(float(y))
        if not xs:
            return None

        mnx = min(xs) - 28
        mny = min(ys) - 28
        mxx = max(xs) + 28
        mxy = max(ys) + 28

        # mnx = min(xs)
        # mny = min(ys)
        # mxx = max(xs)
        # mxy = max(ys)
        return [mnx, mny, mxx, mxy] 

    
    def _save_message(self, ptr, messages):
        """Save message with episode ID"""
        episode_id = f"episode_{ptr}"
        message_data = {
            'episode_id': episode_id,
            'messages': messages
        }
        self.all_messages.append(message_data)
    
    def save_messages_to_json(self, filename=None):
        """Save all messages to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"messages_{timestamp}.json"
                
        with open(filename, 'a', encoding='utf-8') as f:
            for message in self.all_messages:
                json.dump(message, f, ensure_ascii=False)
                f.write('\n')  # 换行分隔
                
        print(f"Saved {len(self.all_messages)} messages to {filename}")
        return filename


def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line



if __name__ == "__main__":
    from x.io import read_json
    lines = read_json("/mnt/HithinkOmniSSD/user_workspace/songyurun/UI-S1/datasets/android_control_train_example.jsonl")
    batch_lines = lines[:1]

    print(batch_lines)
    print("----"*15)
    
    msg_man = QwenMessages2Inputs(
        hf_tokenizer("/mnt/HithinkOmni/user_workspace/songyurun/models/Qwen2.5-VL-7B-Instruct"),
        {},
        hf_processor("/mnt/HithinkOmni/user_workspace/songyurun/models/Qwen2.5-VL-7B-Instruct")
    )
    
    batch_dict = collate_fn([
        {'line': np.array(fix_line(line), dtype=object)}
        for line in batch_lines])
    
    batch = DataProto.from_single_dict(batch_dict)

    #print(batch)
    i = 0
    mr_gen = MultiRoundGenerator(batch, rollout_n=2, msg_man=msg_man)
    for sub_batch in mr_gen.fetch_batch():
        
        print("=" * 50)
        print("BATCH INFO:")
        print("=" * 50)
        sub_batch = DataProto.from_single_dict(sub_batch)
        print(f"Number of items in batch: {len(sub_batch.non_tensor_batch.get('ptr', []))}")
        
        # This is used for compute one batch/rollout step 
        for ptr in sub_batch.non_tensor_batch['ptr']:
            mr_gen.current_response[ptr] = '<think>\nxxx\n</think>\n<action>\n{\"action\": \"click' +'\", \"coordinate\": [2'+ str(ptr)+', 6'+ str(i)+']}\n</action>'
        
            print("Finish step: " + str(i) + "++"*20)
            
            model_response = mr_gen.fm.parse_response(mr_gen.current_response[ptr])
            
            # Add current response to history before updating current_response
            if mr_gen.current_response[ptr] is not None:
                mr_gen.previous_response[ptr].append(model_response['action_content'])
            else:
                mr_gen.previous_response[ptr].append(None)
            
        i+=1

    ## calculate reward
