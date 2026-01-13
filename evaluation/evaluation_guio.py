import json
import os
import re
import copy
from collections import defaultdict
from typing import Sequence

IGNORE_INDEX = -100
import itertools
import math
import logging

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format

```
<action> ... </action>
```

## Action Space

You can perform the following actions:
- CLICK: Click the point on the screen with specified (x, y) coordinates.
- SCROLL: Swipe from starting point with specified (x, y) coordinates to endpoint with specified (x2, y2) coordinates.
- LONG_PRESS: Long press at the specified coordinate.
- TYPE: Input the specified text into the activated input box.
- COMPLETE: Mark the task as successfully completed.
- IMPOSSIBLE: Indicate that the task cannot be completed.
- PRESS_HOME / PRESS_BACK / PRESS_RECENT: Press the corresponding system button.

The arguments you can use are:
- coordinate: (x, y): The x and y pixels coordinates from the left and top edges.
- coordinate2: (x, y): The x and y pixels coordinates from the left and top edges for the endpoint of a swipe.
- text: Text input required by the TYPE action.

Format your output as a JSON object with the selected action and its arguments at the same level.

Example outputs:
<action>
{"action": "CLICK", "coordinate": [512, 982]}
</action>
<action>
{"action": "SCROLL", "coordinate": [520, 1480], "coordinate2": [520, 420]}
</action>

## Note

- Planing the task and explain your reasoning step-by-step in `think` part.
- Write your action in the `action` part according to the action space.
"""

from PIL import Image, ImageDraw
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import math
import logging

import pdb
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="eval_results/",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data6/GUIModels/model/qwen25vl_guiodyssey",
                    help='The path where the model is loaded from')
parser.add_argument('--input_path', type=str, default='/mnt/HithinkOmniSSD/user_workspace/yinjiong/dataset/GUIOdyssey/multi_turn_anno/new_guiodyssey/for_rl/test_goal_steps_random_split_sorted.jsonl',
                    help='The path where the model is loaded from')
parser.add_argument('--batch_size', type=int, default=1,
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=1,
                    help='The path where the model is loaded from')
parser.add_argument('--use_multi_gpu', action="store_true",
                    help='The path where the model is loaded from')
parser.add_argument('--pred_coord_mode', type=str, default='auto',
                    choices=['auto', 'pixel', 'normalized'],
                    help='Interpretation of predicted coordinates: auto-detect, force pixel-to-normalized conversion, or treat as already normalized.')
device = 'auto'
# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + args.model_path.split('/')[-1] + '.json'
save_dir = os.path.dirname(args.save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)

import torch

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training if using multi-GPU
if args.use_multi_gpu:
    # Set up distributed training
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize distributed process group
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    print(f"Initialized distributed inference: local_rank={local_rank}, rank={rank}, world_size={world_size}")
else:
    device = 'cuda:0'
    rank = 0
    world_size = 1
    print("Using single GPU")

print("Loading model...")
if args.use_multi_gpu:
    # Multi-GPU loading with device_map="auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},  # One full copy per rank
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # print(f"Model loaded with device_map: {model.device_map}")
else:
    # Single GPU loading
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Model loaded on single GPU")

# -------------------------- 强制修改生成配置 --------------------------
# Force deterministic decoding regardless of checkpoint generation config
gen_cfg = getattr(model, "generation_config", None)
if gen_cfg is not None:
    gen_cfg.do_sample = False
    for attr in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, None)
# -------------------------- 强制修改生成配置 --------------------------


# Simple memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Available GPUs: {torch.cuda.device_count()}")



# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     args.model_path , torch_dtype=torch.bfloat16, device_map=device
# )




min_pixels = 256*28*28
max_pixels = 12800*28*28

# min_pixels = 4*28*28
# max_pixels = 12800*28*28

processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_tags(xml_content, tag_names):
    result = {}
    
    for tag_name in tag_names:
        # Define a regex pattern to match content for the current tag
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        
        # Use re.search to find the first match of pattern in xml_content
        match = re.search(pattern, xml_content, re.DOTALL)
        
        if match:
            # Extract and return the captured content within the tags
            tag_content = match.group(1).strip()
            result[tag_name] = tag_content
        else:
            result[tag_name] = None
    
    return result


# 坐标转换: 从调整后的坐标到原始坐标
def resize_coordinate(coordinate, source_size, target_size):
    x, y = coordinate
    target_width,  target_height = target_size
    source_width, source_height = source_size
    # 计算比例
    width_ratio = target_width / source_width
    height_ratio =  target_height / source_height
    # 转换坐标
    target_x = x * width_ratio
    target_y = y * height_ratio
    return [target_x, target_y]


def convert_originl_resize_coordinate(action_content, image_ele):
    width, height = image_ele['width'], image_ele['height']
    resized_width, resized_height = image_ele['resized_width'], image_ele['resized_height']
    
    if 'coordinate' in action_content:
        action_content['coordinate'] = resize_coordinate(action_content['coordinate'], (width, height), (resized_width, resized_height))
        action_content['coordinate'] = list(map(round, action_content['coordinate']))
    if 'coordinate2' in action_content:
        action_content['coordinate2'] = resize_coordinate(action_content['coordinate2'], (width, height), (resized_width, resized_height)) 
        action_content['coordinate2'] = list(map(round, action_content['coordinate2']))
    return action_content

def convert_resize_original_coordinate(action_content, image_ele):
    width, height = image_ele['width'], image_ele['height']
    resized_width, resized_height = image_ele['resized_width'], image_ele['resized_height']
    
    if 'coordinate' in action_content:
        action_content['coordinate'] = resize_coordinate(action_content['coordinate'], (resized_width, resized_height), (width, height))
        action_content['coordinate'] = list(map(round, action_content['coordinate']))
    if 'coordinate2' in action_content:
        action_content['coordinate2'] = resize_coordinate(action_content['coordinate2'], (resized_width, resized_height), (width, height)) 
        action_content['coordinate2'] = list(map(round, action_content['coordinate2']))
    return action_content

def normalize_action_coordinates(action_content, image_ele, mode='auto'):
    action = copy.deepcopy(action_content)
    width = image_ele.get('width')
    height = image_ele.get('height')
    resized_width = image_ele.get('resized_width', width)
    resized_height = image_ele.get('resized_height', height)
    mode = mode or 'auto'

    def _normalize(coord):
        if not coord or len(coord) < 2 or width in (None, 0) or height in (None, 0):
            return coord
        try:
            x = float(coord[0])
            y = float(coord[1])
        except (TypeError, ValueError):
            return coord
        if mode != 'pixel' and max(abs(x), abs(y)) <= 1000:
            return [x, y]
        resized_width_valid = resized_width if resized_width else width
        resized_height_valid = resized_height if resized_height else height
        orig_x = x * width / resized_width_valid if resized_width_valid else x
        orig_y = y * height / resized_height_valid if resized_height_valid else y
        norm_x = orig_x / width * 1000 if width else orig_x
        norm_y = orig_y / height * 1000 if height else orig_y
        norm_x = max(min(norm_x, 1000.0), 0.0)
        norm_y = max(min(norm_y, 1000.0), 0.0)
        return [round(norm_x, 3), round(norm_y, 3)]

    if mode == 'normalized':
        return action

    if 'coordinate' in action:
        action['coordinate'] = _normalize(action['coordinate'])
    if 'coordinate2' in action:
        action['coordinate2'] = _normalize(action['coordinate2'])
    return action

def format_action(action_content, image_ele):
    action = copy.deepcopy(action_content)
    action = deal_with_coordinate(action, image_ele)
    return json.dumps(action, ensure_ascii=False)


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score




def _point_in_bbox(coord: Sequence[float], bboxes: Sequence[Sequence[float]]) -> bool:
    if coord is None or not bboxes:
        return False
    x, y = coord
    for box in bboxes:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def new_action_matching_evaluation(outputs):
    """Evaluation adapted for GUI Odyssey action space."""
    step_acc_res_dict = defaultdict(int)
    sample_number_dict = defaultdict(int)

    for sample in outputs:
        pred = sample['pred']
        gt = sample['gt']

        try:
            pred_tags = parse_tags(pred, ['action'])
            pred_action_content = pred_tags.get('action', '')
            if pred_action_content:
                pred_action = json.loads(pred_action_content)
            else:
                pred_action = {}
        except:
            pred_action = {}

        try:
            gt_tags = parse_tags(gt, ['action'])
            gt_action_content = gt_tags.get('action', '')
            if gt_action_content:
                gt_action = json.loads(gt_action_content)
            else:
                gt_action = {}
        except:
            gt_action = {}

        gt_action_type = str(gt_action.get('action', '')).upper()
        pred_action_type = str(pred_action.get('action', '')).upper()

        sample_number_dict["full"] += 1
        sample_number_dict[gt_action_type] += 1

        if gt_action_type == pred_action_type:
            step_acc_res_dict["type_match"] += 1
            step_acc_res_dict[gt_action_type + "_type_match"] += 1

            # image_info = sample.get('image_info') or {}
            # width = image_info.get('width', 1080)
            # height = image_info.get('height', 1920)
            # threshold = 0.14 * max(width, height)

            # candidate_bbox = (
            #     (sample.get('more_info') or {}).get('check_options') or {}
            # ).get('candidate_bbox') or []

            full_match = False
            match_key = None

            if gt_action_type in ["CLICK", "LONG_PRESS"]:
                step_acc_res_dict["click_type_match"] += 1
                match_key = f"{gt_action_type}_all_match"
                try:
                    pred_x, pred_y = pred_action['coordinate'][0], pred_action['coordinate'][1]
                except:
                    pred_x, pred_y = 0, 0
                # gt_coord = gt_action.get('coordinate')
                # pred_coord = pred_action.get('coordinate')
                gt_x, gt_y = gt_action['coordinate'][0], gt_action['coordinate'][1]

                if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14*1000:
                    full_match = True

            elif gt_action_type == "SCROLL":
                match_key = f"{gt_action_type}_all_match"
                gt_start = gt_action.get('coordinate')
                gt_end = gt_action.get('coordinate2')
                pred_start = pred_action.get('coordinate')
                pred_end = pred_action.get('coordinate2')
                # 比较方向
                def get_direction(start, end):
                    if (
                        not isinstance(start, (list, tuple))
                        or not isinstance(end, (list, tuple))
                        or len(start) < 2
                        or len(end) < 2
                    ):
                        return None
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    angle = math.atan2(dy, dx) * 180 / math.pi
                    if -45 <= angle <= 45:
                        return "right"
                    elif 45 < angle <= 135:
                        return "down"
                    elif angle > 135 or angle < -135:
                        return "left"
                    else:
                        return "up"
                gt_direction = get_direction(gt_start, gt_end)
                pred_direction = get_direction(pred_start, pred_end)
                if gt_direction is not None and gt_direction == pred_direction:
                    full_match = True

            elif gt_action_type == "TYPE":
                match_key = f"{gt_action_type}_all_match"
                gt_text = gt_action.get('text', '')
                pred_text = pred_action.get('text', '')
                full_match = calculate_f1_score(pred_text, gt_text) > 0.5

            elif gt_action_type in {"PRESS_HOME", "PRESS_BACK", "PRESS_RECENT", "COMPLETE", "IMPOSSIBLE"}:
                full_match = True
                match_key = f"{gt_action_type}_all_match"

            if full_match:
                step_acc_res_dict["full"] += 1
                if match_key:
                    step_acc_res_dict[match_key] += 1
                if gt_action_type in ["CLICK", "LONG_PRESS"]:
                    step_acc_res_dict["click_all_match"] += 1

    total = sample_number_dict["full"] or 1
    click_type = step_acc_res_dict["click_type_match"]
    grounding = step_acc_res_dict["click_all_match"] / click_type if click_type else 0.0

    logger.info("="*30 + " GUI Odyssey Step Acc " + "="*30)
    logger.info("Acc: %f", step_acc_res_dict["full"] / total)
    logger.info("type_match acc: %f", step_acc_res_dict["type_match"] / total)
    logger.info("grounding acc: %f", grounding)

    return {
        'step_acc': step_acc_res_dict["full"] / total,
        'type_match_acc': step_acc_res_dict["type_match"] / total,
        'grounding_acc': grounding,
        'total_actions': sample_number_dict["full"],
        'step_acc_res_dict': dict(step_acc_res_dict),
        'sample_number_dict': dict(sample_number_dict)
    }



def action2step(step_data):
    action_type = str(step_data.get("action", "")).upper()

    if action_type in {"CLICK", "LONG_PRESS"}:
        coord = [float(i) for i in step_data.get('coordinate', [0, 0])]
        result = {"action": action_type, "coordinate": coord}
        if action_type == "LONG_PRESS" and "time" in step_data:
            result["time"] = step_data["time"]
        return result

    if action_type in {"SCROLL", "SWIPE"}:
        coord = [float(i) for i in step_data.get('coordinate', [0, 0])]
        coord2 = [float(i) for i in step_data.get('coordinate2', coord)]
        return {"action": "SCROLL", "coordinate": coord, "coordinate2": coord2}

    if action_type == "TYPE":
        return {"action": "TYPE", "text": step_data.get('text', "")}

    if action_type in {"PRESS_HOME", "PRESS_BACK", "PRESS_RECENT"}:
        return {"action": action_type}

    if action_type in {"COMPLETE", "IMPOSSIBLE"}:
        return {"action": action_type}

    return {"action": action_type}
        

def action_matching_evaluation(pred_output, metric='macro'):
    """Main evaluation function for Android Control - uses new_action_matching_evaluation"""
    # Use the new evaluation method that matches evaluation_unify_AC.py exactly
    metrics = new_action_matching_evaluation(pred_output)
    return metrics


from PIL import Image
def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=12800 * 28 * 28):
    """Process image for model input"""
    messages = [
        {"role": "user", 
         "content": [
            {
               "type": "image", 
               "image": image_path,
               "min_pixels": min_pixel,
               "max_pixels": max_pixel,
            }
            ]
        }
    ]

    # print(Image.open(image_path))
    # image_input, _ = process_vision_info_with_resize(messages)
    # print(image_input)
    
    image_input, _ = process_vision_info(messages)
    return image_input[-1]



import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
import numpy as np

# Load test data - match ac_process_999_sequence_llavaformat.py format
with open(args.input_path, 'r') as f:
    ac_test_full = [json.loads(line) for line in f.readlines()]

total_episodes = len(ac_test_full)
if world_size > 1:
    def _split_dataset(dataset, rank, world_size):
        total = len(dataset)
        start = (total * rank) // world_size
        end = (total * (rank + 1)) // world_size
        return dataset[start:end]
    ac_test = _split_dataset(ac_test_full, rank, world_size)
    print(f"[Rank {rank}] processing {len(ac_test)} / {total_episodes} episodes")
else:
    ac_test = ac_test_full

score_average = 0
all_save_results = []
all_eval_results = []
outputs = []

# Define batch size (adjust based on GPU memory)
BATCH_SIZE = args.batch_size
image_cache = {}


def get_image_info_cached(image_path, min_pixel=min_pixels, max_pixel=max_pixels):
    """Cached version of get_image_info to avoid reloading images"""
    if image_path not in image_cache:
        # Get the processed image
        processed_image = get_image_info(image_path, min_pixel, max_pixel)
        
        # Get original image dimensions
        from PIL import Image
        original_img = Image.open(image_path)
        original_width, original_height = original_img.size
        
        # Get resized dimensions from processed image
        resized_width, resized_height = processed_image.size
        
        image_cache[image_path] = {
            'processed_image': processed_image,
            'width': original_width,
            'height': original_height,
            'resized_width': resized_width,
            'resized_height': resized_height
        }
        # print(image_cache[image_path])
    return image_cache[image_path]


def batch_generate_grounding(batch_prompts, batch_image_paths):
    """Process a batch of examples simultaneously"""
    texts = []
    all_images = []
    batch_image_info = []  # Store image info for coordinate conversion

    for conversations, image_paths in zip(batch_prompts, batch_image_paths):
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        for conv in conversations:
            if conv["from"] == "human":
                # Handle human messages with potential images
                content = [{"type": "text", "text": conv["value"]}]
                messages.append({"role": "user", "content": content})
            elif conv["from"] == "assistant":
                messages.append({"role": "assistant", "content": conv["value"]})
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN)

        # print(text)
        
        texts.append(text)

        images_for_example = []
        example_image_info = []  # Store image info for this example
        for img_path in image_paths:
            img_info = get_image_info_cached(img_path)
            images_for_example.append(img_info['processed_image'])
            # Store the full image info for coordinate conversion
            example_image_info.append({
                'width': img_info['width'],
                'height': img_info['height'],
                'resized_width': img_info['resized_width'],
                'resized_height': img_info['resized_height']
            })
        all_images.append(images_for_example)
        batch_image_info.append(example_image_info)

    inputs = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Get the token ID for <|im_end|>
    eos_token_id = processor.tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]

    # Generate with explicit stopping criteria
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=eos_token_id,  # Stop when this token is generated
        pad_token_id=processor.tokenizer.pad_token_id,
        num_return_sequences=1,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    # Clean up the output by removing everything after <|im_end|>
    cleaned_outputs = []
    for text in output_texts:
        # Find the position of <|im_end|>
        end_pos = text.find("<|im_end|>")
        if end_pos != -1:
            # Remove everything including and after <|im_end|>
            cleaned_text = text[:end_pos].strip()
        else:
            # If no stop token found, use the whole text but remove any trailing special tokens
            cleaned_text = text.strip().replace("<|im_end|>", "")
        cleaned_outputs.append(cleaned_text)

    return cleaned_outputs, batch_image_info



user_response = "User Instruction: {}.\nOutput Format: <action> ... </action>\n"

additional = 'If the query asks a question, please answer the question through the answer action before terminating the process.\n'

step_i = 0
batch_prompts = []
batch_image_paths = []
batch_steps = []
outputs = []

BATCH_SIZE = args.batch_size

for j, episode in enumerate(tqdm(ac_test)):

    previous_actions = []
    previous_imgs = []

    for step in episode['steps']:
        
        img_filename = step["screenshot"]
        
        img_path = img_filename
        if not os.path.exists(img_path):
            print('image not found')
            continue

        goal = episode["goal"]

        prompt = user_response.format(goal)
        
        conversations = []

        prompt_value = prompt + ("\n" + additional if additional else "")
        conversations_his = [{"value": prompt_value + "\n", "from": "human"}]
        cur_step_preimg = previous_imgs[-args.his_num:]
        cur_step_idx = len(previous_imgs[-args.his_num:])
        cur_all_imgs = []

        for i, action in enumerate(previous_actions[-args.his_num:]):
            conversations_his[-1]["value"] += "<image>\n"
            conversations_his.append({"value": action, "from": "assistant"})
            conversations_his.append({"value":"Output Format: <action> ... </action>\n", "from": "human"})
            cur_all_imgs.append(previous_imgs[-args.his_num:][i])

        action_step = '<action>' + json.dumps(action2step(step['action_content'])) + '</action>'
        previous_actions.append(action_step)
        previous_imgs.append(img_path)

        conversations.extend(conversations_his)
        conversations[-1]["value"] += "<image>\n"
        # conversations.append({"value": str(action_step), "from": "assistant"})
        
        cur_all_imgs.append(img_path)
        # Add to batch
        batch_prompts.append(conversations)
        batch_image_paths.append(cur_all_imgs)

        batch_steps.append(step)
        step_i += 1

        
        # print('===='*20)
        # print(step)
        # print("prompts conversation:")
        # for i in conversations:
        #     print(i)
        # print(cur_all_imgs)

        # # Process batch when full or at last item
        if len(batch_prompts) == BATCH_SIZE or j == len(ac_test) - 1:
            # Process batch
            responses, batch_image_info = batch_generate_grounding(batch_prompts, batch_image_paths)
            
            # Process each response in batch
            for idx, resp in enumerate(responses):
                step_info = batch_steps[idx]

                # Use stored image dimensions for coordinate conversion (no need to call get_image_info_cached again)
                image_ele = batch_image_info[idx][-1]  # Get the last image info (current screenshot)

                # Apply coordinate conversion to predicted action
                
                try:
                    pred_tags = parse_tags(resp, ['action'])
                    pred_action_content = pred_tags.get('action', '')
                    if pred_action_content:
                        pred_action = json.loads(pred_action_content)
                        pred_action['action'] = str(pred_action.get('action', '')).upper()
                        if 'coordinate' in pred_action and isinstance(pred_action['coordinate'], list):
                            pred_action['coordinate'] = [float(x) for x in pred_action['coordinate']]
                        if 'coordinate2' in pred_action and isinstance(pred_action['coordinate2'], list):
                            pred_action['coordinate2'] = [float(x) for x in pred_action['coordinate2']]
                        if 'text' in pred_action and isinstance(pred_action['text'], str):
                            pred_action['text'] = pred_action['text']
                        pred_action = normalize_action_coordinates(pred_action, image_ele, mode=args.pred_coord_mode)
                        resp = '<action>' + json.dumps(pred_action, ensure_ascii=False) + '</action>'
                except:
                    # If parsing fails, keep original response
                    pass

                # print("****"*20)
                # print(step_info["screenshot"])
                # print(image_ele)
                # print("%%%"*20)
                # print(json.loads(pred_action_content))
                # print(resp)
                # print('<action>' + json.dumps(action2step(step['action_content'])) + '</action>')
                
                outputs.append({
                    'question': batch_prompts[idx],
                    'image': step_info["screenshot"],
                    'pred': resp,
                    'gt': '<action>' + json.dumps(action2step(step['action_content']), ensure_ascii=False) + '</action>',
                    'more_info': {
                        'category': step_info.get('category', 'unknown'),
                        'episode_id': step_info.get('episode_id', 0),
                        'check_options': step_info.get('check_options', None)
                    },
                    'image_info': image_ele,
                })

            # Reset batch
            batch_prompts = []
            batch_image_paths = []
            batch_steps = []

    
# Evaluation
print(f"Evaluating ...")
print(len(outputs))
if world_size > 1:
    gathered_outputs = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_outputs, outputs)
    if rank == 0:
        outputs = list(itertools.chain.from_iterable(gathered_outputs))
    dist.barrier()

if rank == 0:
    metrics = action_matching_evaluation(outputs, metric='macro')
    print(metrics)
    write_json(outputs, args.save_path)
    metrics_path = os.path.splitext(args.save_path)[0] + "_metrics.json"
    write_json(metrics, metrics_path)
else:
    metrics = None
