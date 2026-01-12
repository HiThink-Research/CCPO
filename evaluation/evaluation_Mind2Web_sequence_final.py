import json
import os
import re
import copy
from collections import defaultdict

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
from PIL import Image

SYSTEM_MESSAGE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format

```
<action> ... </action>
```

## Action Space

You can perform the following actions:
- click: Click the point on the screen with specified (x, y) coordinates.
- type: Input the specified text into the activated input box at the specified coordinates.
- select: Select an option from a dropdown or list at the specified coordinates.
- hover: Hover over the point on the screen with specified (x, y) coordinates.
- enter: Press enter at the specified coordinates.

The arguments you can use are:
- coordinate: (x, y): The x and y pixels coordinates from the left and top edges.
- value: Text input required by actions like `type` or `select`.

Format your output as a JSON object with the selected action and its arguments at the same level.

Example outputs:
<action>
{"action": "click", "coordinate": <value>}
</action>
<action>
{"action": "type", "coordinate": <value>, "value": <value>}
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
parser.add_argument('--model_path', type=str, default="/data6/GUIModels/model/aitw_4AI_run1_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--input_path', type=str, default='/cpfs01/HithinkOmniSSD/user_workspace/songyurun/gui-agent-compression/SimpAgent/data/mind2web/annot',
                    help='The path where the test data is loaded from')
parser.add_argument('--imgs_dir', type=str,  default='/cpfs01/HithinkOmniSSD/user_workspace/songyurun/gui-agent-compression/SimpAgent/data/mind2web/mind2web_images',
                    help='The directory where Mind2Web images are stored')
parser.add_argument('--task', type=str, default=None,
                    help='Task name for data file (e.g., shopping, travel)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for processing')
parser.add_argument('--his_num', type=int, default=4,
                    help='Number of previous actions/images to include in history')
parser.add_argument('--use_multi_gpu', action="store_true",
                    help='Use multi-GPU processing')
device = 'auto'
# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + args.model_path.split('/')[-1] + '.json'

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
    device = f'cuda:{local_rank}'
    print(f"Initialized distributed training: local_rank={local_rank}, world_size={world_size}")
else:
    device = 'cuda:0'
    print("Using single GPU")

print("Loading model...")
if args.use_multi_gpu:
    # Multi-GPU loading with device_map="auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
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

# Simple memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Available GPUs: {torch.cuda.device_count()}")


# OS-Atlas
# max_pixels= 1048576
# min_pixels= 3136

# UI-S1
min_pixels = 256*28*28
max_pixels = 12800*28*28 


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


# Convert Mind2Web action to step format (for model input/output)
def action2step_mind2web(action, image_size, return_bbox=False):
    """Convert Mind2Web action format to step format"""
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    # Calculate center point from bbox (in pixel coordinates)
    bbox = action.get("bbox", {})
    point_x = bbox.get("x", 0) + (bbox.get("width", 0) / 2)
    point_y = bbox.get("y", 0) + (bbox.get("height", 0) / 2)
    coordinate = [int(point_x), int(point_y)]

    # Build action content
    action_content = {
        "action": action_type.lower(),
        "coordinate": coordinate
    }
    
    # Add value for TYPE and SELECT actions
    if action_type in ['TYPE', 'SELECT']:
        value = action["operation"].get("value", "")
        if value:
            if action_type == 'TYPE':
                action_content["text"] = value
            elif action_type == 'SELECT':
                action_content["value"] = value

    if return_bbox:
        # Return normalized bbox for evaluation (0-1 range)
        bbox_norm = [
            bbox.get("x", 0) / image_size[0],
            bbox.get("y", 0) / image_size[1],
            (bbox.get("x", 0) + bbox.get("width", 0)) / image_size[0],
            (bbox.get("y", 0) + bbox.get("height", 0)) / image_size[1]
        ]
        bbox_norm = [round(item, 3) for item in bbox_norm]
        return action_content, bbox_norm
    else:
        return action_content


# Convert model output to Mind2Web evaluation format (dict format for evaluation)
def action2step_eval_format(action_content, image_size):
    """Convert action content to Mind2Web evaluation format (returns dict with action_type, click_point as tuple)"""
    action_type_map = {
        'click': 4,
        'select': 2,
        'type': 3,
        'hover': 4,
        'enter': 4
    }
    
    action_type = action_content.get('action', '').lower()
    action_type_num = action_type_map.get(action_type, 4)
    
    # Get coordinate (should be in pixel coordinates)
    if 'coordinate' in action_content:
        coord = action_content['coordinate']
        # Normalize to 0-1 range
        click_point = [coord[0] / image_size[0], coord[1] / image_size[1]]
        click_point = [round(item, 3) for item in click_point]
    else:
        click_point = [0.0, 0.0]
    
    # Build evaluation format dict
    eval_dict = {
        "action_type": action_type_num,
        "click_point": tuple(click_point)
    }
    
    if action_type_num in [2, 3]:  # SELECT or TYPE
        value = action_content.get('value', action_content.get('text', ''))
        eval_dict["value"] = value
    
    return eval_dict


# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def mind2web_evaluation(outputs):
    """Mind2Web evaluation - matches mind2web_test.py evaluation logic"""
    results = []
    
    for sample in outputs:
        pred = sample['pred']
        gt = sample['gt']
        image_size = sample.get('image_size', (1280, 720))  # Default size
        
        # Parse prediction
        try:
            pred_tags = parse_tags(pred, ['action'])
            pred_action_content = pred_tags.get('action', '')
            if pred_action_content:
                pred_action = json.loads(pred_action_content)
            else:
                pred_action = {}
        except:
            pred_action = {}
        
        # Parse ground truth
        try:
            gt_tags = parse_tags(gt, ['action'])
            gt_action_content = gt_tags.get('action', '')
            if gt_action_content:
                gt_action = json.loads(gt_action_content)
            else:
                gt_action = {}
        except:
            gt_action = {}
        
        # Get reference action and bbox for evaluation
        ref_action = sample.get('ref_action', {})
        bbox_ref = sample.get('bbox_ref', [0, 0, 1, 1])
        
        # Convert to evaluation format
        try:
            pred_eval = action2step_eval_format(pred_action, image_size)
            ref_eval = action2step_eval_format(gt_action, image_size)
        except Exception as e:
            logger.info(f"Error converting to eval format: {e}")
            pred_eval = {"action_type": 0, "click_point": (0.0, 0.0)}
            ref_eval = {"action_type": 0, "click_point": (0.0, 0.0)}
        
        step_result = {
            "annot_id": sample.get('annot_id', ''),
            "img_path": sample.get('image', ''),
            "instruction": sample.get('instruction', ''),
            "sentence": pred,
            "Op_match": False,
            "Ele_match": False,
            "Op_F1": [0, ref_eval.get("action_type", 0)]
        }
        
        try:
            # Check operation match
            if pred_eval.get("action_type") == ref_eval.get("action_type"):
                step_result["Op_match"] = True
            
            # Check element match (click point within bbox)
            click_point = pred_eval.get("click_point", (0.0, 0.0))
            # print(click_point)
            # print(bbox_ref)
            # Ensure click_point is a list/tuple of numbers
            if isinstance(click_point, (list, tuple)) and len(click_point) == 2:
                click_point = [float(click_point[0]), float(click_point[1])]
            else:
                click_point = [0.0, 0.0]
            
            if len(click_point) == 2 and len(bbox_ref) == 4:
                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True
            
            # Calculate F1 score for action
            pred_str = str(pred_eval.get("action_type", 0))
            if pred_eval.get("action_type") in [2, 3]:  # SELECT or TYPE
                pred_str += ' '
                pred_str += pred_eval.get("value", "").lower()
            
            ref_str = str(ref_eval.get("action_type", 0))
            if ref_eval.get("action_type") in [2, 3]:  # SELECT or TYPE
                ref_str += ' '
                ref_str += ref_eval.get("value", "").lower()
            
            op_f1 = calculate_f1(pred_str, ref_str)
            step_result["Op_F1"][0] = op_f1
            
        except Exception as e:
            logger.info(f"Evaluation error: {e}")
        
        results.append(step_result)
    
    return results


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

    # image_input, _ = process_vision_info_with_resize(messages)    
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

# Load test data
if args.task:
    input_file = os.path.join(args.input_path, f'mind2web_data_test_{args.task}.json')
else:
    input_file = args.input_path

with open(input_file, 'r') as f:
    mind2web_test = json.load(f)

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
    return image_cache[image_path]


def batch_generate_grounding(batch_prompts, batch_image_paths, precomputed_image_info=None):
    """Process a batch of examples simultaneously
    
    Args:
        batch_prompts: List of conversation prompts
        batch_image_paths: List of lists of image paths
        precomputed_image_info: Optional dict mapping image paths to pre-computed image info
    """
    if precomputed_image_info is None:
        precomputed_image_info = {}
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
        
        texts.append(text)

        images_for_example = []
        example_image_info = []  # Store image info for this example
        for img_path in image_paths:
            # Use precomputed info if available, otherwise get from cache
            if img_path in precomputed_image_info:
                img_info = precomputed_image_info[img_path]
            else:
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
        do_sample=True,
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



user_response = """User Instruction: {} .\nOutput Format: ```\n<action> ... </action>\n```"""

additional = 'If the query asks a question, please answer the question through the answer action before terminating the process.\n'

step_i = 0
batch_prompts = []
batch_image_paths = []
batch_steps = []
outputs = []
resize_image_info = {}  # Accumulate precomputed image info to avoid duplicate calls

BATCH_SIZE = args.batch_size

for j, episode in enumerate(tqdm(mind2web_test)):
    
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []
    previous_imgs = []

    for step_idx, step in enumerate(episode["actions"]):
        if "bbox" not in step:
            print("action not found")
            continue

        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(args.imgs_dir, filename)
        if not os.path.exists(img_path):
            print('image not found')
            continue

        # Get image size
        image = Image.open(img_path)
        image_size = image.size

        prompt = user_response.format(goal)
        
        conversations = []

        conversations_his = [{"value": prompt + "\n" + additional + "\n", "from": "human"}]
        cur_step_preimg = previous_imgs[-args.his_num:]
        cur_step_idx = len(previous_imgs[-args.his_num:])
        cur_all_imgs = []

        for i, action in enumerate(previous_actions[-args.his_num:]):
            conversations_his[-1]["value"] += "<image>\n"
            conversations_his.append({"value": action, "from": "assistant"})
            conversations_his.append({"value": "Output Format: ```\n<action> ... </action>\n", "from": "human"})
            cur_all_imgs.append(previous_imgs[-args.his_num:][i])
        
        # Get image info once and store it for reuse in batch_generate_grounding()
        img_info = get_image_info_cached(img_path, min_pixels, max_pixels)
        # Store in precomputed dict to avoid duplicate call in batch_generate_grounding()
        resize_image_info[img_path] = img_info

        image_ele = {
            'width': img_info['width'],
            'height': img_info['height'],
            'resized_width': img_info['resized_width'],
            'resized_height': img_info['resized_height']
        }

        # Convert action to step format and normalize coordinates for history
        action_content, bbox_ref = action2step_mind2web(step, image_size, return_bbox=True)
        resize_action = convert_originl_resize_coordinate(action_content, image_ele)
    
        action_step = '<action>' + json.dumps(resize_action, ensure_ascii=False) + '</action>'
        previous_actions.append(action_step)
        previous_imgs.append(img_path)
        
        # Extend conversations and add the current image for the new human turn
        conversations.extend(conversations_his)
        conversations[-1]["value"] += "<image>\n"
        cur_all_imgs.append(img_path)
        
        # Add to batch
        batch_prompts.append(conversations)
        batch_image_paths.append(cur_all_imgs)

        batch_steps.append({
            'step': step,
            'image_size': image_size,
            'bbox_ref': bbox_ref,
            'annot_id': annot_id,
            'goal': goal
        })
        step_i += 1


        # Process batch when full or at last item
        is_last_step = (j == len(mind2web_test) - 1) and (step_idx == len(episode["actions"]) - 1)
        if len(batch_prompts) == BATCH_SIZE or is_last_step:
            # Process batch - pass precomputed image info to avoid duplicate processing
            responses, batch_image_info = batch_generate_grounding(batch_prompts, batch_image_paths, resize_image_info)
            # Clear precomputed info after processing batch
            resize_image_info = {}
            
            # Process each response in batch
            for idx, resp in enumerate(responses):
                step_info = batch_steps[idx]

                # Use stored image dimensions for coordinate conversion
                image_ele = batch_image_info[idx][-1]  # Get the last image info (current screenshot)

                # Apply coordinate conversion to predicted action
                try:
                    pred_tags = parse_tags(resp, ['action'])
                    pred_action_content = pred_tags.get('action', '')
                    if pred_action_content:
                        pred_action = json.loads(pred_action_content)
                        # Convert coordinates from resized to original image coordinates
                        pred_action = convert_resize_original_coordinate(pred_action, image_ele)
                    else:
                        pred_action = {}
                except:
                    print("prediction error")
                    pred_action = {}

                # print(pred_action)
                # Get reference action
                ref_action, bbox_ref = action2step_mind2web(step_info['step'], step_info['image_size'], return_bbox=True)
                ref_action_str = '<action>' + json.dumps(ref_action, ensure_ascii=False) + '</action>'
                
                outputs.append({
                    'question': batch_prompts[idx],
                    'image': step_info['step'].get('action_uid', ''),
                    'pred': '<action>' + json.dumps(pred_action, ensure_ascii=False) + '</action>',
                    'gt': ref_action_str,
                    'image_size': step_info['image_size'],
                    'ref_action': ref_action,
                    'bbox_ref': bbox_ref,
                    'annot_id': step_info['annot_id'],
                    'instruction': step_info['goal'],
                })

            # Reset batch
            batch_prompts = []
            batch_image_paths = []
            batch_steps = []


print(f"Evaluating ...")
print(len(outputs))


#############################################################################
## Mind2Web Evaluation from SeeClick, SimpAgent
## https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/aitw_test.py
## https://github.com/JiuTian-VL/SimpAgent/blob/main/AITW_eval.py
#############################################################################

print(f"Evaluating Mind2Web format outputs...")
eval_results = mind2web_evaluation(outputs)

# Calculate metrics
num_step = 0
num_episode = 0
num_op = 0
num_ele = 0
op_f1 = {4: [], 2: [], 3: []}
macro_ele_acc = {}
macro_step_acc = {}
macro_action_f1 = {}
num_step_success = 0
num_episode_success = 0

# Group results by episode
episode_results = defaultdict(list)
for result in eval_results:
    annot_id = result['annot_id']
    episode_results[annot_id].append(result)

for i, (annot_id, item) in enumerate(episode_results.items()):
    macro_ele_acc[i] = []
    macro_step_acc[i] = []
    macro_action_f1[i] = []
    num_episode += 1
    episode_success = True
    for step_result in item:
        num_step += 1

        if step_result["Op_match"]:
            num_op += 1

        if step_result["Ele_match"]:
            num_ele += 1
            macro_ele_acc[i].append(1)
        else:
            macro_ele_acc[i].append(0)

        if step_result["Op_F1"][1] in op_f1:
            op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
        macro_action_f1[i].append(step_result["Op_F1"][0])

        if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
            num_step_success += 1
            macro_step_acc[i].append(1)
        else:
            macro_step_acc[i].append(0)
            episode_success = False

    if episode_success:
        num_episode_success += 1

marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()]) if op_f1 else 0

print("Operation F1: " + str(marco_op_f1))
print("Element Acc: " + str(num_ele / num_step) if num_step > 0 else "N/A")
print("Step Success: " + str(num_step_success / num_step) if num_step > 0 else "N/A")
print("Episode Success: " + str(num_episode_success / num_episode) if num_episode > 0 else "N/A")
print("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()]) if macro_ele_acc else 0
macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()]) if macro_step_acc else 0
macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()]) if macro_action_f1 else 0
print("Macro Ele Acc: " + str(macro_ele_acc))
print("Macro Op F1: " + str(macro_action_f1))
print("Macro Step SR: " + str(macro_step_acc))

# Save results
write_json(outputs, args.save_path)
print(f"Results saved to {args.save_path}")