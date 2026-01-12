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
- key: Perform a key event on the mobile device using adb's `keyevent` syntax.
- click: Click the point on the screen with specified (x, y) coordinates.
- long_press: Press the point on the screen with specified (x, y) coordinates for a specified number of seconds.
- swipe: Swipe from starting point with specified (x, y) coordinates to endpoint with specified (x2, y2) coordinates.
- type: Input the specified text into the activated input box.
- answer: Output the specified answer.
- system_button: Press the specified system button: Back, Home, Menu, or Enter.
- open: Open an application on the device specified by text.
- wait: Wait for a specified number of seconds for changes to occur.
- terminate: Terminate the current task and report its completion status: success or failure.

The arguments you can use are:
- coordinate: (x, y): The x and y pixels coordinates from the left and top edges.
- coordinate2: (x, y): The x and y pixels coordinates from the left and top edges for the endpoint of a swipe.
- text: Text input required by actions like `key`, `type`, `answer`, and `open`.
- time: The time in seconds required by actions like `long_press` and `wait`.
- button: System buttons available for pressing: Back, Home, Menu, or Enter. Possible values: Back, Home, Menu, Enter.
- status: The completion status of a terminated task. Possible values: success, failure.

Format your output as a JSON object with the selected action and its arguments at the same level.

Example outputs:
<action>
{"action": "key", "text": "<value>"}
</action>
<action>
{"action": "click", "coordinate": "<value>"}
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

parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="eval_results/",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data6/GUIModels/model/aitw_4AI_run1_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--input_path', type=str, default='/cpfs01/HithinkOmniSSD/user_workspace/songyurun/gui-agent-compression/UI-S1/datasets/android_control_evaluation_fixed_local.jsonl',
                    help='The path where the model is loaded from')
parser.add_argument('--batch_size', type=int, default=1,
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=4,
                    help='The path where the model is loaded from')
parser.add_argument('--use_multi_gpu', action="store_true",
                    help='The path where the model is loaded from')
device = 'auto'

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


def resize_coordinate(coordinate, source_size, target_size):
    x, y = coordinate
    target_width,  target_height = target_size
    source_width, source_height = source_size
    width_ratio = target_width / source_width
    height_ratio =  target_height / source_height

    # Convert Coordinates 
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


def predict_direction(start, end):
    x1, y1 = start
    x2, y2 = end
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return 'right'
        else:
            return 'left'
    else:
        if delta_y > 0:
            return 'down'
        else:
            return 'up'

#############################################################################
## Android Control Evaluation from SeeClick, SimpAgent
## https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/aitw_test.py
## https://github.com/JiuTian-VL/SimpAgent/blob/main/AITW_eval.py
#############################################################################


def action_matching_evaluation(outputs):
    """New evaluation method for Android Control - matches evaluation_unify_AC.py exactly"""
    step_acc_res_dict = defaultdict(int)
    sample_number_dict = defaultdict(int)

    for sample in outputs:
        pred = sample['pred']
        gt = sample['gt']

        # Parse prediction action using parse_tags to extract only <action> content
        try:
            pred_tags = parse_tags(pred, ['action'])
            pred_action_content = pred_tags.get('action', '')
            if pred_action_content:
                pred_action = json.loads(pred_action_content)
            else:
                pred_action = {}
        except:
            pred_action = {}

        # Parse ground truth action using parse_tags to extract only <action> content
        try:
            gt_tags = parse_tags(gt, ['action'])
            gt_action_content = gt_tags.get('action', '')
            if gt_action_content:
                gt_action = json.loads(gt_action_content)
            else:
                gt_action = {}
        except:
            gt_action = {}

        gt_action_type = gt_action['action'] if 'action' in gt_action else ""
        pred_action_type = pred_action['action'] if 'action' in pred_action else ""
        
        sample_number_dict["full"] += 1
        sample_number_dict[gt_action_type] += 1


        # Calculate step accuracy based on types - match original AC logic exactly
        if gt_action_type == pred_action_type:
            
            step_acc_res_dict["type_match"] += 1
            step_acc_res_dict[gt_action_type + "_type_match"] += 1
            
            if gt_action_type in ["click", "long_press"]:  # evaluate click type
                step_acc_res_dict["click_match"] += 1
                try:
                    pred_x, pred_y = pred_action['coordinate'][0], pred_action['coordinate'][1]
                except:
                    pred_x, pred_y = -100, -100
                gt_x, gt_y = gt_action['coordinate'][0], gt_action['coordinate'][1]

                if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <= 0.14 * 1000:  # set 14 % of screen size as the ratio
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type + "_all_match"] += 1

            elif gt_action_type == "open":
                if gt_action == pred_action or ('text' in pred_action and calculate_f1_score(pred_action['text'], gt_action['text'])) > 0.5:
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type + "_all_match"] += 1

            elif gt_action_type == "type":
                if pred_action == gt_action or ('text' in pred_action and calculate_f1_score(pred_action['text'], gt_action['text'])) > 0.5:
                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type + "_all_match"] += 1
                    
            elif gt_action_type == "swipe":   # Important compare swipe direction, due to coordinate normalization  
                if "coordinate" in pred_action and "coordinate2" in pred_action:
                    pred_dir = predict_direction(pred_action["coordinate"], pred_action["coordinate2"])
                    gt_dir = predict_direction(gt_action["coordinate"], gt_action["coordinate2"])
                    if pred_dir == gt_dir:
                        step_acc_res_dict["full"] += 1
                        step_acc_res_dict[gt_action_type + "_all_match"] += 1
                    
            elif gt_action == pred_action:  # evaluate other types
                step_acc_res_dict["full"] += 1
                step_acc_res_dict[gt_action_type + "_all_match"] += 1


    # Print the low-level results - match original format exactly
    logger.info("="*30 + " AC Step Acc " + "="*30)
    logger.info("Acc: %f" % (step_acc_res_dict["full"] / sample_number_dict["full"]))
    logger.info(f"type_match acc: %f" % (step_acc_res_dict["type_match"] / sample_number_dict["full"]))
    logger.info(f"grounding acc: %f" % (step_acc_res_dict["click_all_match"] / step_acc_res_dict["click_type_match"]))

    return {
        'step_acc': step_acc_res_dict["full"] / sample_number_dict["full"],
        'type_match_acc': step_acc_res_dict["type_match"] / sample_number_dict["full"],
        'grounding_acc': step_acc_res_dict["click_all_match"] / step_acc_res_dict["click_type_match"] if step_acc_res_dict["click_type_match"] > 0 else 0,
        'total_actions': sample_number_dict["full"],
        'step_acc_res_dict': dict(step_acc_res_dict),
        'sample_number_dict': dict(sample_number_dict)
    }



def action2step(step_data):
    action_type = step_data["action"]

    if action_type == "click":
        return {"action": action_type, "coordinate": [int(i) for i in step_data['coordinate']]}

    elif action_type == "long_press":
        return {"action": action_type, "coordinate": [int(i) for i in step_data['coordinate']], "time": step_data['time']}

    elif action_type == "swipe":
        return {"action": action_type, "coordinate": [int(i) for i in step_data['coordinate']], "coordinate2": [int(i) for i in step_data['coordinate2']]}

    elif action_type == "type":
        return {"action": action_type, "text": step_data['text']}

    elif action_type == "system_button":
        return {"action": action_type, "button": step_data['button']}

    elif action_type == "open":
        return {"action": action_type, "text": step_data['text']}

    elif action_type == "wait":
        return {"action": action_type, "time": step_data['time']}



from PIL import Image
def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=1280 * 28 * 28):
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

# Load test data - match ac_process_999_sequence_llavaformat.py format
with open(args.input_path, 'r') as f:
    ac_test = [json.loads(line) for line in f.readlines()]

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
        # messages = [{"role": "system", "content": SYSTEM_MESSAGE_THINK}] 
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
# user_response_think = """User Instruction: {} .\nOutput Format: ```\n<think> ... </think>\n\n<action> ... </action>\n```"""

additional = 'If the query asks a question, please answer the question through the answer action before terminating the process.\n'

step_i = 0
batch_prompts = []
batch_image_paths = []
batch_steps = []
outputs = []
resize_image_info = {} 

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

        conversations_his= [{"value": prompt + "\n"+ additional +"\n", "from": "human"}]
        cur_step_preimg = previous_imgs[-args.his_num:]
        cur_step_idx = len(previous_imgs[-args.his_num:])
        cur_all_imgs = []

        for i, action in enumerate(previous_actions[-args.his_num:]):
            conversations_his[-1]["value"] += "<image>\n"
            conversations_his.append({"value": action, "from": "assistant"})
            conversations_his.append({"value":"Output Format: ```\n<action> ... </action>\n", "from": "human"})
            cur_all_imgs.append(previous_imgs[-args.his_num:][i])
        
        #  History coordinates need to normalize

        ########################
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
        ########################

        resize_action = convert_originl_resize_coordinate(action2step(step['action_content']), image_ele)
    
        action_step = '<action>' + json.dumps(resize_action) + '</action>'
        previous_actions.append(action_step)
        previous_imgs.append(img_path)
        
        # Extend conversations and add the current image for the new human turn
        conversations.extend(conversations_his)
        conversations[-1]["value"] += "<image>\n"
        cur_all_imgs.append(img_path)
        
        # Add to batch
        batch_prompts.append(conversations)
        batch_image_paths.append(cur_all_imgs)

        batch_steps.append(step)
        step_i += 1

        
        # # Process batch when full or at last item
        if len(batch_prompts) == BATCH_SIZE or j == len(ac_test) - 1:
            # Process batch - pass precomputed image info to avoid duplicate processing
            responses, batch_image_info = batch_generate_grounding(batch_prompts, batch_image_paths, resize_image_info)
            # Clear precomputed info after processing batch
            resize_image_info = {}
            
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
                        # Convert coordinates from resized to original image coordinates
                        pred_action = convert_resize_original_coordinate(pred_action, image_ele)
                    else:
                        pred_action = {}
                except:
                    pred_action = {}

                ref_action = action2step(step_info['action_content'])
                
                outputs.append({
                    'question': batch_prompts[idx],
                    'image': step_info["screenshot"],
                    'pred': '<action>' + json.dumps(pred_action, ensure_ascii=False) + '</action>',
                    'gt': '<action>' + json.dumps(ref_action, ensure_ascii=False) + '</action>',
                    'more_info': {
                        'category': step_info.get('category', 'unknown'),
                        'episode_id': step_info.get('episode_id', 0),
                        'check_options': step_info.get('check_options', None)
                    },
                })

            # Reset batch
            batch_prompts = []
            batch_image_paths = []
            batch_steps = []



def convert_outputs_to_ac_scales(outputs, os_atlas_min_pixels=3136, os_atlas_max_pixels=1048576):
    """
    Convert all outputs to AC format based on OS-ATLAS max_pixels.
    This converts both pred_action and ref_action coordinates from original to OS-ATLAS resized format.
    """
    converted_outputs = []
    
    for output in outputs:
        image_path = output['image']
        
        # Get OS-ATLAS image info
        original_img = Image.open(image_path)
        processed_image = get_image_info(image_path, os_atlas_min_pixels, os_atlas_max_pixels)
        original_width, original_height = original_img.size
        resized_width, resized_height = processed_image.size
        
        AC_image_ele = {
            'width': original_width,
            'height': original_height,
            'resized_width': resized_width,
            'resized_height': resized_height
        }
        
        # Parse and convert pred_action
        try:
            pred_tags = parse_tags(output['pred'], ['action'])
            pred_action_content = pred_tags.get('action', '')
            if pred_action_content:
                pred_action = json.loads(pred_action_content)
                pred_action = convert_originl_resize_coordinate(pred_action, AC_image_ele)
            else:
                pred_action = {}
        except:
            pred_action = {}
        
        # Parse and convert ref_action
        try:
            gt_tags = parse_tags(output['gt'], ['action'])
            gt_action_content = gt_tags.get('action', '')
            if gt_action_content:
                ref_action = json.loads(gt_action_content)
                ref_action = convert_originl_resize_coordinate(ref_action, AC_image_ele)
            else:
                ref_action = {}
        except:
            ref_action = {}
        
        # Create converted output
        converted_output = {
            'question': output['question'],
            'image': output['image'],
            'pred': '<action>' + json.dumps(pred_action, ensure_ascii=False) + '</action>',
            'gt': '<action>' + json.dumps(ref_action, ensure_ascii=False) + '</action>',
            'more_info': output.get('more_info', {})
        }
        
        converted_outputs.append(converted_output)
    
    return converted_outputs


print(f"Evaluating ...")
print(len(outputs))

print(f"Evaluating Full Resolution AC format outputs...")
metrics = action_matching_evaluation(outputs, metric='macro')
print(metrics)
write_json(outputs, args.save_path)

# Convert outputs to AC format based on OS-ATLAS max_pixels for a fair comparsion
print(f"Converting {len(outputs)} outputs to AC Scales (OS-ATLAS max_pixels=1048576)...")
ac_outputs = convert_outputs_to_ac_scales(outputs, os_atlas_min_pixels=3136, os_atlas_max_pixels=1048576)

# Evaluation on AC format outputs
print(f"Evaluating AC format outputs...")
metrics = action_matching_evaluation(ac_outputs, metric='macro')
print(metrics)


