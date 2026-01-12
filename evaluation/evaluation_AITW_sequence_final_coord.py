import json
import os
import re
import ast
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import action_matching

DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

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
parser.add_argument('--his_num', type=int, default=1,
                    help='The number of history actions to include')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for processing')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# device = 'cuda'
# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + args.model_path.split('/')[-1] + '.json'


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


set_seed(args.seed)

# device = 'cuda'

import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, 
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="cuda"
)


model.gradient_checkpointing_enable()


device = next(model.parameters()).device

min_pixels = 256*28*28
max_pixels = 12800*28*28

processor = AutoProcessor.from_pretrained(args.model_path)

torch.cuda.empty_cache()

if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled (may slow down inference)")
else:
    print("Warning: Gradient checkpointing not available for this model")


def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=1280 * 28 * 28):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
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

    #image_input, _ = process_vision_info_with_resize(messages)
    image_input, _ = process_vision_info(messages)
    return image_input[0]


# Image cache for batch processing
image_cache = {}

def get_image_info_cached(image_path, min_pixel=min_pixels, max_pixel=max_pixels):
    """Cached version of get_image_info to avoid reloading images"""
    if image_path not in image_cache:
        # Get the processed image
        processed_image = get_image_info(image_path, min_pixel, max_pixel)
        
        # Get original image dimensions
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

    input_ids = inputs.input_ids
    
    # Delete inputs to free memory immediately
    del inputs
    torch.cuda.empty_cache()

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    
    # Delete intermediate tensors
    del input_ids, generated_ids
    torch.cuda.empty_cache()

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    del generated_ids_trimmed
    torch.cuda.empty_cache()

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



from tqdm import tqdm
import action_matching 


def convert_aitw_to_pred_format(aitw_dict, image_path=None, image_resize=None):
    """
    Convert action dict from AITW format (action_2_format) to predictions.json format.
    Input: {"action_type_text": "click", "touch": [0.151, 0.107], "lift": [0.151, 0.107]}  (normalized 0-1)
    Output: {"action": "click", "coordinate": [163, 205]}  (pixel coordinates)
    
    For prediction format, coordinates should be in pixels based on resized image dimensions.
    Handles touch and lift coordinates by denormalizing from 0-1 range to pixel coordinates.
    Uses action_type_text to determine action type (always present in AITW dataset).
    """
    if aitw_dict is None:
        return None
    
    # Get image dimensions for denormalization (prioritize image_resize)
    # img_width, img_height = get_image_size(image_path, image_resize)
    
    img_width = image_resize.get("width", "")
    img_height = image_resize.get("height", "")
    
    # Use action_type_text to determine action type (always present in AITW dataset)
    action_type_text = aitw_dict.get("action_type_text", "").lower().strip()
    
    # Helper function to denormalize coordinates
    def denormalize_coord(norm_coord):
        """Convert normalized [0-1] coordinates to pixel coordinates."""
        if len(norm_coord) < 2 or norm_coord[0] < 0 or norm_coord[1] < 0:
            return [0, 0]
        return [int(norm_coord[0] * img_width), int(norm_coord[1] * img_height)]
    
    # Determine action based on action_type_text
    touch = aitw_dict.get("touch")
    lift = aitw_dict.get("lift")
    
    # Click action (touch and lift are the same for clicks)
    if action_type_text == "click":
        touch = aitw_dict.get("touch")
        return {
            "action": "click",
            "coordinate": denormalize_coord(touch)
        }
    
    # All scroll actions (down, up, left, right) use the same swipe format
    if "scroll" in action_type_text:
    
        if "down" in action_type_text:
            return {
                "action": "swipe",
                "coordinate": [int(0.5*img_width), int(0.8*img_height)],
                "coordinate2": [int(0.5*img_width), int(0.2*img_height)]
            }  
        elif "up" in action_type_text:
            return {
                "action": "swipe",
                "coordinate": [int(0.5*img_width), int(0.2*img_height)],
                "coordinate2": [int(0.5*img_width), int(0.8*img_height)]
            } 
        elif "left" in action_type_text:
            return {
                "action": "swipe",
                "coordinate": [int(0.2*img_width), int(0.5*img_height)],
                "coordinate2": [int(0.8*img_width), int(0.5*img_height)]
            } 
        elif "right" in action_type_text:
            return {
                "action": "swipe",
                "coordinate": [int(0.2*img_width), int(0.5*img_height)],
                "coordinate2": [int(0.8*img_width), int(0.5*img_height)]
            } 
    
    # Type action
    elif action_type_text == "type" or "type" in action_type_text:
        typed_text = aitw_dict.get("typed_text") or aitw_dict.get("type_text", "")
        return {
            "action": "type",
            "text": typed_text
        }
    
    # Press back
    elif "back" in action_type_text or action_type_text == "press_back":
        return {
            "action": "system_button",
            "button": "back"
        }
    
    # Press home
    elif "home" in action_type_text or action_type_text == "press_home":
        return {
            "action": "system_button",
            "button": "home"
        }
    
    # Press enter
    elif "enter" in action_type_text or action_type_text == "press_enter":
        return {
            "action": "system_button",
            "button": "enter"
        }
    
    # Task complete / terminate
    elif "complete" in action_type_text:
        return {"action": "terminate", "status": "success"}
        
    elif "impossible" in action_type_text: 
        return {"action": "terminate", "status": "failure"}
    
    # Unknown action type
    return None


def convert_predict_to_aitw_format(pred_dict, image_path=None, image_resize=None):
    """
    Convert action dict from predictions.json format to AITW format (action_2_format).
    Input: {"action": "click", "coordinate": [163, 205]}  (pixel coordinates)
    Output: {"action_type_id": 4, "action_type_text": "click", "touch": [0.151, 0.107], "lift": [0.151, 0.107]}  (normalized 0-1)
    
    For AITW format, coordinates should be normalized (0-1) based on resized image dimensions.
    This is the reverse of convert_aitw_to_pred_format.
    """
    if pred_dict is None:
        return None
    
    # Get image dimensions for normalization (prioritize image_resize)
    # img_width, img_height = get_image_size(image_path, image_resize)
    img_width = image_resize.get("width", "")
    img_height = image_resize.get("height", "")

    # Helper function to normalize coordinates
    def normalize_coord(pixel_coord):
        """Convert pixel coordinates to normalized [0-1] coordinates."""
        if len(pixel_coord) < 2:
            return [0.0, 0.0]
        return [round(pixel_coord[0] / img_width, 3), round(pixel_coord[1] / img_height, 3)]
    
    # Common invalid coordinate tuple
    INVALID_COORD = [-1.0, -1.0]
    
    action = pred_dict.get("action", "")
    
    if action == "click":
        coord = pred_dict.get("coordinate", [0, 0])
        return {
            "action_type": 4,
            "action_type_text": "click",
            "click_point": normalize_coord(coord)
        }
    
    elif action == "swipe":
        
        coord = pred_dict.get("coordinate", [0, 0])
        coord2 = pred_dict.get("coordinate2", [0, 0])
        
        ## Normalize coordinates
        touch = normalize_coord(coord)
        lift = normalize_coord(coord2)
   
        # Determine scroll direction based on dominant axis
        dx = coord2[0] - coord[0]
        dy = coord2[1] - coord[1]
        
        if abs(dy) > abs(dx):
            if dy < 0:
                action_type_text = "scroll down"
                action_type = 0
            else:
                action_type_text = "scroll up"
                action_type = 1
        else:
            if dx > 0:
                action_type_text = "scroll left"
                action_type = 8
            else:
                action_type_text = "scroll right"
                action_type = 9

        return {
            "action_type": action_type,
            "action_type_text": action_type_text,
            # "touch": touch,
            # "lift": lift
        }
        
    elif action == "type":
        text = pred_dict.get("text", "")
        return {
            "action_type": 3,
            "action_type_text": "type",
            "touch": [-1.0, -1.0],
            "lift": [-1.0, -1.0],
            "typed_text": text
        }
    elif action == "system_button":
        button = pred_dict.get("button", "")
        if button.lower() == "enter":
            return {
                "action_type": 7,
                "action_type_text": "press_enter",
                "touch": [-1.0, -1.0],
                "lift": [-1.0, -1.0]
            }
        elif button.lower() == "back":
            return {
                "action_type": 5,
                "action_type_text": "press_back",
                "touch": [-1.0, -1.0],
                "lift": [-1.0, -1.0]
            }
        elif button.lower() == "home":
            return {
                "action_type": 6,
                "action_type_text": "press_home",
                "touch": [-1.0, -1.0],
                "lift": [-1.0, -1.0]
            }
    elif action == "terminate":
        if pred_dict.get("status", "success") == "success":
            action_type = 10
            action_type_text = "status_task_complete"
        else:
            action_type = 11
            action_type_text = "status_task_impossible"
        return {
            "action_type": action_type,
            "action_type_text": action_type_text,
            "touch": [-1.0, -1.0],
            "lift": [-1.0, -1.0]
        }
    else:
        return None


def parse_action_string(action_str):
    """
    Parse action string from predictions.json format.
    Input: '<action>{"action": "click", "coordinate": [163, 205]}</action>'
    Output: {"action": "click", "coordinate": [163, 205]}
    """
    if not action_str or action_str.strip() == "":
        return None
    
    # Extract content between <action>...</action>
    match = re.search(r'<action>(.*?)</action>', action_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content == "null":
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try with ast.literal_eval as fallback
            try:
                return ast.literal_eval(content)
            except:
                return None
    else:
        # Try parsing as direct JSON
        try:
            return json.loads(action_str)
        except:
            return None


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

def get_image_size(image_path=None, image_resize=None):
    """
    Get image dimensions (width, height).    
    Args:
        image_path: Path to image file (optional)
        image_resize: Dict with 'resized_width' and 'resized_height' (optional)
    
    Returns:
        (width, height) tuple - uses resized dimensions if available
    """
    # Use image_resize attribute if available
    if image_resize and isinstance(image_resize, dict):
        resized_w = image_resize.get('width')
        resized_h = image_resize.get('height')
        if resized_w is not None and resized_h is not None:
            return (resized_w, resized_h)
    
    # Fallback to loading from image_path
    if image_path:
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception:
            pass
    
    # Default to common mobile screen size if can't load
    return (None, None)


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

aitw_imgs_dir = "/cpfs01/HithinkOmniHDD/user_workspace/songyurun/github/AITW/aitw_images"
aitw_test = json.load(open('/cpfs01/HithinkOmniHDD/user_workspace/songyurun/github/AITW/aitw_annots/aitw_data_test.json', 'r'))
# aitw_test = json.load(open('/home/noah/zhouxurui/GUIAgent-Data/aitw_seeclick/aitw_data_train.json', 'r'))

# prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
user_response = """User Instruction: {} .\nOutput Format: ```\n<action> ... </action>\n```"""
# user_response_think = """User Instruction: {} .\nOutput Format: ```\n<think> ... </think>\n\n<action> ... </action>\n```"""
additional = 'If the query asks a question, please answer the question through the answer action before terminating the process.\n'


score_average = 0
all_save_results = []
all_eval_results = []

# Batch processing variables
BATCH_SIZE = args.batch_size
batch_prompts = []
batch_image_paths = []
batch_steps = []
resize_image_info = {}  # Accumulate precomputed image info to avoid duplicate calls

for task, episodes in aitw_test.items():
    print("Task: " + task)

    # if task == 'general':
    #     continue

    corr_action = 0
    corr_type = 0
    num_text = 0
    corr_text = 0
    num_scroll = 0
    corr_scroll = 0
    num_click = 0
    corr_click = 0
    num_both_click = 0
    corr_both_click = 0
    num_wrong_format = 0
    num = 0
    
    print("sample num:", len(episodes))

    for j, episode in tqdm(enumerate(episodes)):
        previous_actions = []
        previous_imgs = []

        for step in episode:
            step_json = {'task': task, 'episode': step['ep_id'], 'correct': 'no'}

            # Handle image path - check if it's a filename or full path
            img_filename = step.get("screenshot", step.get("img_filename", ""))
            if not img_filename:
                print('image filename not found')
                continue
                
            # Construct full path if needed
            if not os.path.isabs(img_filename):
                img_path = os.path.join(aitw_imgs_dir, img_filename)
                if not img_path.endswith('.png'):
                    img_path += '.png'
            else:
                img_path = img_filename

            if not os.path.exists(img_path):
                print(f'image not found: {img_path}')
                continue

            goal = step["goal"]
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
                    
            # Build and store the current step
            # action_step = action2step(step)

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

            action_step = convert_aitw_to_pred_format(step, None, image_ele)
            
            previous_actions.append("<action>" + json.dumps(action_step, ensure_ascii=False) + "</action>")

            previous_imgs.append(img_path)
    
            conversations.extend(conversations_his)
            conversations[-1]["value"] += "<image>\n"            
            cur_all_imgs.append(img_path)
            # Add to batch
            batch_prompts.append(conversations)
            batch_image_paths.append(cur_all_imgs)
            batch_steps.append((step, step_json))


            # Process batch when full or at last item
            is_last_episode = (j == len(episodes) - 1)
            is_last_step = (step == episode[-1])
        

            if len(batch_prompts) == BATCH_SIZE or (is_last_episode and is_last_step):
                # Process batch - pass precomputed image info to avoid duplicate processing
                responses, batch_image_info = batch_generate_grounding(batch_prompts, batch_image_paths, resize_image_info)
                # Clear precomputed info after processing batch
                resize_image_info = {}

                # Clear image cache periodically to prevent memory buildup
                # Keep cache size reasonable (e.g., max 1000 images)
                if len(image_cache) > 50:
                    # Clear half of the cache (oldest entries)
                    keys_to_remove = list(image_cache.keys())[:len(image_cache)//2]
                    for key in keys_to_remove:
                        del image_cache[key]
                    torch.cuda.empty_cache()
                    

                # Process each response in batch
                for idx, resp in enumerate(responses):
                    
                    step_info, step_json = batch_steps[idx]
                    
                    # Parse response from predictions.json format
                    action_dict = parse_action_string(resp)
                    
                    action_pred = None
                    
                    action_ref = action_matching.action_2_format(step_info)

                    num += 1
                    

                    #############################################################################
                    ## AITW Evaluation from SeeClick, SimpAgent
                    ## https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/aitw_test.py
                    ## https://github.com/JiuTian-VL/SimpAgent/blob/main/AITW_eval.py
                    #############################################################################

                    try:
                        # Get image info for coordinate conversion (use last image - current screenshot)
                        image_paths = batch_image_paths[idx]
                        image_info_list = batch_image_info[idx]
                        current_image_path = image_paths[-1] if image_paths else None
                        current_image_info = image_info_list[-1] if image_info_list else None
                        # Convert from predictions.json format to pred_2_format input format
                                                
                        pred_dict = convert_predict_to_aitw_format(action_dict, None, current_image_info)
                        # print(f"pred_dict: {action_dict}")
    
                        if pred_dict is None:
                            num_wrong_format += 1
                            continue
                        
                        # Convert to final format for action matching
                        action_pred = action_matching.pred_2_format(pred_dict)
                        
                        
                        annot_position = np.array(
                            [step_info["annot_position"][i:i + 4] for i in range(0, len(step_info["annot_position"]), 4)])
                        check_match = action_matching.check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                  action_pred["action_type"], action_ref["touch_point"],
                                                                  action_ref["lift_point"], action_ref["action_type"],
                                                                  annot_position)
                        # step accuracy
                        if check_match == True:
                            corr_action += 1
                            match_label = 1
                            step_json['correct'] = 'yes'
                        else:
                            match_label = 0
    
                        # type accuracy
                        if action_pred["action_type"] == action_ref["action_type"]:
                            corr_type += 1
    
                        # text accuracy
                        if action_ref["action_type"] == 3:
                            num_text += 1
                            if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                                    action_pred["typed_text"] in action_ref["typed_text"]) or (
                                    action_ref["typed_text"] in action_pred["typed_text"]):
                                corr_text += 1
    
                        if action_ref["action_type"] == 4:
                            # click accuracy
                            if action_matching.is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                                num_click += 1
                                if match_label:
                                    corr_click += 1
                                    
                            # scroll accuracy
                            else:
                                num_scroll += 1
                                if match_label:
                                    corr_scroll += 1
     
                            if (action_pred["action_type"] == 4) and action_matching.is_tap_action(action_ref["touch_point"],
                                                                                                   action_ref[
                                                                                                       "lift_point"]) and action_matching.is_tap_action(
                                    action_pred["touch_point"], action_pred["lift_point"]):
                                num_both_click += 1
                                if match_label:
                                    corr_both_click += 1

                    except:
                        num_wrong_format += 1
                        print("Step: " + str(j) + " wrong format")

                    step_json['goal'] = goal
                    step_json['image_size'] = current_image_info
                    step_json["raw_resp"] = action_dict
                    step_json['action_pred'] = action_pred
                    step_json['action_ref'] = action_ref
                    step_json['annot'] = annot_position.tolist()
                    
                    all_save_results.append(step_json)
                
                # Reset batch
                batch_prompts = []
                batch_image_paths = []
                batch_steps = []


    # print("batch prompts num: " + len(batch_prompts))
    
    score_average += corr_action / num
    
    if num_scroll == 0: num_scroll = 1

    print("Action Acc: " + str(corr_action / num))
    print("Type Acc: " + str(corr_type / num))
    print("Text Acc: " + str(corr_text / num_text))
    print("Click Acc: " + str(corr_click / num_click))
    print("Scroll Acc: " + str(corr_scroll / num_scroll))
    print("Both Click Acc: " + str(corr_both_click / num_both_click))
    print("Num Both Click: " + str(num_both_click))
    print("Num wrong format: " + str(num_wrong_format))

print("Average score: " + str(score_average / 5))

write_json(all_save_results, args.save_path)
