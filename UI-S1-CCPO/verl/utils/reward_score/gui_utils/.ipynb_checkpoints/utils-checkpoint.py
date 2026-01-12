import base64
import copy
import json
import re
import time
import traceback
import math

import numpy as np
import requests
# from x.data.agent.fake_uitars import PlainCallFormat
# from x.data.agent.space.std_space import RAW_SPACE
# from x.data.text import parse_tags
import torch

BBOX_ENLARGE_FACTOR = 1.2
POINT_DISTANCE_THRESHOLD = 0.04
CLICK_DISTANCE_THRESHOLD_PIXELS = 0.12 * 1000  # 120 pixels


def norm_coordinate(action, width, height):
    if 'candidate_bbox' in action and len(action['candidate_bbox']) == 4: # fix bug
        x, y, w, h = action['candidate_bbox']
        action['candidate_bbox'] = [[x / width, y / height, w / width, h / height]]
    if 'coordinate' in action:
        action['coordinate'] = [action['coordinate'][0]/width, action['coordinate'][1]/height]
    if 'coordinate2' in action:
        action['coordinate2'] = [action['coordinate2'][0]/width, action['coordinate2'][1]/height]
    return action


def check_text(text_pred, text_gt, text_retrict=False):
    text_pred = text_pred.lower().strip()
    text_gt = text_gt.lower().strip()
    if text_retrict:
        return text_pred == text_gt
    return (text_pred in text_gt) or (text_gt in text_pred)

def calculate_f1_score(predicted_str, ground_truth_str):
    """
    Calculate F1 score between two strings based on token overlap.
    Matches the implementation in new_action_matching_evaluation.
    """
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
    
    # 计算坐标变化
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    # 判断方向
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
        
def check_click(click, candidate_bbox, gt_point):
    if len(candidate_bbox):
        candidate_bbox = enlarge_bbox(candidate_bbox, scale_factor=BBOX_ENLARGE_FACTOR)
        for bbox in candidate_bbox:
            if (bbox[0] <= click[0] <= bbox[2]) and (bbox[1] <= click[1] <= bbox[3]):
                return True
    if gt_point is not None:
        return np.linalg.norm([gt_point[0]-click[0], gt_point[1]-click[1]]) <= POINT_DISTANCE_THRESHOLD
    return False

def enlarge_bbox(bbox_list, scale_factor=1.2)->np.ndarray:
    """
    将每个 bounding box 放大一定倍数。

    :param bbox_list: bounding box 列表, 每个 bbox 是一个包含四个值的元组或列表, 表示 (xmin, ymin, xmax, ymax)
    :param scale_factor: 放大倍数
    :return: 放大后的 bounding box 列表
    """
    bbox_array = np.array(bbox_list)
    try:
        x_min, y_min, x_max, y_max = \
            bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    except:
        print(bbox_array)
        raise
    
    # 计算每个 bounding box 的中心点
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # 计算每个 bounding box 的宽度和高度
    width = (x_max - x_min) * scale_factor
    height = (y_max - y_min) * scale_factor
    
    # 计算放大后的 bounding box 的新的坐标
    new_x_min = x_center - width / 2
    new_y_min = y_center - height / 2
    new_x_max = x_center + width / 2
    new_y_max = y_center + height / 2
    
    # 将新的坐标组合成 bounding box 列表
    enlarged_bbox_list = np.vstack((new_x_min, new_y_min, new_x_max, new_y_max)).T
    
    return enlarged_bbox_list


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
        

def check_response_match(pred_action, current_check_pam, width, height, resized_width, resized_height, text_retrict=False):
    """
    Check if predicted action matches ground truth action.
    Updated to match new_action_matching_evaluation logic:
    - Click/long_press: Uses raw pixel distance <= 140 pixels (0.14 * 1000)
    - Type/open: Uses exact match OR F1 score > 0.5
    - Swipe: Uses direction matching
    - Others: Uses exact match
    """
    # Keep raw coordinates before normalization for distance calculations
    pred_action_raw = copy.deepcopy(pred_action)
    current_check_pam_raw = copy.deepcopy(action2step(current_check_pam))
    
    # Normalize coordinates for other operations (if needed)
    pred_action = norm_coordinate(copy.deepcopy(pred_action), resized_width, resized_height)
    current_check_pam = norm_coordinate(copy.deepcopy(current_check_pam), width, height)
    
    # Check action type match first (using raw actions)
    gt_action_type = current_check_pam_raw.get('action', '')
    pred_action_type = pred_action_raw.get('action', '')
    
    if gt_action_type != pred_action_type:
        return False, False
    
    # Action types match, now check full match based on action type
    if gt_action_type in ['click', 'long_press']:
        # Use raw pixel coordinates for distance calculation (matching new_action_matching_evaluation)
        try:
            pred_x = pred_action_raw.get('coordinate', [0, 0])[0]
            pred_y = pred_action_raw.get('coordinate', [0, 0])[1]
        except (KeyError, IndexError):
            pred_x, pred_y = -100, -100
        
        try:
            gt_x = current_check_pam_raw.get('coordinate', [0, 0])[0]
            gt_y = current_check_pam_raw.get('coordinate', [0, 0])[1]
        except (KeyError, IndexError):
            gt_x, gt_y = 0, 0
        
        # Calculate distance in pixels (matching new_action_matching_evaluation line 392)
        distance = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        full_match = distance <= CLICK_DISTANCE_THRESHOLD_PIXELS
        return True, full_match
    
    elif gt_action_type == 'swipe':
        # Check direction match using raw coordinates
        if 'coordinate' in pred_action_raw and 'coordinate2' in pred_action_raw and \
           'coordinate' in current_check_pam_raw and 'coordinate2' in current_check_pam_raw:
            pred_dir = predict_direction(pred_action_raw['coordinate'], pred_action_raw['coordinate2'])
            gt_dir = predict_direction(current_check_pam_raw['coordinate'], current_check_pam_raw['coordinate2'])
            return True, pred_dir == gt_dir
        else:
            return True, False
    
    elif gt_action_type in ['type', 'answer', 'key']:
        # Use exact match OR F1 score > 0.5 (matching new_action_matching_evaluation line 402)
        # Compare raw actions (not normalized) for exact match
        if pred_action_raw == current_check_pam_raw:
            return True, True
        elif 'text' in pred_action_raw and 'text' in current_check_pam_raw:
            f1 = calculate_f1_score(pred_action_raw['text'], current_check_pam_raw['text'])
            return True, f1 > 0.5
        else:
            return True, False
    
    elif gt_action_type == 'open':
        # Use exact match OR F1 score > 0.5 (matching new_action_matching_evaluation line 397)
        # Compare raw actions (not normalized) for exact match
        if pred_action_raw == current_check_pam_raw:
            return True, True
        elif 'text' in pred_action_raw and 'text' in current_check_pam_raw:
            f1 = calculate_f1_score(pred_action_raw['text'], current_check_pam_raw['text'])
            return True, f1 > 0.5
        else:
            return True, False
    else:
        # For other action types, use exact match (matching new_action_matching_evaluation line 414)
        # Compare raw actions (not normalized) for exact match
        return True, pred_action_raw == current_check_pam_raw