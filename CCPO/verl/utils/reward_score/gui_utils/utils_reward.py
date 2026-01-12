import base64
import copy
import json
import re
import time
import traceback

import numpy as np
import requests
# from x.data.agent.fake_uitars import PlainCallFormat
# from x.data.agent.space.std_space import RAW_SPACE
# from x.data.text import parse_tags
import torch

BBOX_ENLARGE_FACTOR = 1.2
POINT_DISTANCE_THRESHOLD = 0.1


def _safe_dim(value):
    try:
        value = float(value)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return None


def _normalize_point(point, width, height):
    if width is None or height is None:
        return [-1, -1]
    if isinstance(point, list) and len(point) == 2:
        x, y = point
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return [x / width, y / height]
    return [-1, -1]


def _normalize_candidate_bboxes(candidate_bbox, width, height):
    if width is None or height is None:
        return []
    normalized = []
    if isinstance(candidate_bbox, list):
        items = candidate_bbox
        if len(candidate_bbox) == 4 and all(isinstance(v, (int, float)) for v in candidate_bbox):
            items = [candidate_bbox]
        for bbox in items:
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                x1, y1, x2, y2 = bbox
                normalized.append([x1 / width, y1 / height, x2 / width, y2 / height])
    return normalized


def norm_coordinate(action, width, height):
    width = _safe_dim(width)
    height = _safe_dim(height)
    if 'candidate_bbox' in action:
        normalized_bboxes = _normalize_candidate_bboxes(action['candidate_bbox'], width, height)
        if normalized_bboxes:
            action['candidate_bbox'] = normalized_bboxes
        else:
            action.pop('candidate_bbox', None)
    if 'coordinate' in action:
        action['coordinate'] = _normalize_point(action['coordinate'], width, height)
    if 'coordinate2' in action:
        action['coordinate2'] = _normalize_point(action['coordinate2'], width, height)
    return action


def check_text(text_pred, text_gt, text_retrict=False):
    text_pred = text_pred.lower().strip()
    text_gt = text_gt.lower().strip()
    if text_retrict:
        return text_pred == text_gt
    return (text_pred in text_gt) or (text_gt in text_pred)

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
    if not (isinstance(click, list) and len(click) == 2):
        return False
    if click == [-1, -1]:
        return False
    # if candidate_bbox:
    #     candidate_bbox = enlarge_bbox(candidate_bbox, scale_factor=BBOX_ENLARGE_FACTOR)
    #     for bbox in candidate_bbox:
    #         if (bbox[0] <= click[0] <= bbox[2]) and (bbox[1] <= click[1] <= bbox[3]):
    #             return True
    if gt_point is not None and isinstance(gt_point, list) and len(gt_point) == 2:
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
def check_response_match(pred_action, current_check_pam, width, height, resized_width, resized_height, text_retrict=False):
    pred_action = norm_coordinate(copy.deepcopy(pred_action), resized_width, resized_height) # todo use resized width
    current_check_pam = norm_coordinate(copy.deepcopy(current_check_pam), width, height)
    # print(current_check_pam, pred_action)
    ratio = 0.5
    if current_check_pam['action'] in ['wait', 'terminate']:
        if pred_action['action'] == current_check_pam['action']:
            return True, True, ratio
        return False, False, ratio
    elif current_check_pam['action'] == 'system_button':
        if pred_action['action'] == 'system_button':
            if 'button' not in pred_action:
                return True, False, ratio
            if not isinstance(pred_action['button'], str):
                return True, False, ratio
            
            return True, current_check_pam['button'].lower().strip() == pred_action['button'].lower().strip(), ratio
        else:
            return False, False, ratio
    elif current_check_pam['action'] in ['type', 'answer', 'key']:
        if pred_action['action'] == 'type':
            if 'text' not in pred_action:
                return True, False, ratio
            return True, check_text(pred_action['text'], current_check_pam['text'], text_retrict=text_retrict), ratio
        else:
            return False, False, ratio
    elif current_check_pam['action'] == 'open':
        if pred_action['action'] == 'open':
            if 'text' not in pred_action:
                return True, False, ratio
            return True, check_text(pred_action['text'], current_check_pam['text'], text_retrict=text_retrict), ratio 
        else:
            return False, False, ratio
    elif current_check_pam['action'] == 'swipe':
        if pred_action['action'] == 'swipe':
            # if 'direction' in current_check_pam:
            #     gt_direction = current_check_pam['direction']
            # else:
            if 'coordinate' not in pred_action or "coordinate2" not in pred_action:
                return False, False, ratio
                
            if pred_action['coordinate'] == [-1, -1] or pred_action['coordinate2'] == [-1, -1]:
                return False, False, ratio
                
            gt_direction = predict_direction(current_check_pam['coordinate'], current_check_pam['coordinate2'])
            direction = predict_direction(pred_action['coordinate'], pred_action['coordinate2'])
            # if gt_direction == 'down':
            #     gt_direction = 'up'
            # elif gt_direction == 'up':
            #     gt_direction = 'down'
            return True, direction == gt_direction, ratio
        else:
            return False, False, ratio
    elif current_check_pam['action'] in ['long_press', 'click']:
        if pred_action['action'] == current_check_pam['action']:
            return evaluate_click_action(pred_action, current_check_pam)
        else:
            return False, False, ratio
    return False, False, ratio




# Distance thresholds for normalized coordinates (0-1 range)
MIN_WEIGHT = 0.5
MAX_TOLERANCE_DISTANCE_THRESHOLD = 0.5  # Beyond half of the screen width or height, give 0 credit
TOLERANCE_DISTANCE_THRESHOLD = 0.1      # Within this gives full credit


def _distance_for_penalty(click, candidate_bbox, gt_point):
    
    """
    Calculate distance for penalty calculation.
    
    Returns minimum distance to gt_point (if exists) or to candidate bbox boundaries.
    If both exist, returns the minimum among all options.
    Returns None if nothing to compare against.
    
    Args:
        click: Click coordinates [x, y] in normalized space [0-1]
        candidate_bbox: List of bounding boxes [[x1, y1, x2, y2], ...] in normalized space
        gt_point: Ground truth point [x, y] in normalized space, or None
    
    Returns:
        float: Distance in normalized coordinates, or None if nothing to compare
    """

    click = np.array(click)
    distances = []
    
    # Calculate distance to gt_point if it exists
    if gt_point is not None:
        distances.append(float(np.linalg.norm(np.array(gt_point) - click)))
    
    # Calculate distance to bbox boundaries if they exist
    if candidate_bbox:
        bbox_array = np.array(candidate_bbox)
        for bbox in bbox_array:
            x1, y1, x2, y2 = bbox
            # Calculate distance to bbox boundary
            # If point is inside bbox, distance is 0
            # Otherwise, distance is to the nearest edge/corner
            dx = max(0, max(x1 - click[0], click[0] - x2))
            dy = max(0, max(y1 - click[1], click[1] - y2))
            bbox_distance = np.sqrt(dx**2 + dy**2)
            distances.append(float(bbox_distance))
    
    if distances:
        return float(np.min(distances))
    
    return None


def _linear_weight_from_distance(distance, tolerance_threshold, max_tolerance_threshold):
    """
    Compute weight based on distance using linear interpolation.
    
    - If distance <= tolerance_threshold: return 1.0 (full credit)
    - If distance > max_tolerance_threshold: return MIN_WEIGHT (minimum penalty)
    - Otherwise: linear interpolation from 1.0 (at tolerance_threshold) to MIN_WEIGHT (at max_tolerance_threshold)
    
    Args:
        distance: Distance in normalized coordinates (0-1 range)
        tolerance_threshold: Distance threshold for full credit
        max_tolerance_threshold: Maximum distance before minimum penalty
    
    Returns:
        float: Weight between MIN_WEIGHT and 1.0
    """

    if distance <= tolerance_threshold:
        return 1.0

    if distance > max_tolerance_threshold:
        return MIN_WEIGHT
    
    # Linear interpolation: 1.0 at tolerance_threshold, MIN_WEIGHT at max_tolerance_threshold
    penalty_range = max_tolerance_threshold - tolerance_threshold
    if penalty_range <= 0:
        return MIN_WEIGHT
    
    # Interpolate from 1.0 to MIN_WEIGHT
    weight = 1.0 - (distance - tolerance_threshold) / penalty_range * (1.0 - MIN_WEIGHT)
    
    return float(np.clip(weight, MIN_WEIGHT, 1.0))



def evaluate_click_action(pred_action, current_check_pam,
                          tolerance_threshold=None, max_tolerance_threshold=None):
    """
    Evaluate click action with distance-based penalty.
    
    Returns (action_match: bool, click_ok: bool, weight: float)
    
    - If click_ok is True -> weight = 1.0
    - If click_ok is False -> weight based on distance:
      * If distance <= tolerance_threshold: weight = 1.0 (full credit)
      * If distance > max_tolerance_threshold: weight = 0.0 (full penalty)
      * Otherwise: linear penalty between threshold and max_tolerance_threshold
    - Distance is computed to gt_point if available, else to nearest bbox center.
    - All coordinates are in normalized space (0-1 range).
    
    Args:
        pred_action: Dict with keys 'action' and 'coordinate' (x,y) in normalized coordinates [0-1]
        current_check_pam: Dict with keys:
            - 'action' (expected: 'click' or 'long_press')
            - 'coordinate' (gt_point) [optional] in normalized coordinates [0-1]
            - 'candidate_bbox' (list of [x1,y1,x2,y2]) [optional] in normalized coordinates [0-1]
        tolerance_threshold: Distance threshold in normalized coordinates [0-1] for full credit
            (default: TOLERANCE_DISTANCE_THRESHOLD)
        max_tolerance_threshold: Maximum distance in normalized coordinates [0-1] before full penalty
            (default: MAX_TOLERANCE_DISTANCE_THRESHOLD)
    
    Returns:
        tuple: (action_match: bool, click_ok: bool, weight: float)
    """
    # expected = current_check_pam.get('action')
    # if expected not in ('click', 'long_press'):
    #     return False, False, 0.0
    
    # action_match = (pred_action.get('action') == expected)
    # if not action_match:
    #     return False, False, 0.0

    if 'coordinate' not in pred_action:
        return True, False, MIN_WEIGHT
        
    click = pred_action['coordinate']
    
    if click == [-1, -1]:
        return True, False, MIN_WEIGHT
        
    candidate_bbox = current_check_pam.get('candidate_bbox', [])
    gt_point = current_check_pam.get('coordinate')
    
    # Exact correctness check
    click_ok = check_click(click, candidate_bbox, gt_point)
    if click_ok:
        return True, True, 1.0
    
    # Compute distance for penalty calculation
    distance = _distance_for_penalty(click, candidate_bbox, gt_point)

    if distance is None:
        return True, False, MIN_WEIGHT
    
    # Apply distance-based penalty
    
    tolerance_threshold = tolerance_threshold or TOLERANCE_DISTANCE_THRESHOLD
    max_tolerance_threshold = max_tolerance_threshold or MAX_TOLERANCE_DISTANCE_THRESHOLD
    
    weight = _linear_weight_from_distance(distance, tolerance_threshold, max_tolerance_threshold)
    
    # print(pred_action)
    # print(current_check_pam)
    # print(weight)
    # weight = 0.5
    return True, False, weight