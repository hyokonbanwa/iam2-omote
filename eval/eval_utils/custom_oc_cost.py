# from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import ot
#from mmdet.evaluation.functional import bbox_overlaps
# from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
# from .evaluations import similariry_score

@dataclass
class DetectedInstance:
    label: int
    x1: float
    y1: float
    x2: float
    y2: float


def bbox_gious(
    bboxes1: npt.ArrayLike,
    bboxes2: npt.ArrayLike,
    eps: float = 1e-6,
    use_legacy_coordinate: bool = False,
) -> npt.ArrayLike:
    """Calculate the generalized ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4) # [[x1, y1, x2, y2], ...]
        bboxes2 (ndarray): Shape (k, 4) # [[x1, y1, x2, y2], ...]
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        gious (ndarray): Shape (n, k)
    """

    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    gious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return gious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        gious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length
    )
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length
    )
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0
        )

        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        ious = overlap / union

        # Finding the coordinate of smallest enclosing box
        x_min = np.minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_min = np.minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_max = np.maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_max = np.maximum(bboxes1[i, 3], bboxes2[:, 3])
        hull = (x_max - x_min + extra_length) * (y_max - y_min + extra_length)

        gious[i, :] = ious - (hull - union) / hull

    if exchange:
        gious = gious.T

    return gious


# def add_label(result: Sequence[Sequence]) -> npt.ArrayLike:
#     labels = [[i] * len(r) for i, r in enumerate(result)]
#     labels = np.hstack(labels)
#     return np.hstack([np.vstack(result), labels[:, None]])


# def split_result(result: list[DetectedInstance]) -> tuple[npt.ArrayLike,list]:
#     bboxes = []
#     labels = []
#     for r in result:
#         bboxes.append([r.x1, r.y1, r.x2, r.y2])
#         labels.append(r.label)
        
#     bboxes = np.asarray(bboxes, dtype=np.float32)
    
#     return (bboxes, labels)



def cost_func(i_j_loc_cost,i_j_cls_cost,mode: str = "giou", alpha: float = 0.8):
    """Calculate a unit cost

    Args:
        x (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        y (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        mode (str, optional): Type of IoUs. Defaults to "giou" (Generalized IoU).
        alpha (float, optional): weights to balance localization and classification errors. Defaults to 0.8.

    Returns:
        float: a unit cost
    """

    return alpha * i_j_loc_cost + (1 - alpha) * i_j_cls_cost


def get_cmap(
    a_result: list[DetectedInstance],
    b_result: list[DetectedInstance],
    alpha: float = 0.8,
    beta: float = 0.4,
    mode="giou",
    label_or_sim: str = "label",
    similarity_model:SentenceTransformer=None,
    similarity_func:Callable=None
) -> Tuple[npt.ArrayLike]:
    """Calculate cost matrix

    Args:
        a_result ([type]): detections
        b_result ([type]): detections
        mode (str, optional): [description]. Defaults to "giou".

    Returns:
        dist_a (np.array): (N+1,) array. distribution over detections.
        dist_b (np.array): (M+1,) array. distribution over detections.
        cost_map:
    """
    # a_result = split_result(a_result)
    # b_result = split_result(b_result)
    
    n = len(a_result)
    m = len(b_result)
    
    a_label_list = []
    a_bboxes = []
    for a in a_result:
        a_label_list.append(a.label)
        a_bboxes.append([a.x1,a.y1,a.x2,a.y2])
    a_bboxes = np.asarray(a_bboxes, dtype=np.float32)
    
    b_label_list = []
    b_bboxes = []
    for b in b_result:
        b_label_list.append(b.label)
        b_bboxes.append([b.x1,b.y1,b.x2,b.y2])
    b_bboxes = np.asarray(b_bboxes, dtype=np.float32)

    giou_map = bbox_gious(a_bboxes, b_bboxes)  # range [-1, 1]
    loc_cost_map = 1 - (giou_map + 1) * 0.5

    cls_cost_map = np.zeros((n, m))
    # print(a_label_list)
    # print(b_label_list)
    if label_or_sim == "label":
        for i in range(n):
            for j in range(m):
                if a_label_list[i] == b_label_list[j]:
                    cls_cost_map[i, j] = 0
                else:
                    cls_cost_map[i, j] = 1
    elif label_or_sim == "sim":
        if n == 0 or m == 0:
            pass
        else:
            score = similarity_func(a_label_list,b_label_list,similarity_model) # 0 ~ 1
            #スコア丸め込み
            score[score > 1.0] = 1.0
            score[score < 0.0] = 0.0
            cls_cost_map = 1 - score # 1 - (0 ~ 1)
        
    

    cost_map = np.zeros((n + 1, m + 1))
    for i in range(n):
        for j in range(m):
            # cost = cost_func(a_result[i], b_result[j], alpha=alpha, mode=mode,label_or_sim=label_or_sim,similarity_model=similarity_model,similariry_score=similarity_score)
            # # print(cost)
            cost_map[i, j] = cost_func(loc_cost_map[i, j],cls_cost_map[i, j], alpha=alpha)
    #metric = lambda x, y: cost_func(x, y, alpha=alpha, mode=mode,label_or_sim=label_or_sim,similarity_model=similarity_model,similariry_score=similarity_score)
    # cost_map[:n, :m] = cdist(a_result, b_result, metric)
    #print(cost_map)
    dist_a = np.ones(n + 1)
    dist_b = np.ones(m + 1)

    # cost for dummy demander / supplier
    cost_map[-1, :] = beta
    cost_map[:, -1] = beta
    dist_a[-1] = m
    dist_b[-1] = n

    return dist_a, dist_b, cost_map


def postprocess(M: npt.ArrayLike, P: npt.ArrayLike) -> float:
    """drop dummy to dummy costs, normalize the transportation plan, and return total cost

    Args:
        M (npt.ArrayLike): correction cost matrix コスト行列
        P (npt.ArrayLike)): optimal transportation plan matrix 輸送計画

    Returns:
        float: _description_
    """
    P[-1, -1] = 0 #ダミーからダミーの輸送は無視する
    P /= P.sum() #輸送計画の正規化
    total_cost = (M * P).sum() #最終的なコストの計算
    return total_cost


def get_ot_cost(
    a_detection: list[DetectedInstance],
    b_detection: list[DetectedInstance],
    costmap_func: Callable,
    return_matrix: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """[summary]

    Args:
        a_detection (list): list of detection results. a_detection[i] contains bounding boxes for i-th class.
        Each element is numpy array whose shape is N x 5. [[x1, y1, x2, y2, label], ...]
        b_detection (list): ditto
        costmap_func (callable): a function that takes a_detection and b_detection as input and returns a unit cost matrix
    Returns:
        [float]: optimal transportation cost
    """

    if len(a_detection) == 0:
        if len(b_detection) == 0:
            return 0

    a, b, M = costmap_func(a_detection, b_detection)
    P = ot.emd(a, b, M)
    total_cost = postprocess(M, P)

    if return_matrix:
        log = {"M": M, "a": a, "b": b}
        return total_cost, log
    else:
        return total_cost
