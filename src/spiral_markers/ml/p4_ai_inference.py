from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from scipy.optimize import linear_sum_assignment

from spiral_markers.ml.p4_blur_reasoner import P4BlurReasoner, build_spiral_node_features, pad_node_batch, vec_to_angle_deg
from spiral_markers.tag_layout import decode_robot_id_bits


def _angle_from_center_to_point_deg(center_xy: tuple[float, float], point_xy: tuple[float, float]) -> float:
    dx = float(point_xy[0]) - float(center_xy[0])
    dy = float(center_xy[1]) - float(point_xy[1])
    return float(np.rad2deg(np.arctan2(dy, dx)))


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)


def _select_node_detections(det_rows: list[dict[str, Any]], max_nodes: int) -> list[dict[str, Any]]:
    return sorted(det_rows, key=lambda item: float(item.get("confidence", 1.0)), reverse=True)[: int(max_nodes)]


def _mean_center_xy(rows: list[dict[str, Any]]) -> tuple[float, float]:
    centers = np.asarray([row["center_xy"] for row in rows], dtype=np.float32)
    return float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1]))


def _distance_xy(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    return float(math.hypot(float(a_xy[0]) - float(b_xy[0]), float(a_xy[1]) - float(b_xy[1])))


def _dedupe_detections(
    det_rows: list[dict[str, Any]],
    merge_radius_px: float = 28.0,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(det_rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        center_xy = tuple(float(v) for v in row["center_xy"])
        if any(_distance_xy(center_xy, tuple(float(v) for v in prev["center_xy"])) < float(merge_radius_px) for prev in kept):
            continue
        kept.append(dict(row))
    return kept


def _score_subset_geometry(
    subset_rows: list[dict[str, Any]],
    expected_radius_px: float = 80.0,
) -> float:
    center_xy = _mean_center_xy(subset_rows)
    radii = np.asarray([_distance_xy(center_xy, tuple(float(v) for v in row["center_xy"])) for row in subset_rows], dtype=np.float32)
    mean_radius = float(np.mean(radii))
    radius_std = float(np.std(radii))
    angles = sorted([(_angle_from_center_to_point_deg(center_xy, tuple(float(v) for v in row["center_xy"]))) % 360.0 for row in subset_rows])
    diffs = [float((angles[(idx + 1) % len(angles)] - angles[idx]) % 360.0) for idx in range(len(angles))]
    angle_error = float(np.mean([abs(diff - 90.0) for diff in diffs]))
    zero_count = int(sum(int(row["class_id"]) == 1 for row in subset_rows))
    conf_sum = float(sum(float(row.get("confidence", 0.0)) for row in subset_rows))
    zero_penalty = 0.0 if zero_count == 1 else (12.0 if zero_count == 0 else 20.0)
    radius_penalty = abs(mean_radius - float(expected_radius_px)) + 2.5 * radius_std
    score = conf_sum - 0.035 * angle_error - 0.040 * radius_penalty - zero_penalty
    return float(score)


def _cluster_fixed_groups_of_four(
    det_rows: list[dict[str, Any]],
    num_clusters: int = 4,
    cluster_size: int = 4,
) -> list[list[dict[str, Any]]]:
    if int(num_clusters) <= 0:
        return []
    required = int(num_clusters) * int(cluster_size)
    if len(det_rows) < required:
        return []
    candidate_rows = sorted(det_rows, key=lambda row: float(row.get("confidence", 0.0)), reverse=True)[:required]
    centers = np.asarray([row["center_xy"] for row in candidate_rows], dtype=np.float32)
    model = KMeansConstrained(
        n_clusters=int(num_clusters),
        size_min=int(cluster_size),
        size_max=int(cluster_size),
        random_state=0,
        n_init=8,
    )
    labels = model.fit_predict(centers)
    groups: list[list[dict[str, Any]]] = []
    for cluster_index in range(int(num_clusters)):
        cluster_rows = [dict(candidate_rows[row_idx]) for row_idx, label in enumerate(labels.tolist()) if int(label) == int(cluster_index)]
        if len(cluster_rows) == int(cluster_size):
            groups.append(cluster_rows)
    return groups


def _decode_id_from_group(
    center_xy: tuple[float, float],
    zero_row: dict[str, Any],
    identity_rows: list[dict[str, Any]],
) -> tuple[int | None, dict[int, dict[str, Any]]]:
    heading_deg = _angle_from_center_to_point_deg(center_xy, tuple(float(v) for v in zero_row["center_xy"]))
    targets = {
        1: float(heading_deg - 90.0),
        2: float(heading_deg - 180.0),
        3: float(heading_deg - 270.0),
    }
    cost = np.zeros((len(identity_rows), 3), dtype=np.float32)
    for row_idx, row in enumerate(identity_rows):
        angle_deg = _angle_from_center_to_point_deg(center_xy, tuple(float(v) for v in row["center_xy"]))
        for col_idx, slot_index in enumerate((1, 2, 3)):
            cost[row_idx, col_idx] = float(_angle_delta_deg(angle_deg, targets[slot_index]))
    row_ind, col_ind = linear_sum_assignment(cost)
    selected_slots: dict[int, dict[str, Any]] = {0: dict(zero_row)}
    bits_by_slot: dict[int, int] = {}
    for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
        slot_index = (1, 2, 3)[col_idx]
        row = dict(identity_rows[row_idx])
        if float(cost[row_idx, col_idx]) > 50.0:
            continue
        selected_slots[int(slot_index)] = row
        if int(row["class_id"]) == 0:
            bits_by_slot[int(slot_index)] = 0
        elif int(row["class_id"]) == 2:
            bits_by_slot[int(slot_index)] = 1
    if len(bits_by_slot) != 3:
        return None, selected_slots
    robot_id = decode_robot_id_bits([bits_by_slot[1], bits_by_slot[2], bits_by_slot[3]])
    return int(robot_id), selected_slots


def _build_classical_groups(
    det_rows: list[dict[str, Any]],
    expected_group_radius_px: float = 80.0,
    num_clusters: int = 4,
    dedupe_radius_px: float = 28.0,
) -> list[dict[str, Any]]:
    deduped_rows = _dedupe_detections(det_rows, merge_radius_px=float(dedupe_radius_px))
    groups: list[dict[str, Any]] = []
    cluster_groups = _cluster_fixed_groups_of_four(deduped_rows, num_clusters=int(num_clusters))
    for subset_rows in cluster_groups:
        center_xy = _mean_center_xy(subset_rows)
        zero_rows = [dict(row) for row in subset_rows if int(row["class_id"]) == 1]
        zero_row = None
        if zero_rows:
            zero_row = max(zero_rows, key=lambda row: float(row.get("confidence", 0.0)))
        identity_rows = [dict(row) for row in subset_rows if row is not zero_row]
        robot_id = None
        selected_slots: dict[int, dict[str, Any]] = {}
        heading_deg = None
        heading_source = "unavailable"
        zero_center_xy = None
        if zero_row is not None:
            heading_deg = _angle_from_center_to_point_deg(center_xy, tuple(float(v) for v in zero_row["center_xy"]))
            zero_center_xy = tuple(float(v) for v in zero_row["center_xy"])
            heading_source = "center_to_zero_slot"
            decoded_robot_id, slot_rows = _decode_id_from_group(center_xy, zero_row, identity_rows)
            robot_id = decoded_robot_id
            selected_slots = slot_rows
        group_score = _score_subset_geometry(subset_rows, expected_radius_px=expected_group_radius_px)
        groups.append(
            {
                "center_xy": center_xy,
                "heading_deg": heading_deg,
                "heading_source": heading_source,
                "zero_center_xy": zero_center_xy,
                "selected_slots": {
                    int(slot_idx): {
                        "center_xy": tuple(float(v) for v in slot_row["center_xy"]),
                        "class_id": int(slot_row["class_id"]),
                        "slot_score": float(slot_row.get("confidence", 1.0)),
                    }
                    for slot_idx, slot_row in selected_slots.items()
                },
                "group_score": float(group_score),
                "confidence": float(np.mean([float(row.get("confidence", 0.0)) for row in subset_rows])),
                "robot_id": robot_id,
                "component_size": int(len(subset_rows)),
                "center_source": "slot_average_4",
                "raw_rows": subset_rows,
            }
        )
    groups.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return groups


@torch.no_grad()
def predict_reasoner_queries(
    model: P4BlurReasoner,
    image_rgb: np.ndarray,
    det_rows: list[dict[str, Any]],
    device: torch.device,
    presence_thresh: float = 0.45,
    crop_size: int = 64,
    max_nodes: int = 24,
) -> list[dict[str, Any]]:
    selected_det_rows = _select_node_detections(det_rows, max_nodes=max_nodes)
    node_item = build_spiral_node_features(
        image_rgb,
        selected_det_rows,
        image_rgb.shape[:2],
        crop_size=int(crop_size),
        max_nodes=int(max_nodes),
    )
    patches, geom, node_mask = pad_node_batch([node_item])
    patches = patches.to(device)
    geom = geom.to(device)
    node_mask = node_mask.to(device)
    output = model(patches, geom, node_mask)
    probs = torch.sigmoid(output.query_presence_logits[0]).cpu().numpy()
    centers = output.query_center_xy[0].cpu().numpy()
    heading = output.query_heading_vec[0].cpu().numpy()
    blur_vec = output.query_blur_vec[0].cpu().numpy()
    blur_len = output.query_blur_len[0].cpu().numpy()
    id_logits = output.query_id_logits[0].cpu().numpy()
    height, width = image_rgb.shape[:2]
    rows: list[dict[str, Any]] = []
    for qi in range(output.query_presence_logits.shape[1]):
        if float(probs[qi]) < float(presence_thresh):
            continue
        rows.append(
            {
                "query_index": int(qi),
                "confidence": float(probs[qi]),
                "center_xy": (float(centers[qi, 0] * width), float(centers[qi, 1] * height)),
                "heading_deg": float(vec_to_angle_deg(float(heading[qi, 0]), float(heading[qi, 1]))),
                "blur_angle_deg": float(vec_to_angle_deg(float(blur_vec[qi, 0]), float(blur_vec[qi, 1]))),
                "blur_length_px": float(max(0.0, blur_len[qi]) * 30.0),
                "robot_id": int(np.argmax(id_logits[qi])),
                "center_source": "query_regression",
                "heading_source": "query_regression",
                "zero_center_xy": None,
                "selected_slots": {},
            }
        )
    return rows


def _match_groups_to_queries(
    groups: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    max_distance_px: float = 140.0,
) -> dict[int, int]:
    if not groups or not query_rows:
        return {}
    cost = np.zeros((len(groups), len(query_rows)), dtype=np.float32)
    for gi, group in enumerate(groups):
        for qi, query in enumerate(query_rows):
            cost[gi, qi] = _distance_xy(tuple(group["center_xy"]), tuple(query["center_xy"]))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict[int, int] = {}
    for gi, qi in zip(row_ind.tolist(), col_ind.tolist()):
        if float(cost[gi, qi]) > float(max_distance_px):
            continue
        mapping[int(gi)] = int(qi)
    return mapping


@torch.no_grad()
def predict_reasoner_robots(
    model: P4BlurReasoner,
    image_rgb: np.ndarray,
    det_rows: list[dict[str, Any]],
    device: torch.device,
    presence_thresh: float = 0.45,
    crop_size: int = 64,
    max_nodes: int = 24,
) -> list[dict[str, Any]]:
    query_rows = predict_reasoner_queries(
        model=model,
        image_rgb=image_rgb,
        det_rows=det_rows,
        device=device,
        presence_thresh=presence_thresh,
        crop_size=crop_size,
        max_nodes=max_nodes,
    )

    classical_groups = _build_classical_groups(det_rows)
    group_to_query = _match_groups_to_queries(classical_groups, query_rows)
    used_query_indices: set[int] = set()

    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(classical_groups):
        row = {
            "confidence": float(group["confidence"]),
            "center_xy": tuple(float(v) for v in group["center_xy"]),
            "center_source": str(group["center_source"]),
            "heading_deg": None,
            "heading_source": "query_regression",
            "zero_center_xy": group.get("zero_center_xy"),
            "selected_slots": dict(group.get("selected_slots", {})),
            "blur_angle_deg": 0.0,
            "blur_length_px": 0.0,
            "robot_id": group.get("robot_id"),
            "query_index": None,
        }
        query_index = group_to_query.get(group_index)
        if query_index is not None:
            used_query_indices.add(int(query_index))
            query_row = query_rows[query_index]
            row["query_index"] = int(query_row["query_index"])
            row["confidence"] = max(float(row["confidence"]), float(query_row["confidence"]))
            row["blur_angle_deg"] = float(query_row["blur_angle_deg"])
            row["blur_length_px"] = float(query_row["blur_length_px"])
            if row["robot_id"] is None:
                row["robot_id"] = int(query_row["robot_id"])
            if group.get("heading_deg") is None:
                row["heading_deg"] = float(query_row["heading_deg"])
                row["heading_source"] = "query_regression"
            else:
                row["heading_deg"] = float(group["heading_deg"])
                row["heading_source"] = str(group["heading_source"])
        else:
            if row["robot_id"] is None:
                row["robot_id"] = 0
            if group.get("heading_deg") is None:
                row["heading_deg"] = 0.0
                row["heading_source"] = "default_zero"
            else:
                row["heading_deg"] = float(group["heading_deg"])
                row["heading_source"] = str(group["heading_source"])
        rows.append(row)

    for query_row in query_rows:
        query_index = int(query_row["query_index"])
        if query_index in used_query_indices:
            continue
        rows.append(dict(query_row))

    rows.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    return rows
