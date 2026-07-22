from __future__ import annotations

import numpy as np


def _flatten_pair(observation_matrix, kb_matrix) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(observation_matrix, dtype=float).reshape(-1)
    right = np.asarray(kb_matrix, dtype=float).reshape(-1)
    return left, right


def calculate_cosine_similarity(observation_matrix, kb_matrix) -> float:
    left, right = _flatten_pair(observation_matrix, kb_matrix)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def calculate_frobenius_similarity(observation_matrix, kb_matrix) -> float:
    distance = float(np.linalg.norm(np.asarray(observation_matrix, dtype=float) - np.asarray(kb_matrix, dtype=float), ord="fro"))
    return float(1.0 / (1.0 + distance))


def calculate_jaccard_similarity_topk(observation_matrix, kb_matrix, top_k: int = 2) -> float:
    left = np.asarray(observation_matrix, dtype=float)
    right = np.asarray(kb_matrix, dtype=float)
    left_edges = set(map(tuple, np.argwhere(left > 0)))
    right_edges = set(map(tuple, np.argwhere(right > 0)))
    if top_k > 0:
        left_top = np.argsort(left.reshape(-1))[::-1][:top_k]
        right_top = np.argsort(right.reshape(-1))[::-1][:top_k]
        left_edges = left_edges.intersection({tuple(np.unravel_index(idx, left.shape)) for idx in left_top}) or left_edges
        right_edges = right_edges.intersection({tuple(np.unravel_index(idx, right.shape)) for idx in right_top}) or right_edges
    union = left_edges | right_edges
    if not union:
        return 1.0
    return float(len(left_edges & right_edges) / len(union))


def calculate_pearson_similarity(observation_matrix, kb_matrix) -> float:
    left, right = _flatten_pair(observation_matrix, kb_matrix)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    if np.isnan(corr):
        return 0.0
    return float((corr + 1.0) / 2.0)
