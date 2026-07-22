from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity_from_df(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    a = np.asarray(df_a, dtype=float).reshape(-1)
    b = np.asarray(df_b, dtype=float).reshape(-1)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def fro_similarity_from_df(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    a = np.asarray(df_a, dtype=float)
    b = np.asarray(df_b, dtype=float)

    dist = float(np.linalg.norm(a - b, ord="fro"))
    return float(1.0 / (1.0 + dist))


def build_kb_matrices_df(
    *,
    knowledge_matrices: dict[str, np.ndarray],
    sensor_order: list[str],
    state_order: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        mode: pd.DataFrame(np.asarray(mat, dtype=float), index=sensor_order, columns=state_order)
        for mode, mat in knowledge_matrices.items()
    }


def build_mode_proximity_matrix(
    *,
    kb_matrices_df: dict[str, pd.DataFrame],
    mode_order: list[str] | None = None,
    source: str = "reference_graph",
    zero_normal_links: bool = True,
    manual_overrides: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        W_raw: raw nonnegative proximity matrix
        W_row: row-normalized proximity matrix
    """
    if mode_order is None:
        mode_order = list(kb_matrices_df.keys())

    if manual_overrides is None:
        manual_overrides = {}

    W = pd.DataFrame(0.0, index=mode_order, columns=mode_order, dtype=float)

    for mode_i in mode_order:
        for mode_j in mode_order:
            if mode_i == mode_j:
                W.loc[mode_i, mode_j] = 0.0
                continue

            if source == "reference_graph":
                sim_cos = cosine_similarity_from_df(kb_matrices_df[mode_i], kb_matrices_df[mode_j])
                sim_fro = fro_similarity_from_df(kb_matrices_df[mode_i], kb_matrices_df[mode_j])
                sim = 0.5 * sim_cos + 0.5 * sim_fro
            elif source == "manual":
                sim = 0.0
            else:
                raise ValueError(f"unknown W source: {source}")

            W.loc[mode_i, mode_j] = float(max(0.0, sim))

    if zero_normal_links and "Normal" in W.index:
        W.loc["Normal", :] = 0.0
        W.loc[:, "Normal"] = 0.0

    for (src, dst), val in manual_overrides.items():
        if src in W.index and dst in W.columns:
            W.loc[src, dst] = float(val)

    W_row = W.copy()
    for mode in W_row.index:
        row_sum = float(W_row.loc[mode].sum())
        if row_sum > 1e-12:
            W_row.loc[mode] = W_row.loc[mode] / row_sum

    return W, W_row