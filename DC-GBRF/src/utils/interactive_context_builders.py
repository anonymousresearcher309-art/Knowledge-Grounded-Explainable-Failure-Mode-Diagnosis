from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain.schema import Document

from utils.inference import json_to_matrix


@dataclass
class CandidateBuildResult:
    candidate_context: str
    trace: dict[str, Any]


def _call_retriever(retriever, query: str) -> list[Document]:
    if retriever is None:
        return []
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query) or [])
    return list(retriever.get_relevant_documents(query) or [])


def _safe_json_pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return str(text)


def _truncate(text: str, max_chars: int = 1000) -> str:
    text = str(text).strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n... (truncated)"


def _find_doc_by_mode(knowledge_graph_docs: list[Document], mode: str) -> Document | None:
    for doc in knowledge_graph_docs:
        if doc.metadata.get("mode") == mode:
            return doc
    return None


def build_supplementary_context(*, observed_graph_str: str, retriever_supplementary, top_k: int = 2) -> str:
    docs = _call_retriever(retriever_supplementary, observed_graph_str)[:top_k]
    if not docs:
        return "No supplementary information retrieved."
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source_name = doc.metadata.get("source", f"supplementary_{index}")
        blocks.append(f"## Supplementary {index}: {source_name}\n{_truncate(_safe_json_pretty(doc.page_content))}")
    return "\n\n".join(blocks)


def build_supplementary_context_from_json(
    *,
    supplementary_source_path,
    selected_modes: list[str] | None = None,
) -> str:
    path_obj = Path(supplementary_source_path)
    if not path_obj.exists():
        return f"Supplementary JSON not found: {path_obj}"

    with open(path_obj, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    lines = []
    knowledge_base_name = payload.get("knowledge_base_name")
    system_name = payload.get("machine_info", {}).get("system_name")
    description = payload.get("machine_info", {}).get("description")

    if knowledge_base_name:
        lines.append(f"- knowledge_base_name: {knowledge_base_name}")
    if system_name:
        lines.append(f"- system_name: {system_name}")
    if description:
        lines.append(f"- description: {description}")
    if selected_modes:
        lines.append(f"- preview_target_modes: {', '.join(selected_modes)}")

    payload_keys = sorted(payload.keys())
    lines.append(f"- available_top_level_keys: {', '.join(payload_keys)}")

    if len(payload_keys) <= 2 and "machine_info" in payload and "knowledge_base_name" in payload:
        lines.append("- note: current lite JSON stores package-level metadata only; richer supplementary passages are loaded from the vector store during the API run cell.")

    return "\n".join(lines)


def build_direct_candidate_context(*, knowledge_graph_docs: list[Document]) -> CandidateBuildResult:
    blocks = []
    modes = []
    for index, doc in enumerate(knowledge_graph_docs, start=1):
        mode = doc.metadata.get("mode", f"mode_{index}")
        modes.append(mode)
        blocks.append(f"## Reference Mode {index}: {mode}\n{_truncate(_safe_json_pretty(doc.page_content))}")
    return CandidateBuildResult(candidate_context="\n\n".join(blocks), trace={"strategy_family": "direct", "modes": modes})


def _compute_single_graph_scores(
    *,
    observed_graph_json: dict,
    knowledge_matrices: dict[str, np.ndarray],
    graph_method_registry: dict[str, Any],
    method_name: str,
    sensor_order: list[str],
    state_order: list[str],
    ged_default_params: dict | None = None,
) -> dict[str, float]:
    observation_matrix = json_to_matrix(observed_graph_json, sensor_order, state_order)
    score_func = graph_method_registry[method_name]
    return {mode: float(score_func(observation_matrix, kb_matrix)) for mode, kb_matrix in knowledge_matrices.items()}


def _rank_positions(score_dict: dict[str, float]) -> dict[str, int]:
    return {mode: index + 1 for index, (mode, _) in enumerate(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))}


def _topk_mode_set(score_dict: dict[str, float], top_k: int) -> set[str]:
    return {mode for mode, _ in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]}


def _minmax_normalize_scores(score_dict: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(score_dict.values()), dtype=float)
    if values.size == 0:
        return {}
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return {key: 1.0 for key in score_dict}
    return {key: float((value - low) / (high - low)) for key, value in score_dict.items()}


def _build_mode_evidence_rows(*, score_dict_by_method: dict[str, dict[str, float]], mode_order_local: list[str], method_names: list[str], evidence_top_k: int = 3, rrf_k: int = 10) -> list[dict]:
    normalized = {method_name: _minmax_normalize_scores(score_dict_by_method[method_name]) for method_name in method_names}
    ranks = {method_name: _rank_positions(score_dict_by_method[method_name]) for method_name in method_names}
    topk = {method_name: _topk_mode_set(score_dict_by_method[method_name], evidence_top_k) for method_name in method_names}
    candidate_modes = set().union(*topk.values()) if topk else set()

    rows = []
    for mode in mode_order_local:
        if mode not in candidate_modes:
            continue
        rrf_score = sum(1.0 / (rrf_k + ranks[method_name][mode]) for method_name in method_names)
        support = [method_name for method_name in method_names if mode in topk[method_name]]
        rows.append({
            "mode": mode,
            "rrf_score": float(rrf_score),
            "topk_vote_count": int(len(support)),
            "best_rank": int(min(ranks[method_name][mode] for method_name in method_names)),
            "mean_rank": float(np.mean([ranks[method_name][mode] for method_name in method_names])),
            "norm_score_mean": float(np.mean([normalized[method_name][mode] for method_name in method_names])),
            "supporting_methods": "|".join(support),
        })
    return rows


def _sort_fusion_table_before(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["rrf_score", "topk_vote_count", "norm_score_mean", "best_rank"], ascending=[False, False, False, True]).reset_index(drop=True)


def _coerce_mode_proximity_matrix(mode_proximity_matrix, modes: list[str]) -> pd.DataFrame:
    if isinstance(mode_proximity_matrix, pd.DataFrame):
        return mode_proximity_matrix.loc[modes, modes]
    array = np.asarray(mode_proximity_matrix, dtype=float)
    return pd.DataFrame(array, index=modes, columns=modes)


def _apply_domain_calibration_to_evidence(df_evidence: pd.DataFrame, W: pd.DataFrame, alpha_dk: float = 0.15) -> pd.DataFrame:
    calibrated = df_evidence.copy().reset_index(drop=True)
    base_rrf = {row["mode"]: float(row["rrf_score"]) for _, row in calibrated.iterrows()}
    calibrated["neighbor_support"] = calibrated["mode"].map(lambda mode: sum(float(W.loc[mode, other]) * base_rrf[other] for other in calibrated["mode"] if other != mode))
    calibrated["rrf_score_base"] = calibrated["rrf_score"].astype(float)
    calibrated["rrf_score_dk"] = (1.0 - alpha_dk) * calibrated["rrf_score_base"] + alpha_dk * calibrated["neighbor_support"]
    before = _sort_fusion_table_before(calibrated.copy())
    after = calibrated.sort_values(["rrf_score_dk", "topk_vote_count", "norm_score_mean", "best_rank"], ascending=[False, False, False, True]).reset_index(drop=True)
    before_rank = {mode: index + 1 for index, mode in enumerate(before["mode"].tolist())}
    after_rank = {mode: index + 1 for index, mode in enumerate(after["mode"].tolist())}
    after["rank_before_rrf"] = after["mode"].map(before_rank)
    after["rank_after_rrf_dk"] = after["mode"].map(after_rank)
    return after


def _format_candidate_blocks(rows: pd.DataFrame, knowledge_graph_docs: list[Document], retrieval_k: int, alpha_dk: float | None = None) -> str:
    blocks = []
    for index, row in enumerate(rows.head(retrieval_k).itertuples(index=False), start=1):
        doc = _find_doc_by_mode(knowledge_graph_docs, row.mode)
        header = [f"## Candidate {index}: {row.mode}"]
        if alpha_dk is None:
            header.extend([
                f"- rrf_score: {row.rrf_score:.6f}",
                f"- topk_vote_count: {int(row.topk_vote_count)}",
                f"- supporting_methods: {row.supporting_methods}",
            ])
        else:
            header.extend([
                f"- alpha_dk: {alpha_dk:.2f}",
                f"- rrf_score_base: {row.rrf_score_base:.6f}",
                f"- neighbor_support: {row.neighbor_support:.6f}",
                f"- rrf_score_dk: {row.rrf_score_dk:.6f}",
            ])
        content = _truncate(_safe_json_pretty(doc.page_content), max_chars=1200) if doc else "Reference graph document not found."
        blocks.append("\n".join(header + ["", content]))
    return "\n\n".join(blocks)


def build_gbrf_candidate_context(*, observed_graph_json: dict, knowledge_graph_docs: list[Document], knowledge_matrices: dict[str, np.ndarray], graph_method_registry: dict[str, Any], fusion_base_methods: list[str], sensor_order: list[str], state_order: list[str], retrieval_k: int = 3, fusion_rrf_k: int = 10, ged_default_params: dict | None = None) -> CandidateBuildResult:
    score_dict_by_method = {method_name: _compute_single_graph_scores(observed_graph_json=observed_graph_json, knowledge_matrices=knowledge_matrices, graph_method_registry=graph_method_registry, method_name=method_name, sensor_order=sensor_order, state_order=state_order, ged_default_params=ged_default_params) for method_name in fusion_base_methods}
    fused_df = pd.DataFrame(_build_mode_evidence_rows(score_dict_by_method=score_dict_by_method, mode_order_local=list(knowledge_matrices.keys()), method_names=fusion_base_methods, evidence_top_k=retrieval_k, rrf_k=fusion_rrf_k))
    if fused_df.empty:
        return CandidateBuildResult(candidate_context="No GBRF candidates found.", trace={"strategy_family": "gbrf", "fused_table": []})
    fused_df = _sort_fusion_table_before(fused_df)
    return CandidateBuildResult(candidate_context=_format_candidate_blocks(fused_df, knowledge_graph_docs, retrieval_k), trace={"strategy_family": "gbrf", "fused_table": fused_df.to_dict(orient="records")})


def build_dc_gbrf_candidate_context(*, observed_graph_json: dict, knowledge_graph_docs: list[Document], knowledge_matrices: dict[str, np.ndarray], graph_method_registry: dict[str, Any], fusion_base_methods: list[str], sensor_order: list[str], state_order: list[str], alpha_dk: float, mode_proximity_matrix, retrieval_k: int = 3, fusion_rrf_k: int = 10, ged_default_params: dict | None = None) -> CandidateBuildResult:
    score_dict_by_method = {method_name: _compute_single_graph_scores(observed_graph_json=observed_graph_json, knowledge_matrices=knowledge_matrices, graph_method_registry=graph_method_registry, method_name=method_name, sensor_order=sensor_order, state_order=state_order, ged_default_params=ged_default_params) for method_name in fusion_base_methods}
    fused_df = pd.DataFrame(_build_mode_evidence_rows(score_dict_by_method=score_dict_by_method, mode_order_local=list(knowledge_matrices.keys()), method_names=fusion_base_methods, evidence_top_k=retrieval_k, rrf_k=fusion_rrf_k))
    if fused_df.empty:
        return CandidateBuildResult(candidate_context="No DC-GBRF candidates found.", trace={"strategy_family": "dc_gbrf", "fused_table_after_dk": []})
    modes = fused_df["mode"].tolist()
    w_df = _coerce_mode_proximity_matrix(mode_proximity_matrix, modes)
    after_df = _apply_domain_calibration_to_evidence(fused_df, w_df, alpha_dk=float(alpha_dk))
    return CandidateBuildResult(candidate_context=_format_candidate_blocks(after_df, knowledge_graph_docs, retrieval_k, alpha_dk=float(alpha_dk)), trace={"strategy_family": "dc_gbrf", "alpha_dk": float(alpha_dk), "fused_table_after_dk": after_df.to_dict(orient="records")})
