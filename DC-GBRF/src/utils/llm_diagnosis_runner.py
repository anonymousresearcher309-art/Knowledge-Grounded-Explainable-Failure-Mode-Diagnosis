from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from utils.interactive_context_builders import (
    build_dc_gbrf_candidate_context,
    build_direct_candidate_context,
    build_gbrf_candidate_context,
    build_supplementary_context,
)
from utils.llm_diagnosis_prompting import (
    build_human_input,
    build_observed_evidence_summary,
    invoke_llm_once,
    load_observed_graph,
    parse_structured_answer,
)


def reciprocal_rank(gt_mode: str | None, ranked_modes: list[str]) -> float:
    if gt_mode is None or not ranked_modes:
        return np.nan
    for index, mode in enumerate(ranked_modes, start=1):
        if mode == gt_mode:
            return 1.0 / index
    return 0.0


def ranking_from_trace(trace) -> list[str]:
    if not isinstance(trace, dict):
        return []
    for key in ["ranked_modes", "fused_modes", "retrieval_ranked_modes"]:
        value = trace.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
    fused_table = trace.get("fused_table_after_dk") or trace.get("fused_table") or []
    return [row.get("mode") for row in fused_table if isinstance(row, dict) and row.get("mode")]


def extract_gt_mode(test_case_path: Path) -> str | None:
    match = re.search(r"GT-([A-Za-z0-9_]+)", test_case_path.stem)
    return match.group(1) if match else None


def prepare_single_case_payload(
    *,
    test_case_path_resolved: Path,
    prompt_version: str,
    supplementary_variant: str,
    supplementary_retriever,
    method_name: str,
    alpha_dk: float | None,
    observed_summary_top_k: int,
    knowledge_graph_docs: list,
    knowledge_matrices: dict,
    graph_method_registry: dict,
    fusion_base_methods: list[str],
    sensor_order: list[str],
    state_order: list[str],
    mode_proximity_matrix,
    ged_default_params: dict,
) -> dict:
    observed_graph_json, observed_graph_str = load_observed_graph(test_case_path_resolved)
    observed_evidence_summary = build_observed_evidence_summary(observed_graph_json, top_k=observed_summary_top_k)

    if supplementary_variant == "none":
        supplementary_context = "Supplementary information not used."
    else:
        supplementary_context = build_supplementary_context(
            observed_graph_str=observed_graph_str,
            retriever_supplementary=supplementary_retriever,
            top_k=2,
        )

    if method_name == "Direct":
        build_result = None
        candidate_context = ""
        prompt_key = "direct"
    elif method_name == "Full-reference Direct":
        build_result = build_direct_candidate_context(knowledge_graph_docs=knowledge_graph_docs)
        candidate_context = build_result.candidate_context
        prompt_key = "direct_full_reference"
    elif method_name == "GBRF":
        build_result = build_gbrf_candidate_context(
            observed_graph_json=observed_graph_json,
            knowledge_graph_docs=knowledge_graph_docs,
            knowledge_matrices=knowledge_matrices,
            graph_method_registry=graph_method_registry,
            fusion_base_methods=fusion_base_methods,
            sensor_order=sensor_order,
            state_order=state_order,
            retrieval_k=3,
            fusion_rrf_k=10,
            ged_default_params=ged_default_params,
        )
        candidate_context = build_result.candidate_context
        prompt_key = "fusion"
    elif method_name == "DC-GBRF":
        build_result = build_dc_gbrf_candidate_context(
            observed_graph_json=observed_graph_json,
            knowledge_graph_docs=knowledge_graph_docs,
            knowledge_matrices=knowledge_matrices,
            graph_method_registry=graph_method_registry,
            fusion_base_methods=fusion_base_methods,
            sensor_order=sensor_order,
            state_order=state_order,
            alpha_dk=float(alpha_dk),
            mode_proximity_matrix=mode_proximity_matrix,
            retrieval_k=3,
            fusion_rrf_k=10,
            ged_default_params=ged_default_params,
        )
        candidate_context = build_result.candidate_context
        prompt_key = "dc_fusion"
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")

    human_input = build_human_input(
        prompt_key=prompt_key,
        observed_graph_str=observed_graph_str,
        observed_evidence_summary=observed_evidence_summary,
        candidate_context=candidate_context,
        supplementary_context=supplementary_context,
        method_name=method_name,
        alpha_dk=(alpha_dk if method_name == "DC-GBRF" else None),
        prompt_version=prompt_version,
    )

    return {
        "observed_graph_json": observed_graph_json,
        "observed_graph_str": observed_graph_str,
        "observed_evidence_summary": observed_evidence_summary,
        "candidate_context": candidate_context,
        "supplementary_context": supplementary_context,
        "prompt_key": prompt_key,
        "human_input": human_input,
        "retrieval_trace": None if build_result is None else build_result.trace,
    }


def run_single_case(
    *,
    llm,
    system_prompt: str,
    mode_labels: list[str],
    run_root: Path,
    stage_name: str,
    test_case_path_resolved: Path,
    method_name: str,
    llm_name: str,
    llm_build_name: str,
    supplementary_variant: str,
    alpha_dk: float | None,
    prompt_version: str,
    observed_summary_version: str,
    observed_summary_top_k: int,
    supplementary_retriever,
    knowledge_graph_docs: list,
    knowledge_matrices: dict,
    graph_method_registry: dict,
    fusion_base_methods: list[str],
    sensor_order: list[str],
    state_order: list[str],
    mode_proximity_matrix,
    ged_default_params: dict,
) -> dict:
    prepared = prepare_single_case_payload(
        test_case_path_resolved=test_case_path_resolved,
        prompt_version=prompt_version,
        supplementary_variant=supplementary_variant,
        supplementary_retriever=supplementary_retriever,
        method_name=method_name,
        alpha_dk=alpha_dk,
        observed_summary_top_k=observed_summary_top_k,
        knowledge_graph_docs=knowledge_graph_docs,
        knowledge_matrices=knowledge_matrices,
        graph_method_registry=graph_method_registry,
        fusion_base_methods=fusion_base_methods,
        sensor_order=sensor_order,
        state_order=state_order,
        mode_proximity_matrix=mode_proximity_matrix,
        ged_default_params=ged_default_params,
    )
    llm_result = invoke_llm_once(llm_obj=llm, system_prompt=system_prompt, human_input=prepared["human_input"])
    parsed = parse_structured_answer(llm_result["answer_text"], mode_labels)
    gt_mode = extract_gt_mode(test_case_path_resolved)
    retrieval_ranking = ranking_from_trace(prepared["retrieval_trace"])
    llm_ranked_modes = list(parsed["llm_top2_modes"]) or parsed["fallback_mode_mentions"][:2]

    return {
        "run_timestamp": datetime.now().isoformat(),
        "run_root": str(run_root),
        "stage_name": stage_name,
        "test_case_path": str(test_case_path_resolved),
        "file_name": test_case_path_resolved.name,
        "gt_mode": gt_mode,
        "method_name": method_name,
        "llm_name": llm_name,
        "llm_build_name": llm_build_name,
        "supplementary_variant": supplementary_variant,
        "alpha_dk": alpha_dk if method_name == "DC-GBRF" else None,
        "prompt_version": prompt_version,
        "observed_summary_version": observed_summary_version,
        "prompt_key": prepared["prompt_key"],
        "observed_evidence_summary": prepared["observed_evidence_summary"],
        "candidate_context": prepared["candidate_context"],
        "supplementary_context": prepared["supplementary_context"],
        "raw_answer": llm_result["answer_text"],
        "structured_parse": parsed,
        "final_most_likely_failure_mode": parsed["final_most_likely_failure_mode"],
        "initial_top_candidate": parsed["initial_top_candidate_mode"],
        "alternative_candidate": parsed["alternative_candidate_mode"],
        "retrieval_ranked_modes": retrieval_ranking,
        "llm_ranked_modes": llm_ranked_modes,
        "retrieval_rr": reciprocal_rank(gt_mode, retrieval_ranking),
        "llm_rr": reciprocal_rank(gt_mode, llm_ranked_modes),
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"],
        "total_tokens": llm_result["total_tokens"],
        "response_metadata": llm_result["response_metadata"],
        "human_input": prepared["human_input"],
        "retrieval_trace": prepared["retrieval_trace"],
    }
