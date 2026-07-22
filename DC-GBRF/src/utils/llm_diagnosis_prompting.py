from __future__ import annotations

import json
import re
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage


PROMPT_RELEASE_TO_VERSION = {
    "public_default": "obs1_v4_pairwise_fault_signature_contrast",
}
PROMPT_VERSION_TO_RELEASE = {value: key for key, value in PROMPT_RELEASE_TO_VERSION.items()}
OBSERVED_SUMMARY_VERSION = "obs1_v1"

COMMON_SYSTEM_PROMPT_BASE = """You are an expert AI diagnostician for industrial machines.
Your task is to identify the most likely failure mode from an observed causal graph.
Return exactly these 8 labeled items:
1. Initial Top Candidate:
2. Final Most Likely Failure Mode:
3. Alternative Candidate:
4. Confidence:
5. Retrieval Summary:
6. Supplementary Summary:
7. Key Evidence:
8. Why Final Over Alternative:""".strip()

PROMPT_TEMPLATES = {
    "direct": "# Observed Graph\n{observed_graph}\n\n# Observed Evidence Summary\n{observed_evidence_summary}\n\n# Supplementary Information\n{supplementary_context}",
    "direct_full_reference": "# Observed Graph\n{observed_graph}\n\n# Observed Evidence Summary\n{observed_evidence_summary}\n\n# Reference Failure Modes\n{candidate_context}\n\n# Supplementary Information\n{supplementary_context}",
    "fusion": "# Observed Graph\n{observed_graph}\n\n# Observed Evidence Summary\n{observed_evidence_summary}\n\n# Retrieval Method\n{method_name}\n\n# Compact Candidate Context\n{candidate_context}\n\n# Supplementary Information\n{supplementary_context}",
    "dc_fusion": "# Observed Graph\n{observed_graph}\n\n# Observed Evidence Summary\n{observed_evidence_summary}\n\n# Retrieval Method\n{method_name}\n\n# Domain Calibration Alpha\n{alpha_dk}\n\n# Compact Candidate Context\n{candidate_context}\n\n# Supplementary Information\n{supplementary_context}",
}


def common_system_prompt_for_version(prompt_version: str | None = None) -> str:
    return COMMON_SYSTEM_PROMPT_BASE


def build_observed_evidence_summary(observed_graph_json: dict, top_k: int = 4) -> str:
    edges = sorted(observed_graph_json.get("edges", []), key=lambda edge: float(edge.get("probability", 0.0)), reverse=True)
    if not edges:
        return "No observed edges were found."
    lines = []
    for edge in edges[:top_k]:
        lines.append(f"- {edge.get('source')} -> {edge.get('target')}: {float(edge.get('probability', 0.0)):.3f}")
    return "\n".join(lines)


def build_human_input(*, prompt_key: str, observed_graph_str: str, observed_evidence_summary: str | None, candidate_context: str, supplementary_context: str, method_name: str | None, alpha_dk: float | None, prompt_version: str | None = None) -> str:
    template = PROMPT_TEMPLATES[prompt_key]
    return template.format(
        observed_graph=observed_graph_str,
        observed_evidence_summary=observed_evidence_summary or "(empty)",
        candidate_context=candidate_context or "(empty)",
        supplementary_context=supplementary_context or "(empty)",
        method_name=method_name or "Unknown",
        alpha_dk="N/A" if alpha_dk is None else f"{float(alpha_dk):.2f}",
    )


def normalize_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
                    continue
                nested_content = item.get("content")
                if nested_content is not None:
                    nested_text = normalize_message_content(nested_content)
                    if nested_text.strip():
                        parts.append(nested_text)
        return "\n".join(part for part in parts if str(part).strip())
    return str(content)


def invoke_llm_once(*, llm_obj, system_prompt: str, human_input: str):
    response = llm_obj.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_input)])
    usage = getattr(response, "usage_metadata", {}) or {}
    answer_text = normalize_message_content(getattr(response, "content", response))
    return {
        "answer_text": answer_text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_metadata": getattr(response, "response_metadata", {}),
    }


def load_observed_graph(test_case_path) -> tuple[dict, str]:
    path_obj = Path(test_case_path)
    with open(path_obj, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return payload, json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_labeled_value(answer_text: str, label: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(answer_text)
    if not match:
        return None
    return match.group(1).strip() or None


def _find_mode_mentions(answer_text: str, mode_labels: list[str]) -> list[str]:
    mentions = []
    lowered_text = answer_text.lower()
    for mode in mode_labels:
        if str(mode).lower() in lowered_text and mode not in mentions:
            mentions.append(mode)
    return mentions


def parse_structured_answer(answer_text: str, mode_labels: list[str]) -> dict:
    initial = _extract_labeled_value(answer_text, "1. Initial Top Candidate:")
    final = _extract_labeled_value(answer_text, "2. Final Most Likely Failure Mode:")
    alternative = _extract_labeled_value(answer_text, "3. Alternative Candidate:")
    mentions = _find_mode_mentions(answer_text, mode_labels)
    return {
        "initial_top_candidate_mode": initial,
        "final_most_likely_failure_mode": final,
        "alternative_candidate_mode": alternative,
        "confidence": _extract_labeled_value(answer_text, "4. Confidence:"),
        "retrieval_summary": _extract_labeled_value(answer_text, "5. Retrieval Summary:"),
        "supplementary_summary": _extract_labeled_value(answer_text, "6. Supplementary Summary:"),
        "key_evidence": _extract_labeled_value(answer_text, "7. Key Evidence:"),
        "why_final_over_alternative": _extract_labeled_value(answer_text, "8. Why Final Over Alternative:"),
        "llm_top2_modes": [mode for mode in [final, alternative] if mode],
        "fallback_mode_mentions": mentions,
    }
