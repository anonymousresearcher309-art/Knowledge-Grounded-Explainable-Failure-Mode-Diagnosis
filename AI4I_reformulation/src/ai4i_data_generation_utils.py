from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SENSOR_ORDER = [
    "thermal_risk",
    "low_speed_risk",
    "low_power_risk",
    "high_power_risk",
    "wear_risk",
    "overstrain_risk",
]
DEFAULT_STATE_ORDER = ["safe", "caution", "warning", "danger"]
DEFAULT_MODE_ORDER = ["Normal", "HDF", "TWF", "OSF", "PWF_low", "PWF_high"]
DEFAULT_MODE_RELEVANT_SENSORS = {
    "Normal": DEFAULT_SENSOR_ORDER.copy(),
    "HDF": ["thermal_risk", "low_speed_risk"],
    "TWF": ["wear_risk", "overstrain_risk"],
    "OSF": ["overstrain_risk", "wear_risk"],
    "PWF_low": ["low_power_risk"],
    "PWF_high": ["high_power_risk"],
}


def make_state_distribution(safe: float, caution: float, warning: float, danger: float) -> dict[str, float]:
    return {
        "safe": float(safe),
        "caution": float(caution),
        "warning": float(warning),
        "danger": float(danger),
    }


def make_safe_distribution() -> dict[str, float]:
    return make_state_distribution(1.0, 0.0, 0.0, 0.0)


def get_sparse_reference_templates() -> dict[str, dict[str, dict[str, float]]]:
    safe = make_safe_distribution()
    caution_dominant = make_state_distribution(0.0, 0.85, 0.15, 0.0)
    warning_dominant = make_state_distribution(0.0, 0.15, 0.85, 0.0)
    danger_dominant = make_state_distribution(0.0, 0.0, 0.20, 0.80)

    return {
        "Normal": {
            "thermal_risk": safe,
            "low_speed_risk": safe,
            "low_power_risk": safe,
            "high_power_risk": safe,
            "wear_risk": safe,
            "overstrain_risk": safe,
        },
        "HDF": {
            "thermal_risk": warning_dominant,
            "low_speed_risk": warning_dominant,
        },
        "TWF": {
            "wear_risk": warning_dominant,
            "overstrain_risk": caution_dominant,
        },
        "OSF": {
            "overstrain_risk": danger_dominant,
            "wear_risk": caution_dominant,
        },
        "PWF_low": {
            "low_power_risk": danger_dominant,
        },
        "PWF_high": {
            "high_power_risk": danger_dominant,
        },
    }


def blend_state_distributions(
    base_dist: dict[str, float],
    empirical_dist: dict[str, float],
    alpha: float,
    state_order: list[str],
) -> dict[str, float]:
    return {
        state: (1.0 - alpha) * float(base_dist[state]) + alpha * float(empirical_dist[state]) for state in state_order
    }


@dataclass
class SplitConfig:
    train_frac: float = 0.60
    test_frac_within_holdout: float = 0.50
    random_state: int = 42
    strategy: str = "ambiguity_focused"
    ambiguity_sort_desc: bool = True


@dataclass
class StageConfig:
    stage_name: str
    description: str
    reference_graph_policy: str
    hybrid_alpha: float = 0.0
    run_group: str = "ai4i_graph_gen_base"
    sensor_order: list[str] = field(default_factory=lambda: DEFAULT_SENSOR_ORDER.copy())
    state_order: list[str] = field(default_factory=lambda: DEFAULT_STATE_ORDER.copy())
    mode_order: list[str] = field(default_factory=lambda: DEFAULT_MODE_ORDER.copy())
    mode_relevant_sensors: dict[str, list[str]] = field(
        default_factory=lambda: {k: v.copy() for k, v in DEFAULT_MODE_RELEVANT_SENSORS.items()}
    )
    risk_scale: dict[str, float] = field(default_factory=dict)
    sensor_state_centers: dict[str, np.ndarray] = field(default_factory=dict)
    sensor_state_width: dict[str, float] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class OutputPaths:
    run_root: Path
    graph_dir: Path
    knowledge_base_dir: Path
    supplementary_dir: Path
    split_manifest_csv: Path
    split_summary_json: Path
    knowledge_base_json: Path
    supplementary_full_json: Path
    supplementary_lite_json: Path


def make_default_stage_configs() -> dict[str, StageConfig]:
    risk_scale = {
        "thermal_risk": 2.6,
        "low_speed_risk": 190.0,
        "low_power_risk": 1600.0,
        "high_power_risk": 2200.0,
        "wear_risk": 55.0,
        "overstrain_risk": 2600.0,
    }
    sensor_state_centers = {
        "thermal_risk": np.array([0.00, 0.26, 0.56, 0.86], dtype=float),
        "low_speed_risk": np.array([0.00, 0.28, 0.58, 0.88], dtype=float),
        "low_power_risk": np.array([0.00, 0.12, 0.30, 0.60], dtype=float),
        "high_power_risk": np.array([0.00, 0.12, 0.30, 0.60], dtype=float),
        "wear_risk": np.array([0.00, 0.30, 0.58, 0.86], dtype=float),
        "overstrain_risk": np.array([0.00, 0.20, 0.42, 0.72], dtype=float),
    }
    sensor_state_width = {
        "thermal_risk": 0.36,
        "low_speed_risk": 0.36,
        "low_power_risk": 0.24,
        "high_power_risk": 0.24,
        "wear_risk": 0.30,
        "overstrain_risk": 0.26,
    }

    base = dict(risk_scale=risk_scale, sensor_state_centers=sensor_state_centers, sensor_state_width=sensor_state_width)
    configs = {
        "rule_based": StageConfig("rule_based", "Rule-based reference graphs.", "rule_based", 0.0, **base),
        "hybrid_0.2": StageConfig("hybrid_0.2", "Hybrid reference graphs with alpha 0.2.", "hybrid", 0.2, **base),
        "hybrid_0.4": StageConfig("hybrid_0.4", "Hybrid reference graphs with alpha 0.4.", "hybrid", 0.4, **base),
        "hybrid_0.6": StageConfig("hybrid_0.6", "Hybrid reference graphs with alpha 0.6.", "hybrid", 0.6, **base),
        "hybrid_0.8": StageConfig("hybrid_0.8", "Hybrid reference graphs with alpha 0.8.", "hybrid", 0.8, **base),
        "empirical": StageConfig("empirical", "Empirical reference graphs.", "empirical", 1.0, **base),
    }

    empirical_iid = StageConfig(
        "empirical_iid",
        "Empirical reference graphs with iid validation/test holdout for split comparison.",
        "empirical",
        1.0,
        **base,
    )
    empirical_iid.split = SplitConfig(
        train_frac=0.60,
        test_frac_within_holdout=0.50,
        random_state=42,
        strategy="iid_holdout",
        ambiguity_sort_desc=False,
    )
    configs[empirical_iid.stage_name] = empirical_iid
    return configs


def make_output_paths(output_root: str | Path, stage_name: str) -> OutputPaths:
    run_root = Path(output_root) / stage_name
    return OutputPaths(
        run_root=run_root,
        graph_dir=run_root / "graphs" / "test",
        knowledge_base_dir=run_root / "knowledge_base",
        supplementary_dir=run_root / "supplementary",
        split_manifest_csv=run_root / "split_manifest.csv",
        split_summary_json=run_root / "split_summary.json",
        knowledge_base_json=run_root / "knowledge_base" / f"knowledge_base_{stage_name}_prototypical.json",
        supplementary_full_json=run_root / "supplementary" / f"supplementary_knowledge_{stage_name}_full.json",
        supplementary_lite_json=run_root / "supplementary" / f"supplementary_knowledge_{stage_name}_lite.json",
    )


def ensure_output_dirs(paths: OutputPaths) -> None:
    for path in [
        paths.run_root,
        paths.graph_dir,
        paths.knowledge_base_dir,
        paths.supplementary_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def path_to_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def path_relative_to(path: str | Path, base: str | Path | None = None) -> str:
    target = Path(path)
    if base is None:
        return path_to_posix(target)
    try:
        return target.relative_to(Path(base)).as_posix()
    except ValueError:
        return path_to_posix(target)


def load_ai4i_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.rename(
        columns={
            "Air temperature [K]": "air_temp",
            "Process temperature [K]": "process_temp",
            "Rotational speed [rpm]": "rpm",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
        }
    )
    return df


def add_rule_aligned_channels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    type_threshold = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
    df["thermal_margin"] = df["process_temp"] - df["air_temp"]
    df["power_w"] = df["torque"] * (2.0 * np.pi * df["rpm"] / 60.0)
    df["overstrain_raw"] = df["tool_wear"] * df["torque"]
    df["overstrain_threshold"] = df["Type"].map(type_threshold)
    df["thermal_risk"] = np.clip(8.6 - df["thermal_margin"], 0, None)
    df["low_speed_risk"] = np.clip(1380 - df["rpm"], 0, None)
    df["low_power_risk"] = np.clip(3500 - df["power_w"], 0, None)
    df["high_power_risk"] = np.clip(df["power_w"] - 9000, 0, None)
    df["wear_risk"] = np.clip(df["tool_wear"] - 200, 0, None)
    df["overstrain_risk"] = np.clip(df["overstrain_raw"] - df["overstrain_threshold"], 0, None)
    return df


def assign_primary_mode(row: pd.Series) -> str:
    mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    active = [mode for mode in mode_cols if row[mode] == 1]
    if len(active) == 0:
        return "Normal"
    if len(active) == 1:
        return active[0]
    return "Multi"


def add_mode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    df["mode_count"] = df[mode_cols].sum(axis=1)
    df["primary_mode"] = df.apply(assign_primary_mode, axis=1)
    df["benchmark_mode"] = df["primary_mode"]
    pwf_mask = df["PWF"] == 1
    df.loc[pwf_mask, "benchmark_mode"] = np.where(df.loc[pwf_mask, "power_w"] < 3500, "PWF_low", "PWF_high")
    return df


def triangular_membership(x: float, centers: np.ndarray, width: float) -> np.ndarray:
    d = np.abs(centers - x)
    scores = np.clip(1.0 - d / width, 0.0, None)
    if scores.sum() == 0:
        scores = np.zeros(len(centers), dtype=float)
        scores[np.argmin(d)] = 1.0
    return scores / scores.sum()


def scale_sensor_value(sensor: str, value: float, config: StageConfig) -> float:
    scaled = float(value) / float(config.risk_scale[sensor])
    return float(np.clip(scaled, 0.0, 1.0))


def risk_to_state_probs(row: pd.Series, config: StageConfig) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for sensor in config.sensor_order:
        scaled = scale_sensor_value(sensor, row[sensor], config)
        probs = triangular_membership(scaled, config.sensor_state_centers[sensor], config.sensor_state_width[sensor])
        result[sensor] = dict(zip(config.state_order, probs))
    return result


def row_to_graph(row: pd.Series, config: StageConfig, mode_key: str = "benchmark_mode") -> dict[str, Any]:
    state_probs = risk_to_state_probs(row, config)
    nodes = [{"id": sensor, "type": "Sensor"} for sensor in config.sensor_order] + [{"id": state, "type": "State"} for state in config.state_order]
    edges: list[dict[str, Any]] = []
    for sensor in config.sensor_order:
        for state in config.state_order:
            probability = float(state_probs[sensor][state])
            if probability > 1e-6:
                edges.append({"source": sensor, "target": state, "probability": round(probability, 4)})
    return {"mode": row[mode_key], "nodes": nodes, "edges": edges}


def normalized_entropy(prob_vec: np.ndarray) -> float:
    p = np.asarray(prob_vec, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def compute_row_ambiguity(row: pd.Series, config: StageConfig) -> pd.Series:
    mode = row["benchmark_mode"]
    sensors = config.mode_relevant_sensors.get(mode)
    if sensors is None:
        return pd.Series({"ambiguity_entropy": np.nan, "ambiguity_softness": np.nan, "ambiguity_score": np.nan})
    probs = risk_to_state_probs(row, config)
    entropy_scores = []
    softness_scores = []
    for sensor in sensors:
        prob_vec = np.array([probs[sensor][state] for state in config.state_order], dtype=float)
        entropy_scores.append(normalized_entropy(prob_vec))
        softness_scores.append(1.0 - float(prob_vec.max()))
    ambiguity_entropy = float(np.mean(entropy_scores))
    ambiguity_softness = float(np.mean(softness_scores))
    ambiguity_score = 0.6 * ambiguity_entropy + 0.4 * ambiguity_softness
    return pd.Series(
        {
            "ambiguity_entropy": ambiguity_entropy,
            "ambiguity_softness": ambiguity_softness,
            "ambiguity_score": ambiguity_score,
        }
    )


def add_ambiguity_scores(df: pd.DataFrame, config: StageConfig) -> pd.DataFrame:
    ambiguity_df = df.apply(lambda row: compute_row_ambiguity(row, config), axis=1)
    return pd.concat([df, ambiguity_df], axis=1)


def split_holdout_by_ambiguity(holdout_df: pd.DataFrame, config: StageConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_parts = []
    test_parts = []
    for mode in config.mode_order:
        sub = holdout_df[holdout_df["benchmark_mode"] == mode].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["ambiguity_score", "UDI"], ascending=[not config.split.ambiguity_sort_desc, True]).reset_index(drop=True)
        if len(sub) == 1:
            n_test = 1
        else:
            n_test = max(1, int(np.ceil(len(sub) * config.split.test_frac_within_holdout)))
            n_test = min(n_test, len(sub) - 1)
        test_parts.append(sub.iloc[:n_test].copy())
        val_parts.append(sub.iloc[n_test:].copy())
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    return val_df, test_df


def build_splits(
    df: pd.DataFrame, config: StageConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    clean_df = df[df["benchmark_mode"].isin(config.mode_order)].copy()
    stress_df = df[df["primary_mode"].isin(["RNF", "Multi"])].copy()
    train_proto, holdout_df = train_test_split(
        clean_df,
        test_size=1.0 - config.split.train_frac,
        random_state=config.split.random_state,
        stratify=clean_df["benchmark_mode"],
    )
    if config.split.strategy == "ambiguity_focused":
        val_df, test_df = split_holdout_by_ambiguity(holdout_df, config)
    else:
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=config.split.test_frac_within_holdout,
            random_state=config.split.random_state,
            stratify=holdout_df["benchmark_mode"],
        )
    split_counts = {
        "prototype_construction": train_proto["benchmark_mode"].value_counts().to_dict(),
        "validation": val_df["benchmark_mode"].value_counts().to_dict(),
        "test": test_df["benchmark_mode"].value_counts().to_dict(),
        "stress_test": stress_df["primary_mode"].value_counts().to_dict(),
    }
    return train_proto, val_df, test_df, stress_df, split_counts


def aggregate_reference_graph(mode_name: str, mode_df: pd.DataFrame, config: StageConfig) -> dict[str, Any]:
    nodes = [{"id": sensor, "type": "Sensor"} for sensor in config.sensor_order] + [{"id": state, "type": "State"} for state in config.state_order]
    edges: list[dict[str, Any]] = []
    for sensor in config.sensor_order:
        mat = np.stack(
            [
                np.array([risk_to_state_probs(row, config)[sensor][state] for state in config.state_order], dtype=float)
                for _, row in mode_df.iterrows()
            ]
        )
        mean_probs = mat.mean(axis=0)
        for state, probability in zip(config.state_order, mean_probs):
            edges.append({"source": sensor, "target": state, "probability": round(float(probability), 4)})
    return {"mode": mode_name, "nodes": nodes, "edges": edges}


def build_rule_based_reference_graph(mode_name: str, config: StageConfig) -> dict[str, Any]:
    state_map = get_sparse_reference_templates().get(mode_name, {})
    safe = make_safe_distribution()
    nodes = [{"id": sensor, "type": "Sensor"} for sensor in config.sensor_order] + [{"id": state, "type": "State"} for state in config.state_order]
    edges: list[dict[str, Any]] = []
    for sensor in config.sensor_order:
        probs = state_map.get(sensor, safe)
        for state in config.state_order:
            probability = float(probs[state])
            if probability > 1e-6:
                edges.append({"source": sensor, "target": state, "probability": probability})
    return {"mode": mode_name, "nodes": nodes, "edges": edges}


def build_hybrid_reference_graph(mode_name: str, mode_df: pd.DataFrame, config: StageConfig) -> dict[str, Any]:
    empirical = aggregate_reference_graph(mode_name, mode_df, config)
    template = get_sparse_reference_templates().get(mode_name, {})
    safe = make_safe_distribution()
    relevant_sensors = set(config.mode_relevant_sensors.get(mode_name, []))
    empirical_lookup = {(edge["source"], edge["target"]): edge["probability"] for edge in empirical["edges"]}
    edges: list[dict[str, Any]] = []
    for sensor in config.sensor_order:
        base_dist = template.get(sensor, safe)
        if sensor in relevant_sensors:
            empirical_dist = {state: float(empirical_lookup.get((sensor, state), 0.0)) for state in config.state_order}
            probs = blend_state_distributions(base_dist, empirical_dist, config.hybrid_alpha, config.state_order)
        else:
            probs = safe
        for state in config.state_order:
            blended = float(probs[state])
            if blended > 1e-6:
                edges.append({"source": sensor, "target": state, "probability": round(blended, 4)})
    return {"mode": mode_name, "nodes": empirical["nodes"], "edges": edges}


def aggregate_reference_graphs(df: pd.DataFrame, config: StageConfig) -> dict[str, dict[str, Any]]:
    reference_graphs: dict[str, dict[str, Any]] = {}
    df_ref = df[df["benchmark_mode"].isin(config.mode_order)].copy()
    for mode in config.mode_order:
        mode_df = df_ref[df_ref["benchmark_mode"] == mode]
        if len(mode_df) == 0:
            continue
        if config.reference_graph_policy == "empirical":
            reference_graphs[mode] = aggregate_reference_graph(mode, mode_df, config)
        elif config.reference_graph_policy == "rule_based":
            reference_graphs[mode] = build_rule_based_reference_graph(mode, config)
        elif config.reference_graph_policy == "hybrid":
            reference_graphs[mode] = build_hybrid_reference_graph(mode, mode_df, config)
        else:
            raise ValueError(f"Unknown reference_graph_policy: {config.reference_graph_policy}")
    return reference_graphs


def build_knowledge_base_payload(
    config: StageConfig, reference_graphs: dict[str, dict[str, Any]], split_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    serializable_config = asdict(config)
    serializable_config["sensor_state_centers"] = {k: v.tolist() for k, v in config.sensor_state_centers.items()}
    return {
        "knowledge_base_name": f"knowledge_base_{config.stage_name}_prototypical",
        "machine_info": serializable_config,
        "split_counts": split_counts,
        "failure_modes": list(reference_graphs.values()),
    }


def build_supplementary_payloads(config: StageConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    full_payload = {
        "knowledge_base_name": f"supplementary_knowledge_{config.stage_name}_full",
        "machine_info": {"system_name": config.stage_name, "description": config.description},
        "notes": [
            "This file is a scaffold for later supplementary knowledge construction.",
            "Populate with curated maintenance and machine-specific documents before vector-store build.",
        ],
    }
    lite_payload = {
        "knowledge_base_name": f"supplementary_knowledge_{config.stage_name}_lite",
        "machine_info": {"system_name": config.stage_name, "description": config.description},
        "notes": ["Compact placeholder supplementary payload."],
    }
    return full_payload, lite_payload


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_split_manifest(
    train_proto: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    manifest_df = pd.concat(
        [
            train_proto.assign(split="prototype_construction"),
            val_df.assign(split="validation"),
            test_df.assign(split="test"),
            stress_df.assign(split="stress_test"),
        ],
        axis=0,
    )
    manifest_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def export_test_graphs(test_df: pd.DataFrame, graph_dir: str | Path, config: StageConfig) -> None:
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    for _, row in test_df.iterrows():
        graph = row_to_graph(row, config)
        raw_uid = row.get("UDI", row.name)
        try:
            uid = f"{int(raw_uid):06d}"
        except (TypeError, ValueError):
            uid = str(raw_uid)
        mode = str(row.get("benchmark_mode", "unknown"))
        output_path = graph_dir / f"observed_graph_UID{uid}_SPLIT-test_GT-{mode}.json"
        save_json(graph, output_path)


def build_dataset_bundle(
    dataset_path: str | Path,
    config: StageConfig,
    output_root: str | Path,
    relative_base: str | Path | None = None,
) -> dict[str, Any]:
    paths = make_output_paths(output_root, config.stage_name)
    ensure_output_dirs(paths)
    df = load_ai4i_dataframe(dataset_path)
    df = add_rule_aligned_channels(df)
    df = add_mode_columns(df)
    df = add_ambiguity_scores(df, config)
    train_proto, val_df, test_df, stress_df, split_counts = build_splits(df, config)
    reference_graphs = aggregate_reference_graphs(pd.concat([train_proto, val_df], axis=0), config)
    kb_payload = build_knowledge_base_payload(config, reference_graphs, split_counts)
    supp_full, supp_lite = build_supplementary_payloads(config)
    save_json(kb_payload, paths.knowledge_base_json)
    save_json(supp_full, paths.supplementary_full_json)
    save_json(supp_lite, paths.supplementary_lite_json)
    save_split_manifest(train_proto, val_df, test_df, stress_df, paths.split_manifest_csv)
    export_test_graphs(test_df, paths.graph_dir, config)
    summary = {
        "stage_name": config.stage_name,
        "description": config.description,
        "reference_graph_policy": config.reference_graph_policy,
        "hybrid_alpha": config.hybrid_alpha,
        "split_strategy": config.split.strategy,
        "reference_pool": "prototype_construction_plus_validation",
        "split_counts": split_counts,
        "output_root": path_relative_to(paths.run_root, relative_base),
    }
    save_json(summary, paths.split_summary_json)
    return {
        "paths": paths,
        "summary": summary,
        "dataframe": df,
        "train_proto": train_proto,
        "val_df": val_df,
        "test_df": test_df,
        "stress_df": stress_df,
        "reference_graphs": reference_graphs,
    }


__all__ = [
    "DEFAULT_MODE_ORDER",
    "DEFAULT_MODE_RELEVANT_SENSORS",
    "DEFAULT_SENSOR_ORDER",
    "DEFAULT_STATE_ORDER",
    "OutputPaths",
    "SplitConfig",
    "StageConfig",
    "add_ambiguity_scores",
    "add_mode_columns",
    "add_rule_aligned_channels",
    "aggregate_reference_graphs",
    "build_dataset_bundle",
    "build_splits",
    "ensure_output_dirs",
    "export_test_graphs",
    "load_ai4i_dataframe",
    "make_default_stage_configs",
    "make_output_paths",
    "path_relative_to",
    "path_to_posix",
    "risk_to_state_probs",
    "row_to_graph",
]