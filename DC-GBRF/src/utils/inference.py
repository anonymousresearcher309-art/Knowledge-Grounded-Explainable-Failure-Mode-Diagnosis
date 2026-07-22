from __future__ import annotations

import numpy as np
import pandas as pd


def json_to_matrix(json_data: dict, sensor_order: list[str], state_order: list[str]) -> np.ndarray:
    frame = pd.DataFrame(0.0, index=sensor_order, columns=state_order, dtype=float)
    for edge in json_data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source in frame.index and target in frame.columns:
            frame.loc[source, target] = float(edge.get("probability", 0.0))
    return frame.values
