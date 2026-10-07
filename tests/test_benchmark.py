import numpy as np
import pandas as pd

from program2cell.benchmark import evaluate_attribution, simulate_bulk


def test_simulation_and_evaluation():
    pseudobulk = pd.DataFrame(
        [[10, 0], [8, 2], [0, 10], [2, 8]],
        index=["g1", "g2", "g3", "g4"],
        columns=["Mono", "NK"],
    )
    bulk, truth = simulate_bulk(
        pseudobulk, {"Mono": 0.6, "NK": 0.4}, n_samples=3, random_state=3
    )
    assert bulk.shape == (4, 3)
    assert (bulk.to_numpy() >= 0).all()
    metrics = evaluate_attribution(truth, truth)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert np.isclose(metrics["correlation"], 1)
