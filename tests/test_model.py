from types import SimpleNamespace

import numpy as np
import pandas as pd

from program2cell import Program2Cell


def test_fit_run_attributes_modules_and_saves_plot_ready_result():
    rng = np.random.default_rng(8)
    states = np.repeat(["StateA", "StateB"], 20)
    genes = [f"g{index}" for index in range(24)]
    atlas_values = np.empty((len(states), len(genes)))
    for cell, state in enumerate(states):
        high = 18 if state == "StateA" else 0.5
        low = 0.5 if state == "StateA" else 18
        atlas_values[cell, :12] = high + rng.uniform(0, 1, 12)
        atlas_values[cell, 12:] = low + rng.uniform(0, 1, 12)
    atlas = SimpleNamespace(
        X=atlas_values,
        obs=pd.DataFrame({"cell_state": states}, index=[f"c{i}" for i in range(len(states))]),
        obs_names=[f"c{i}" for i in range(len(states))],
        var_names=genes,
    )

    samples = 12
    latent_a = rng.normal(0, 3, samples)
    latent_b = rng.normal(0, 3, samples)
    bulk_values = np.vstack(
        [
            20 + latent_a + rng.normal(0, 0.1, (12, samples)),
            20 + latent_b + rng.normal(0, 0.1, (12, samples)),
        ]
    )
    bulk = pd.DataFrame(
        bulk_values,
        index=genes,
        columns=[f"sample{i}" for i in range(samples)],
    )

    model = Program2Cell(min_cells=5, n_bootstrap=5, random_state=2).fit(atlas, bulk)
    result = model.run()

    assert set(result.attribution.columns) == {"module", "state", "contribution"}
    for _, group in result.attribution.groupby("module"):
        assert (group["contribution"] >= 0).all()
        assert np.isclose(group["contribution"].sum(), 1.0)
    assert set(result.metrics.columns) == {"r2", "cosine_similarity", "attribution_entropy"}
    assert model.results_ is result
