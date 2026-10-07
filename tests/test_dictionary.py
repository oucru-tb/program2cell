import numpy as np
import pandas as pd

from program2cell.dictionary import build_dictionary


def test_dictionary_is_nonnegative_and_columns_sum_to_one():
    expression = pd.DataFrame(
        [[10, 8, 0, 0], [0, 0, 7, 9], [2, 2, 2, 2]],
        index=["mono", "nk", "shared"],
        columns=["c1", "c2", "c3", "c4"],
    )
    states = pd.Series(["Mono", "Mono", "NK", "NK"], index=expression.columns)
    dictionary, pseudobulk = build_dictionary(expression, states)

    assert list(pseudobulk.columns) == ["Mono", "NK"]
    assert np.all(dictionary.to_numpy() >= 0)
    np.testing.assert_allclose(dictionary.sum(axis=0).to_numpy(), 1.0)
