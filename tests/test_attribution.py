import numpy as np

from program2cell.attribution import fit_simplex
from program2cell.bootstrap import bootstrap_attribution


def test_simplex_solver_recovers_mixture():
    dictionary = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    truth = np.array([0.7, 0.3])
    program = dictionary @ truth
    fitted = fit_simplex(program, dictionary, np.ones(4))
    np.testing.assert_allclose(fitted, truth, atol=1e-5)
    assert np.all(fitted >= 0)
    np.testing.assert_allclose(fitted.sum(), 1.0)


def test_bootstrap_is_reproducible_and_valid():
    dictionary = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    program = dictionary @ np.array([0.7, 0.3])
    first = bootstrap_attribution(program, dictionary, np.ones(4), n_bootstrap=5, random_state=4)
    second = bootstrap_attribution(program, dictionary, np.ones(4), n_bootstrap=5, random_state=4)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
