import statistics

import numpy as np


def sample_vector(
    variance: np.array,
    vector: np.array,
    n: int,
) -> np.array:
    """
    Returns a matrix of n samples for a word from the Normal distribution given by
    the vector and covariance matrix.

    Args:
        variance: Dictionary of word to variance matrices
        vector: Dictionary of word to vectors
        n: Number of samples to draw
    """
    v_sam = np.random.multivariate_normal(mean=vector, cov=variance, size=n)
    return v_sam


def delta_method_variance(
    deriv_dict: dict[str, np.array],
    variance_dict: dict[str, np.array],
) -> float:
    """
    Computes the variance of of a test statistic using the Delta Method, given a dictionary of word-level derivatives and a dictionary of word-level variances.
    """
    variance = 0
    for w in deriv_dict.keys():
        d_i = deriv_dict[w]
        var_i = variance_dict[w]
        variance += np.matmul(np.matmul(d_i, var_i), d_i.T)
    variance = np.squeeze(variance).item()
    return variance


def cosine_derivative(u: np.array, v: np.array) -> np.array:
    """
    Computes the derivative of the cosine similarity cos(u, v) with respect to the vector u.

    Args:
        u: Vector u
        v: Vector v
    """

    def cossim(u, v):
        u_re = u.reshape(-1, 1)
        v_re = v.reshape(-1, 1)
        cs = np.matmul(u_re.T, v_re) / (np.linalg.norm(u_re) * np.linalg.norm(v_re))
        return np.squeeze(cs).item()

    u_re = u.reshape(-1, 1)
    v_re = v.reshape(-1, 1)

    u_norm = np.linalg.norm(u_re)
    v_norm = np.linalg.norm(v_re)

    return v_re / (u_norm * v_norm) - cossim(u, v) * u_re / (u_norm**2)


def compute_normal_confint(
    point_estimate: float, variance: float, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Computes the 100(1-alpha)% two-sided confidence intervals for a Normal distribution.

    Args:
        point_estimate: Point estimate of the test statistic
        variance: Variance of the test statistic
        alpha: Significance level
    """
    normaldist = statistics.NormalDist()

    sd = np.sqrt(variance)
    upper = point_estimate + sd * normaldist.inv_cdf(1 - alpha / 2)
    lower = point_estimate - sd * normaldist.inv_cdf(1 - alpha / 2)
    return lower, upper
