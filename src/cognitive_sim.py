"""The response model and the synthetic learner population.

A learner is a vector of ten latent ability values. A task configuration is
mapped to a requirement vector over the same ten dimensions, and the probability
that the learner succeeds is the weighted geometric mean of ten logistic terms,
one per dimension. The accompanying article states the model and its constants;
this module is the implementation the measurement scripts use.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

MU, SIGMA = 1.0, 0.15      # latent abilities ~ N(mu, sigma)
DISCRIMINATION_A = 10.0    # slope of the logistic response term
NOISE_EPS = 0.05           # eps ~ U(-0.05, 0.05)


@dataclass
class Param:
    ability_index: int
    max_change: float
    min: float
    max: float

    def z(self, value):
        """Normalise a raw parameter value to [0,1]."""
        if self.max == self.min:
            return 0.0
        return (float(value) - self.min) / (self.max - self.min)


class ExpParam(Param):
    """Exponential requirement modifier:  multiplier = max_change ** z.

    max_change > 1  ->  larger parameter value makes the task harder
    max_change < 1  ->  larger parameter value makes the task easier
    Monotone in z by construction, and equals 1.0 at z=0.
    """
    def multiplier(self, value):
        return float(self.max_change) ** self.z(value)


def generate_abilities(n, k, rng=None):
    rng = rng or np.random.default_rng(42)
    return rng.normal(MU, SIGMA, size=(n, k))


def success_probability(abilities, exercise, weights, param_types, params,
                        rng=None, noise=True):
    """P = prod_i sigma(a * (beta_i - theta~_i)) ** w_i over the active dimensions."""
    rng = rng or np.random.default_rng(0)
    abilities = np.atleast_2d(np.asarray(abilities, dtype=float))
    theta = np.asarray(exercise, dtype=float).copy()
    w = np.asarray(weights, dtype=float).copy()

    for ptype, value in zip(param_types, params):
        theta[ptype.ability_index] *= ptype.multiplier(value)

    active = theta > 0
    if not active.any():
        return np.full(len(abilities), 0.5), theta
    wa = w[active]
    wa = wa / wa.sum()

    beta = abilities[:, active]
    if noise:
        beta = beta + rng.uniform(-NOISE_EPS, NOISE_EPS, size=beta.shape)
    z = DISCRIMINATION_A * (beta - theta[active])
    sig = 1.0 / (1.0 + np.exp(-z))
    p = np.exp((wa * np.log(np.clip(sig, 1e-12, 1.0))).sum(axis=1))
    return p, theta


def simulate_exercise_params(n, abilities, exercise, exercise_weights,
                             param_types, params, binary=False, rng=None):
    p, modified = success_probability(abilities, exercise, exercise_weights,
                                      param_types, params, rng=rng)
    results = (p >= 0.5).astype(float) if binary else p
    return None, results, modified


def save_to_df(abilities, results, modified_exercise):
    k = np.asarray(abilities).shape[1]
    df = pd.DataFrame(abilities, columns=[f"ability_{i+1}" for i in range(k)])
    df["result"] = results
    df["person_id"] = np.arange(len(df))
    return df
