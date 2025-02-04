"""Computing privacy budgets using composition theorems."""

import numpy as np


def composition(
    epsilon: float,
    inference_budget: int,
    delta: float = 0.0,
    delta_prime: float = 10**-5,
):
    """
    Compute the total (epsilon_t, delta_t) privacy loss when making inference_budget queries, where
    each query is (epsilon, delta) private.
    delta_prime is the delta budget for the advanced composition theorem.
    """
    assert delta == 0

    # standard composition theorem
    epsilon_t, delta_t = inference_budget * epsilon, inference_budget * delta

    # advanced composition theorem
    epsilon_t_adv = np.sqrt(2 * inference_budget * np.log(1 / delta_prime)) * epsilon
    epsilon_t_adv += inference_budget * (np.exp(epsilon) - 1)
    delta_t_adv = delta_prime + inference_budget * delta

    return min(epsilon_t, epsilon_t_adv), max(delta_t, delta_t_adv)


def inverse_composition(
    epsilon_t: float, inference_budget: float, delta_t: float = 10**-5, delta_prime: float = 10**-5
):
    """
    Given a total privacy budget (epsilon_t, delta_t), compute the privacy budget for each query
    when making inference_budget queries.
    """

    assert delta_prime > 0, "del_prime must be > 0."
    assert delta_prime <= delta_t, "del_prime must be <= global delta."

    log_dp = np.log(1 / delta_prime)
    eps_ind = np.sqrt(log_dp + epsilon_t) - np.sqrt(log_dp)
    eps_ind *= np.sqrt(2 / inference_budget)
    del_ind = (delta_t - delta_prime) / inference_budget
    return max(eps_ind, epsilon_t / inference_budget), del_ind
