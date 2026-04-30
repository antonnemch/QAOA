from __future__ import annotations

import csv
import itertools
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


def bitstring_to_tuple(bits: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(bit) for bit in bits)


def bitstring_to_str(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def validate_portfolio_inputs(mu, Sigma, B):
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)

    if mu_arr.ndim != 1:
        raise ValueError("mu must be a one-dimensional array.")
    if sigma_arr.ndim != 2 or sigma_arr.shape[0] != sigma_arr.shape[1]:
        raise ValueError("Sigma must be a square matrix.")
    if sigma_arr.shape[0] != mu_arr.shape[0]:
        raise ValueError("mu and Sigma must agree on the number of assets.")
    if not np.allclose(sigma_arr, sigma_arr.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric within numerical tolerance.")

    n_assets = mu_arr.shape[0]
    budget = int(B)
    if budget < 0 or budget > n_assets:
        raise ValueError(f"Budget B must satisfy 0 <= B <= {n_assets}.")

    return mu_arr, 0.5 * (sigma_arr + sigma_arr.T), budget


def get_default_worked_example():
    mu = np.array([0.10, 0.20, 0.15], dtype=float)
    sigma = np.array(
        [
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.01],
            [0.02, 0.01, 0.04],
        ],
        dtype=float,
    )

    return {
        "name": "Worked 3-asset example",
        "mu": mu,
        "Sigma": sigma,
        "q": 1.0,
        "B": 2,
        "lam": 1.0,
    }


def generate_random_portfolio_instance(
    n_assets,
    seed=None,
    return_low=0.05,
    return_high=0.25,
    variance_low=0.02,
    variance_high=0.10,
    corr_scale=0.25,
):
    if int(n_assets) <= 0:
        raise ValueError("n_assets must be positive.")
    if corr_scale < 0.0 or corr_scale > 1.0:
        raise ValueError("corr_scale must lie in [0, 1] for stable PSD generation.")

    rng = np.random.default_rng(seed)
    mu = rng.uniform(return_low, return_high, size=int(n_assets))
    variances = rng.uniform(variance_low, variance_high, size=int(n_assets))
    stds = np.sqrt(variances)

    factors = rng.normal(size=(max(2, int(n_assets)), int(n_assets)))
    raw_corr = factors.T @ factors
    diag = np.sqrt(np.clip(np.diag(raw_corr), 1e-12, None))
    corr = raw_corr / np.outer(diag, diag)
    corr = (1.0 - corr_scale) * np.eye(int(n_assets)) + corr_scale * corr

    sigma = np.outer(stds, stds) * corr
    sigma = 0.5 * (sigma + sigma.T)
    return mu.astype(float), sigma.astype(float)


def enumerate_bitstrings(n_assets):
    return list(itertools.product([0, 1], repeat=int(n_assets)))


def expected_return(x, mu):
    x_arr = np.asarray(x, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    return float(mu_arr @ x_arr)


def portfolio_variance(x, Sigma):
    x_arr = np.asarray(x, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)
    return float(x_arr @ sigma_arr @ x_arr)


def portfolio_cost(x, mu, Sigma, q):
    return float(q * portfolio_variance(x, Sigma) - expected_return(x, mu))


def budget_penalty(x, B, lam):
    x_arr = np.asarray(x, dtype=float)
    return float(lam * (np.sum(x_arr) - int(B)) ** 2)


def penalized_cost(x, mu, Sigma, q, B, lam):
    return float(portfolio_cost(x, mu, Sigma, q) + budget_penalty(x, B, lam))


def make_portfolio_record(bits, mu, Sigma, q, B, lam):
    bit_tuple = bitstring_to_tuple(bits)
    ret = expected_return(bit_tuple, mu)
    var = portfolio_variance(bit_tuple, Sigma)
    penalty = budget_penalty(bit_tuple, B, lam)
    raw_objective = portfolio_cost(bit_tuple, mu, Sigma, q)
    penalized_objective = raw_objective + penalty

    return {
        "bitstring": bit_tuple,
        "bitstring_str": bitstring_to_str(bit_tuple),
        "selected_assets": int(sum(bit_tuple)),
        "is_feasible": int(sum(bit_tuple)) == int(B),
        "expected_return": ret,
        "variance": var,
        "stdev": float(np.sqrt(max(var, 0.0))),
        "raw_objective": raw_objective,
        "penalty": penalty,
        "penalized_objective": penalized_objective,
    }


def efficient_frontier_records(feasible_records):
    frontier = []
    for candidate in feasible_records:
        dominated = False
        for other in feasible_records:
            if other["bitstring"] == candidate["bitstring"]:
                continue

            no_worse_risk = other["stdev"] <= candidate["stdev"] + 1e-12
            no_worse_return = other["expected_return"] >= candidate["expected_return"] - 1e-12
            strictly_better = (
                other["stdev"] < candidate["stdev"] - 1e-12
                or other["expected_return"] > candidate["expected_return"] + 1e-12
            )
            if no_worse_risk and no_worse_return and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda record: (record["stdev"], -record["expected_return"]))


def bruteforce_portfolio_baseline(mu, Sigma, q, B, lam):
    mu_arr, sigma_arr, budget = validate_portfolio_inputs(mu, Sigma, B)
    all_records = [
        make_portfolio_record(bits, mu_arr, sigma_arr, q, budget, lam)
        for bits in enumerate_bitstrings(len(mu_arr))
    ]

    feasible_records = [record for record in all_records if record["is_feasible"]]
    if not feasible_records:
        raise ValueError("No feasible portfolios exist for the chosen budget.")

    best_feasible = min(feasible_records, key=lambda record: record["penalized_objective"])
    best_penalized = min(all_records, key=lambda record: record["penalized_objective"])
    frontier_records = efficient_frontier_records(feasible_records)

    return {
        "n_assets": len(mu_arr),
        "records": sorted(all_records, key=lambda record: record["penalized_objective"]),
        "feasible_records": sorted(
            feasible_records, key=lambda record: record["penalized_objective"]
        ),
        "best_feasible": best_feasible,
        "best_penalized": best_penalized,
        "efficient_frontier_records": frontier_records,
    }


def portfolio_qubo_coeffs(mu, Sigma, q, B, lam):
    mu_arr, sigma_arr, budget = validate_portfolio_inputs(mu, Sigma, B)
    const = float(lam * (budget**2))
    linear = q * np.diag(sigma_arr) - mu_arr + lam * (1 - 2 * budget)

    quad = {}
    for i in range(len(mu_arr)):
        for j in range(i + 1, len(mu_arr)):
            quad[(i, j)] = float(2 * q * sigma_arr[i, j] + 2 * lam)

    return const, np.asarray(linear, dtype=float), quad


def evaluate_qubo_cost(bits, const, linear, quad):
    x = np.asarray(bits, dtype=float)
    total = float(const + np.asarray(linear, dtype=float) @ x)
    for (i, j), coeff in quad.items():
        total += float(coeff * x[i] * x[j])
    return float(total)


def qubo_to_ising(const, linear, quad):
    linear_arr = np.asarray(linear, dtype=float)
    z_coeffs = -0.5 * linear_arr.astype(float)
    c0 = float(const + 0.5 * np.sum(linear_arr))
    zz_coeffs = {}

    for (i, j), coeff in quad.items():
        pair_coeff = float(coeff) / 4.0
        c0 += pair_coeff
        z_coeffs[i] -= pair_coeff
        z_coeffs[j] -= pair_coeff
        zz_coeffs[(i, j)] = pair_coeff

    return float(c0), np.asarray(z_coeffs, dtype=float), zz_coeffs


def ising_energy_from_bitstring(bits, c0, z_coeffs, zz_coeffs):
    z_values = [1.0 if int(bit) == 0 else -1.0 for bit in bits]
    energy = float(c0)
    for i, coeff in enumerate(np.asarray(z_coeffs, dtype=float)):
        energy += float(coeff * z_values[i])
    for (i, j), coeff in zz_coeffs.items():
        energy += float(coeff * z_values[i] * z_values[j])
    return float(energy)


def bitstring_cost_table(mu, Sigma, q, B, lam):
    mu_arr, sigma_arr, budget = validate_portfolio_inputs(mu, Sigma, B)
    return {
        bits: penalized_cost(bits, mu_arr, sigma_arr, q, budget, lam)
        for bits in enumerate_bitstrings(len(mu_arr))
    }


def solve_numerical_ising_from_cost_table(cost_table, n_assets):
    basis_terms = [("I",)]
    basis_terms.extend(("Z", i) for i in range(int(n_assets)))
    basis_terms.extend(
        ("ZZ", i, j) for i in range(int(n_assets)) for j in range(i + 1, int(n_assets))
    )

    design = []
    targets = []
    for bits, cost in cost_table.items():
        z_values = [1.0 if bit == 0 else -1.0 for bit in bits]
        row = []
        for term in basis_terms:
            if term[0] == "I":
                row.append(1.0)
            elif term[0] == "Z":
                row.append(z_values[term[1]])
            else:
                row.append(z_values[term[1]] * z_values[term[2]])

        design.append(row)
        targets.append(cost)

    coeffs, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(targets), rcond=None)
    c0 = float(coeffs[0])
    z_coeffs = np.asarray(coeffs[1 : 1 + int(n_assets)], dtype=float)
    zz_coeffs = {}
    idx = 1 + int(n_assets)
    for i in range(int(n_assets)):
        for j in range(i + 1, int(n_assets)):
            zz_coeffs[(i, j)] = float(coeffs[idx])
            idx += 1

    return c0, z_coeffs, zz_coeffs


def compare_analytical_and_numerical_ising(mu, Sigma, q, B, lam):
    const, linear, quad = portfolio_qubo_coeffs(mu, Sigma, q, B, lam)
    analytical = qubo_to_ising(const, linear, quad)
    numerical = solve_numerical_ising_from_cost_table(
        bitstring_cost_table(mu, Sigma, q, B, lam), len(mu)
    )

    c0_a, z_a, zz_a = analytical
    c0_n, z_n, zz_n = numerical
    zz_keys = sorted(set(zz_a) | set(zz_n))

    return {
        "analytical": {"c0": c0_a, "z_coeffs": z_a, "zz_coeffs": zz_a},
        "numerical": {"c0": c0_n, "z_coeffs": z_n, "zz_coeffs": zz_n},
        "max_abs_diff": {
            "c0": abs(c0_a - c0_n),
            "z": float(np.max(np.abs(z_a - z_n))) if len(z_a) else 0.0,
            "zz": max(abs(zz_a.get(key, 0.0) - zz_n.get(key, 0.0)) for key in zz_keys)
            if zz_keys
            else 0.0,
        },
    }


def extract_sampling_metrics(samples, mu, Sigma, q, B, lam, classical_result, top_k=10):
    sample_array = np.asarray(samples, dtype=int)
    if sample_array.ndim == 1:
        sample_array = sample_array.reshape(1, -1)

    total_shots = int(sample_array.shape[0])
    counts = Counter(bitstring_to_tuple(row) for row in sample_array)

    count_records = []
    for bitstring, count in counts.items():
        record = make_portfolio_record(bitstring, mu, Sigma, q, B, lam)
        record["count"] = int(count)
        record["probability"] = float(count / total_shots)
        count_records.append(record)

    count_records.sort(key=lambda record: (-record["count"], record["penalized_objective"]))

    feasible_hits = sum(record["count"] for record in count_records if record["is_feasible"])
    feasible_rate = float(feasible_hits / total_shots)
    infeasible_rate = 1.0 - feasible_rate

    best_sampled = min(count_records, key=lambda record: record["penalized_objective"])
    feasible_count_records = [record for record in count_records if record["is_feasible"]]
    best_feasible_sampled = (
        min(feasible_count_records, key=lambda record: record["penalized_objective"])
        if feasible_count_records
        else None
    )

    best_feasible_bits = classical_result["best_feasible"]["bitstring"]
    best_penalized_bits = classical_result["best_penalized"]["bitstring"]
    best_feasible_prob = float(counts.get(best_feasible_bits, 0) / total_shots)
    best_penalized_prob = float(counts.get(best_penalized_bits, 0) / total_shots)

    return {
        "raw_samples": sample_array,
        "counts": dict(counts),
        "count_records": count_records,
        "best_sampled": best_sampled,
        "best_feasible_sampled": best_feasible_sampled,
        "feasible_sample_rate": feasible_rate,
        "infeasible_sample_rate": infeasible_rate,
        "classical_best_feasible_prob": best_feasible_prob,
        "classical_best_penalized_prob": best_penalized_prob,
        "best_sampled_matches_classical_optimum": (
            best_sampled["bitstring"] == classical_result["best_feasible"]["bitstring"]
        ),
        "top_sampled": count_records[: int(top_k)],
        "total_shots": total_shots,
    }


def summary_record(param_name, param_value, run_result):
    best_feasible_sampled = run_result["samples"]["best_feasible_sampled"]
    best_sampled = run_result["samples"]["best_sampled"]
    classical_best = run_result["classical"]["best_feasible"]

    return {
        "param_name": param_name,
        "param_value": param_value,
        "n_assets": run_result["config"]["n_assets"],
        "q": run_result["config"]["q"],
        "B": run_result["config"]["B"],
        "lam": run_result["config"]["lam"],
        "p": run_result["config"]["p"],
        "final_expected_cost": run_result["optimization"]["final_expected_cost"],
        "best_classical_feasible_objective": classical_best["penalized_objective"],
        "best_sampled_objective": best_sampled["penalized_objective"],
        "best_sampled_bitstring": best_sampled["bitstring_str"],
        "best_feasible_sampled_objective": (
            best_feasible_sampled["penalized_objective"] if best_feasible_sampled else None
        ),
        "best_feasible_sampled_bitstring": (
            best_feasible_sampled["bitstring_str"] if best_feasible_sampled else None
        ),
        "feasible_sample_rate": run_result["samples"]["feasible_sample_rate"],
        "classical_optimum_prob": run_result["samples"]["classical_best_feasible_prob"],
        "runtime_sec": run_result["runtime_sec"],
    }


def instance_kwargs_from_config(config):
    keys = ["return_low", "return_high", "variance_low", "variance_high", "corr_scale"]
    return {key: config[key] for key in keys if key in config}


def apply_budget_rule(config):
    budget_rule = config.get("budget_rule")
    if budget_rule is None:
        return config

    n_assets = int(config["n_assets"])
    if budget_rule == "ceil_half":
        config["B"] = int(np.ceil(n_assets / 2.0))
        return config

    raise ValueError("budget_rule must be None or 'ceil_half'.")


def resolve_instance_for_run(config, param_name, base_instance_seed):
    if param_name != "n_assets" and "mu" in config and "Sigma" in config:
        return np.asarray(config["mu"], dtype=float), np.asarray(config["Sigma"], dtype=float)

    n_assets = int(config["n_assets"])
    if int(config["B"]) > n_assets:
        raise ValueError(f"Budget B={config['B']} exceeds n_assets={n_assets}.")

    if param_name == "n_assets":
        instance_seed = None if base_instance_seed is None else int(base_instance_seed) + n_assets
    else:
        instance_seed = base_instance_seed

    return generate_random_portfolio_instance(
        n_assets=n_assets,
        seed=instance_seed,
        **instance_kwargs_from_config(config),
    )


def build_runner_inputs(config, mu, Sigma, seed):
    return {
        "mu": mu,
        "Sigma": Sigma,
        "q": config["q"],
        "B": config["B"],
        "lam": config["lam"],
        "p": config["p"],
        "shots": config.get("shots", 1000),
        "n_steps": config.get("n_steps", 80),
        "stepsize": config.get("stepsize", 0.1),
        "seed": seed,
        "optimizer_name": config.get("optimizer_name", "adam"),
    }


def sweep_parameter_with_runner(
    runner: Callable,
    base_config,
    param_name,
    values,
    seed=None,
    runner_kwargs_builder: Callable | None = None,
):
    if not values:
        raise ValueError("values must be a non-empty list.")

    config_template = deepcopy(base_config)
    base_instance_seed = config_template.get("instance_seed", seed)
    optimizer_seed = config_template.get("seed", seed)
    shared_instance = None

    if param_name != "n_assets" and "mu" not in config_template and "Sigma" not in config_template:
        shared_instance = resolve_instance_for_run(config_template, param_name, base_instance_seed)

    runs = []
    records = []
    for value in values:
        config = deepcopy(config_template)
        config[param_name] = value
        config = apply_budget_rule(config)

        if shared_instance is not None:
            mu, Sigma = shared_instance
        else:
            mu, Sigma = resolve_instance_for_run(config, param_name, base_instance_seed)

        runner_kwargs = {} if runner_kwargs_builder is None else runner_kwargs_builder(config)
        run_result = runner(**build_runner_inputs(config, mu, Sigma, optimizer_seed), **runner_kwargs)
        runs.append(run_result)
        records.append(summary_record(param_name, value, run_result))

    return {
        "param_name": param_name,
        "values": list(values),
        "records": records,
        "runs": runs,
    }


def aggregate_noisy_runs(runs, extra_fields=None):
    if not runs:
        raise ValueError("runs must contain at least one QAOA result.")

    optimum_probs = [run["samples"]["classical_best_feasible_prob"] for run in runs]
    feasible_rates = [run["samples"]["feasible_sample_rate"] for run in runs]
    final_costs = [run["optimization"]["final_expected_cost"] for run in runs]
    runtimes = [run["runtime_sec"] for run in runs]

    config = runs[0]["config"]
    record = {
        "n_assets": config["n_assets"],
        "B": config["B"],
        "mean_classical_optimum_prob": float(np.mean(optimum_probs)),
        "std_classical_optimum_prob": float(np.std(optimum_probs)),
        "mean_feasible_sample_rate": float(np.mean(feasible_rates)),
        "std_feasible_sample_rate": float(np.std(feasible_rates)),
        "mean_final_expected_cost": float(np.mean(final_costs)),
        "mean_runtime_sec": float(np.mean(runtimes)),
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def build_noisy_instance_catalog(config_template, instance_seeds):
    if "mu" in config_template and "Sigma" in config_template:
        return [
            {
                "instance_seed": config_template.get("instance_seed"),
                "mu": np.asarray(config_template["mu"], dtype=float),
                "Sigma": np.asarray(config_template["Sigma"], dtype=float),
            }
        ]

    if not instance_seeds:
        raise ValueError("instance_seeds must be a non-empty list or tuple.")

    catalog = []
    for instance_seed in instance_seeds:
        mu, Sigma = resolve_instance_for_run(
            config_template,
            param_name="fixed_instance",
            base_instance_seed=instance_seed,
        )
        catalog.append({"instance_seed": instance_seed, "mu": mu, "Sigma": Sigma})

    return catalog


def validate_config_budget(config):
    if int(config["B"]) < 0 or int(config["B"]) > int(config["n_assets"]):
        raise ValueError(
            f"Budget B={config['B']} must satisfy 0 <= B <= n_assets={config['n_assets']}."
        )


def run_config_on_instances_with_runner(
    runner: Callable,
    config,
    instance_catalog,
    runner_kwargs_builder: Callable | None = None,
):
    validate_config_budget(config)
    runs = []

    for instance in instance_catalog:
        instance_seed = instance["instance_seed"]
        runner_kwargs = {} if runner_kwargs_builder is None else runner_kwargs_builder(config)
        run_result = runner(
            **build_runner_inputs(config, instance["mu"], instance["Sigma"], instance_seed),
            **runner_kwargs,
        )
        runs.append(run_result)

    return runs


def sweep_noisy_hyperparam_with_runner(
    runner: Callable,
    base_config,
    param_name,
    values,
    instance_seeds=(0, 1, 2),
    runner_kwargs_builder: Callable | None = None,
):
    if not values:
        raise ValueError("values must be a non-empty list.")

    config_template = deepcopy(base_config)
    config_template = apply_budget_rule(config_template)
    instance_catalog = build_noisy_instance_catalog(config_template, instance_seeds)

    records = []
    runs_by_value = {}

    for value in values:
        config = deepcopy(config_template)
        config[param_name] = value
        config = apply_budget_rule(config)
        value_runs = run_config_on_instances_with_runner(
            runner,
            config,
            instance_catalog,
            runner_kwargs_builder=runner_kwargs_builder,
        )

        runs_by_value[value] = {
            "config": config,
            "instance_seeds": [instance["instance_seed"] for instance in instance_catalog],
            "runs": value_runs,
        }
        records.append(
            aggregate_noisy_runs(
                value_runs,
                extra_fields={"param_name": param_name, "param_value": value},
            )
        )

    return {
        "param_name": param_name,
        "values": list(values),
        "instance_seeds": [instance["instance_seed"] for instance in instance_catalog],
        "base_config": config_template,
        "records": records,
        "runs_by_value": runs_by_value,
    }


def cartesian_hyperparameter_keys():
    return ["p", "lam", "shots", "stepsize", "n_steps"]


def cartesian_configurations(sweep_space):
    keys = cartesian_hyperparameter_keys()
    missing_keys = [key for key in keys if key not in sweep_space]
    if missing_keys:
        missing_display = ", ".join(missing_keys)
        raise ValueError(f"sweep_space is missing required hyperparameters: {missing_display}.")

    values_by_key = []
    for key in keys:
        values = list(sweep_space[key])
        if not values:
            raise ValueError(f"sweep_space['{key}'] must be a non-empty list.")
        values_by_key.append(values)

    return [dict(zip(keys, values)) for values in itertools.product(*values_by_key)]


def noisy_record_sort_key(record):
    return (
        -record["mean_classical_optimum_prob"],
        -record["mean_feasible_sample_rate"],
        record["mean_final_expected_cost"],
        record["mean_runtime_sec"],
        record.get("p", 0),
        record.get("lam", 0.0),
        record.get("shots", 0),
        record.get("stepsize", 0.0),
        record.get("n_steps", 0),
    )


def run_noisy_cartesian_sweep_with_runner(
    runner: Callable,
    base_config,
    sweep_space,
    instance_seeds=(0, 1, 2),
    runner_kwargs_builder: Callable | None = None,
):
    config_template = deepcopy(base_config)
    config_template = apply_budget_rule(config_template)
    instance_catalog = build_noisy_instance_catalog(config_template, instance_seeds)
    configurations = cartesian_configurations(sweep_space)

    records = []
    runs_by_config = {}
    keys = cartesian_hyperparameter_keys()

    for config_values in configurations:
        config = deepcopy(config_template)
        config.update(config_values)
        config = apply_budget_rule(config)

        config_runs = run_config_on_instances_with_runner(
            runner,
            config,
            instance_catalog,
            runner_kwargs_builder=runner_kwargs_builder,
        )
        config_key = " | ".join(f"{key}={config[key]}" for key in keys)
        runs_by_config[config_key] = {
            "config": {key: config[key] for key in keys},
            "instance_seeds": [instance["instance_seed"] for instance in instance_catalog],
            "runs": config_runs,
        }
        records.append(
            aggregate_noisy_runs(
                config_runs,
                extra_fields={key: config[key] for key in keys},
            )
        )

    records.sort(key=noisy_record_sort_key)

    return {
        "base_config": config_template,
        "sweep_space": {key: list(sweep_space[key]) for key in cartesian_hyperparameter_keys()},
        "instance_seeds": [instance["instance_seed"] for instance in instance_catalog],
        "records": records,
        "runs_by_config": runs_by_config,
        "n_configurations": len(configurations),
    }


def sweep_xlabel(param_name):
    labels = {
        "p": "QAOA depth p",
        "lam": "Penalty strength lambda",
        "shots": "Shots",
        "stepsize": "Adam stepsize",
        "n_steps": "Adam n_steps",
        "q": "Risk-aversion q",
        "n_assets": "Number of assets",
        "B": "Budget B",
    }
    return labels.get(param_name, param_name)


def print_noisy_sweep_summary_table(sweep_result):
    records = sweep_result["records"]
    if not records:
        raise ValueError("sweep_result['records'] must contain at least one row.")

    if "param_name" in records[0]:
        columns = [
            "n_assets",
            "B",
            "param_name",
            "param_value",
            "mean_classical_optimum_prob",
            "std_classical_optimum_prob",
            "mean_feasible_sample_rate",
            "std_feasible_sample_rate",
            "mean_final_expected_cost",
            "mean_runtime_sec",
        ]
    else:
        columns = [
            "n_assets",
            "B",
            "p",
            "lam",
            "shots",
            "stepsize",
            "n_steps",
            "mean_classical_optimum_prob",
            "std_classical_optimum_prob",
            "mean_feasible_sample_rate",
            "std_feasible_sample_rate",
            "mean_final_expected_cost",
            "mean_runtime_sec",
        ]

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    widths = {
        column: max(len(column), *(len(format_value(record[column])) for record in records))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    rows = [
        "  ".join(format_value(record[column]).ljust(widths[column]) for column in columns)
        for record in records
    ]
    table = "\n".join([header, divider, *rows])
    print(table)
    return table


def save_noisy_sweep_results_csv(sweep_result, output_dir="noisy_results", prefix="noisy_sweep"):
    records = sweep_result["records"]
    if not records:
        raise ValueError("sweep_result['records'] must contain at least one row.")

    if "param_name" in records[0]:
        columns = [
            "n_assets",
            "B",
            "param_name",
            "param_value",
            "mean_classical_optimum_prob",
            "std_classical_optimum_prob",
            "mean_feasible_sample_rate",
            "std_feasible_sample_rate",
            "mean_final_expected_cost",
            "mean_runtime_sec",
        ]
    else:
        columns = [
            "n_assets",
            "B",
            "p",
            "lam",
            "shots",
            "stepsize",
            "n_steps",
            "mean_classical_optimum_prob",
            "std_classical_optimum_prob",
            "mean_feasible_sample_rate",
            "std_feasible_sample_rate",
            "mean_final_expected_cost",
            "mean_runtime_sec",
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"{prefix}_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in columns})

    return str(csv_path.resolve())


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for CSV post-processing utilities. Install pandas in the active environment first."
        ) from exc
    return pd


def normalized_column_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def search_result_column_candidates():
    return {
        "p": ["p", "depth", "qaoadepth"],
        "lam": ["lam", "lambda", "penalty", "penaltylambda", "penaltystrength"],
        "shots": ["shots", "shotcount"],
        "stepsize": ["stepsize", "learningrate", "lr", "adamstepsize"],
        "n_steps": ["nsteps", "steps", "optimizersteps", "adamnsteps"],
        "classical_optimum_probability": [
            "mean_classical_optimum_prob",
            "classical_optimum_prob",
            "classical optimum probability",
            "classical optimum prob",
        ],
        "feasible_sample_rate": [
            "mean_feasible_sample_rate",
            "feasible_sample_rate",
            "feasible sample rate",
        ],
    }


def resolve_search_result_columns(dataframe, column_mapping=None):
    normalized_lookup = {normalized_column_name(column): column for column in dataframe.columns}
    resolved = {}
    candidates = search_result_column_candidates()

    for canonical_name, canonical_candidates in candidates.items():
        mapped_name = None if column_mapping is None else column_mapping.get(canonical_name)
        candidate_names = []
        if mapped_name is not None:
            candidate_names.append(mapped_name)
        candidate_names.extend(canonical_candidates)

        for candidate in candidate_names:
            if candidate in dataframe.columns:
                resolved[canonical_name] = candidate
                break

            normalized_candidate = normalized_column_name(candidate)
            if normalized_candidate in normalized_lookup:
                resolved[canonical_name] = normalized_lookup[normalized_candidate]
                break

    missing_objectives = [
        name
        for name in ("classical_optimum_probability", "feasible_sample_rate")
        if name not in resolved
    ]
    if missing_objectives:
        missing_display = ", ".join(missing_objectives)
        raise ValueError(
            f"Could not resolve required objective columns for: {missing_display}. "
            "Pass column_mapping to override the automatic detection."
        )

    return resolved


def infer_search_result_dataset_label(dataframe, fallback_name):
    if "n_assets" not in dataframe.columns:
        return str(fallback_name)

    values = [value for value in dataframe["n_assets"].dropna().unique()]
    if len(values) != 1:
        return str(fallback_name)

    value = values[0]
    try:
        n_assets = int(float(value))
        return f"{n_assets} assets"
    except Exception:
        return str(fallback_name)


def load_search_results(csv_path_1, csv_path_2, column_mapping=None):
    pd = import_pandas()
    path_1 = Path(csv_path_1).expanduser().resolve()
    path_2 = Path(csv_path_2).expanduser().resolve()

    dataframe_1 = pd.read_csv(path_1)
    dataframe_2 = pd.read_csv(path_2)
    dataframe_1["source_csv"] = path_1.name
    dataframe_2["source_csv"] = path_2.name
    combined = pd.concat([dataframe_1, dataframe_2], ignore_index=True)
    resolved_columns = resolve_search_result_columns(combined, column_mapping=column_mapping)

    return {
        "dataframe": combined,
        "dataframe_1": dataframe_1,
        "dataframe_2": dataframe_2,
        "datasets": [
            {
                "key": "csv_1",
                "path": str(path_1),
                "filename": path_1.name,
                "label": infer_search_result_dataset_label(dataframe_1, fallback_name=path_1.stem),
                "dataframe": dataframe_1,
            },
            {
                "key": "csv_2",
                "path": str(path_2),
                "filename": path_2.name,
                "label": infer_search_result_dataset_label(dataframe_2, fallback_name=path_2.stem),
                "dataframe": dataframe_2,
            },
        ],
        "paths": [str(path_1), str(path_2)],
        "resolved_columns": resolved_columns,
    }


def search_objective_specs(resolved_columns):
    return [
        {
            "key": "classical_optimum_probability",
            "column": resolved_columns["classical_optimum_probability"],
            "title": "Classical optimum probability",
            "color": "tab:blue",
        },
        {
            "key": "feasible_sample_rate",
            "column": resolved_columns["feasible_sample_rate"],
            "title": "Feasible sample rate",
            "color": "tab:green",
        },
    ]


def available_search_hyperparameters(resolved_columns):
    return [
        hyperparam
        for hyperparam in ["p", "lam", "shots", "stepsize", "n_steps"]
        if hyperparam in resolved_columns
    ]


def sorted_hyperparameter_values(series):
    pd = import_pandas()
    unique_values = [value for value in pd.unique(series.dropna())]
    if not unique_values:
        return []

    try:
        numeric_values = pd.to_numeric(pd.Series(unique_values), errors="raise")
        ordered_pairs = sorted(zip(numeric_values.tolist(), unique_values), key=lambda pair: pair[0])
        return [original for _, original in ordered_pairs]
    except Exception:
        return sorted(unique_values, key=lambda value: str(value))


def format_hyperparameter_tick(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def topk_frequency_summary(dataframe, hyperparam_column, objective_column, top_k, ordered_values):
    topk_frame = dataframe.sort_values(by=objective_column, ascending=False).head(int(top_k))
    counts = topk_frame[hyperparam_column].value_counts(normalize=True, dropna=False)
    return counts.reindex(ordered_values, fill_value=0.0).astype(float)


def score_weighted_summary(dataframe, hyperparam_column, objective_column, ordered_values):
    grouped = dataframe.groupby(hyperparam_column, dropna=False)[objective_column].sum()
    grouped = grouped.reindex(ordered_values, fill_value=0.0).astype(float)
    total_weight = float(grouped.sum())
    if total_weight > 0.0:
        grouped = grouped / total_weight
    else:
        grouped[:] = 0.0
    return grouped


def prepare_postprocess_output_dir(output_dir):
    if output_dir is None:
        return None

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_topk_hyperparam_frequencies(
    csv_path_1,
    csv_path_2,
    top_k,
    output_dir=None,
    column_mapping=None,
):
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")

    pd = import_pandas()
    loaded = load_search_results(csv_path_1, csv_path_2, column_mapping=column_mapping)
    dataframe = loaded["dataframe"]
    resolved_columns = loaded["resolved_columns"]
    output_path = prepare_postprocess_output_dir(output_dir)

    summary_tables = {}
    saved_paths = {}
    figures = {}

    for hyperparam in available_search_hyperparameters(resolved_columns):
        hyperparam_column = resolved_columns[hyperparam]
        ordered_values = sorted_hyperparameter_values(dataframe[hyperparam_column])
        if not ordered_values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        objective_tables = {}

        for axis, objective_spec in zip(axes, search_objective_specs(resolved_columns)):
            distribution = topk_frequency_summary(
                dataframe,
                hyperparam_column=hyperparam_column,
                objective_column=objective_spec["column"],
                top_k=int(top_k),
                ordered_values=ordered_values,
            )
            table = pd.DataFrame(
                {
                    hyperparam: ordered_values,
                    "fraction": distribution.to_numpy(),
                }
            )
            objective_tables[objective_spec["key"]] = table

            axis.bar(
                [format_hyperparameter_tick(value) for value in ordered_values],
                distribution.to_numpy(),
                color=objective_spec["color"],
                alpha=0.85,
            )
            axis.set_title(f"Ranked by {objective_spec['title'].lower()}")
            axis.set_xlabel(hyperparam)
            axis.set_ylabel("Fraction among top-k configs")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Top-k hyperparameter frequencies for {hyperparam}")
        fig.tight_layout()
        if output_path is not None:
            figure_path = output_path / f"topk_freq_{hyperparam}.png"
            fig.savefig(figure_path, bbox_inches="tight")
            saved_paths[hyperparam] = str(figure_path.resolve())

        summary_tables[hyperparam] = objective_tables
        figures[hyperparam] = fig

    return {
        "loaded_results": loaded,
        "top_k": int(top_k),
        "output_dir": None if output_path is None else str(output_path),
        "summary_tables": summary_tables,
        "saved_paths": saved_paths,
        "figures": figures,
    }


def plot_score_weighted_hyperparam_distributions(
    csv_path_1,
    csv_path_2,
    output_dir=None,
    column_mapping=None,
):
    pd = import_pandas()
    loaded = load_search_results(csv_path_1, csv_path_2, column_mapping=column_mapping)
    dataframe = loaded["dataframe"]
    resolved_columns = loaded["resolved_columns"]
    output_path = prepare_postprocess_output_dir(output_dir)

    summary_tables = {}
    saved_paths = {}
    figures = {}

    for hyperparam in available_search_hyperparameters(resolved_columns):
        hyperparam_column = resolved_columns[hyperparam]
        ordered_values = sorted_hyperparameter_values(dataframe[hyperparam_column])
        if not ordered_values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        objective_tables = {}

        for axis, objective_spec in zip(axes, search_objective_specs(resolved_columns)):
            distribution = score_weighted_summary(
                dataframe,
                hyperparam_column=hyperparam_column,
                objective_column=objective_spec["column"],
                ordered_values=ordered_values,
            )
            table = pd.DataFrame(
                {
                    hyperparam: ordered_values,
                    "weighted_fraction": distribution.to_numpy(),
                }
            )
            objective_tables[objective_spec["key"]] = table

            axis.bar(
                [format_hyperparameter_tick(value) for value in ordered_values],
                distribution.to_numpy(),
                color=objective_spec["color"],
                alpha=0.85,
            )
            axis.set_title(f"Weighted by {objective_spec['title'].lower()}")
            axis.set_xlabel(hyperparam)
            axis.set_ylabel("Normalized score-weighted mass")
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Score-weighted hyperparameter distribution for {hyperparam}")
        fig.tight_layout()
        if output_path is not None:
            figure_path = output_path / f"weighted_freq_{hyperparam}.png"
            fig.savefig(figure_path, bbox_inches="tight")
            saved_paths[hyperparam] = str(figure_path.resolve())

        summary_tables[hyperparam] = objective_tables
        figures[hyperparam] = fig

    return {
        "loaded_results": loaded,
        "output_dir": None if output_path is None else str(output_path),
        "summary_tables": summary_tables,
        "saved_paths": saved_paths,
        "figures": figures,
    }


def _grouped_bar_offsets(n_series, width):
    center = 0.5 * (n_series - 1)
    return [(idx - center) * width for idx in range(n_series)]


def _bar_axis_upper_limit(*value_groups):
    maxima = [float(np.max(values)) for values in value_groups if len(values)]
    max_value = max(maxima) if maxima else 0.0
    if max_value <= 0.0:
        return 0.1
    return max(0.1, min(1.15, max_value * 1.18))


def _line_axis_upper_limit(*value_groups):
    maxima = [float(np.max(values)) for values in value_groups if len(values)]
    max_value = max(maxima) if maxima else 0.0
    if max_value <= 0.0:
        return 0.1
    return max(0.15, max_value + 0.16)


def _dataset_line_style(dataset_index):
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    return line_styles[dataset_index % len(line_styles)], markers[dataset_index % len(markers)]


def _line_label_offset(series_index):
    offsets = [8, 18, 28, 38, 48, 58]
    return offsets[series_index] if series_index < len(offsets) else 8 + 10 * series_index


def _annotate_line_series(ax, x_positions, y_values, series_index, color):
    label_offset = _line_label_offset(series_index)
    for x_value, y_value in zip(x_positions, y_values):
        ax.annotate(
            f"{float(y_value):.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, label_offset),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )


def _combined_ordered_hyperparameter_values(datasets, hyperparam_column):
    pd = import_pandas()
    combined_series = pd.concat(
        [dataset["dataframe"][hyperparam_column] for dataset in datasets],
        ignore_index=True,
    )
    return sorted_hyperparameter_values(combined_series)


def _build_combined_postprocess_figures(
    loaded,
    top_k,
    output_dir=None,
):
    pd = import_pandas()
    resolved_columns = loaded["resolved_columns"]
    datasets = loaded["datasets"]
    output_path = prepare_postprocess_output_dir(output_dir)

    weighted_summary_tables = {}
    topk_summary_tables = {}
    figures = {}
    saved_paths = {}

    objective_specs = search_objective_specs(resolved_columns)

    for hyperparam in available_search_hyperparameters(resolved_columns):
        hyperparam_column = resolved_columns[hyperparam]
        ordered_values = _combined_ordered_hyperparameter_values(datasets, hyperparam_column)
        if not ordered_values:
            continue

        x_positions = np.arange(len(ordered_values), dtype=float)
        tick_labels = [format_hyperparameter_tick(value) for value in ordered_values]

        weighted_tables = {}
        topk_tables = {}
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        weighted_ax, topk_ax = axes
        weighted_series_values = []
        topk_series_values = []
        legend_handles = []
        legend_labels = []

        series_index = 0
        for dataset_index, dataset_spec in enumerate(datasets):
            dataset_frame = dataset_spec["dataframe"]
            dataset_label = dataset_spec["label"]
            line_style, marker = _dataset_line_style(dataset_index)
            weighted_tables[dataset_label] = {}
            topk_tables[dataset_label] = {}

            for objective_spec in objective_specs:
                weighted_distribution = score_weighted_summary(
                    dataset_frame,
                    hyperparam_column=hyperparam_column,
                    objective_column=objective_spec["column"],
                    ordered_values=ordered_values,
                )
                topk_distribution = topk_frequency_summary(
                    dataset_frame,
                    hyperparam_column=hyperparam_column,
                    objective_column=objective_spec["column"],
                    top_k=int(top_k),
                    ordered_values=ordered_values,
                )
                weighted_values = weighted_distribution.to_numpy()
                topk_values = topk_distribution.to_numpy()
                weighted_series_values.append(weighted_values)
                topk_series_values.append(topk_values)

                weighted_tables[dataset_label][objective_spec["key"]] = pd.DataFrame(
                    {
                        "dataset": dataset_label,
                        "objective": objective_spec["title"],
                        hyperparam: ordered_values,
                        "weighted_fraction": weighted_values,
                    }
                )
                topk_tables[dataset_label][objective_spec["key"]] = pd.DataFrame(
                    {
                        "dataset": dataset_label,
                        "objective": objective_spec["title"],
                        hyperparam: ordered_values,
                        "fraction": topk_values,
                    }
                )

                line_label = f"{dataset_label}: {objective_spec['title']}"
                weighted_line = weighted_ax.plot(
                    x_positions,
                    weighted_values,
                    color=objective_spec["color"],
                    linestyle=line_style,
                    marker=marker,
                    linewidth=1.8,
                    markersize=6,
                    label=line_label,
                )[0]
                topk_ax.plot(
                    x_positions,
                    topk_values,
                    color=objective_spec["color"],
                    linestyle=line_style,
                    marker=marker,
                    linewidth=1.8,
                    markersize=6,
                    label=line_label,
                )

                _annotate_line_series(
                    weighted_ax,
                    x_positions,
                    weighted_values,
                    series_index=series_index,
                    color=objective_spec["color"],
                )
                _annotate_line_series(
                    topk_ax,
                    x_positions,
                    topk_values,
                    series_index=series_index,
                    color=objective_spec["color"],
                )
                legend_handles.append(weighted_line)
                legend_labels.append(line_label)
                series_index += 1

        weighted_ymax = _line_axis_upper_limit(*weighted_series_values)
        topk_ymax = _line_axis_upper_limit(*topk_series_values)

        weighted_ax.set_title("Weighted")
        weighted_ax.set_xlabel(sweep_xlabel(hyperparam))
        weighted_ax.set_ylabel("Normalized score-weighted mass")
        weighted_ax.set_xticks(x_positions, tick_labels)
        weighted_ax.set_ylim(0.0, weighted_ymax)
        weighted_ax.grid(True, alpha=0.25)

        topk_ax.set_title(f"Top-{int(top_k)}")
        topk_ax.set_xlabel(sweep_xlabel(hyperparam))
        topk_ax.set_ylabel(f"Fraction among top-{int(top_k)} configurations")
        topk_ax.set_xticks(x_positions, tick_labels)
        topk_ax.set_ylim(0.0, topk_ymax)
        topk_ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"Hyperparameter distribution summary for {hyperparam} (3-asset and 4-asset shown separately)",
            y=0.985,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncols=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.82))

        if output_path is not None:
            figure_path = output_path / f"postprocess_{hyperparam}.png"
            fig.savefig(figure_path, bbox_inches="tight")
            saved_paths[hyperparam] = str(figure_path.resolve())

        weighted_summary_tables[hyperparam] = weighted_tables
        topk_summary_tables[hyperparam] = topk_tables
        figures[hyperparam] = fig

    return {
        "loaded_results": loaded,
        "top_k": int(top_k),
        "output_dir": None if output_path is None else str(output_path),
        "weighted_summary_tables": weighted_summary_tables,
        "topk_summary_tables": topk_summary_tables,
        "saved_paths": saved_paths,
        "figures": figures,
    }


def postprocess_saved_search_results(
    csv_path_1,
    csv_path_2,
    top_k,
    output_dir=None,
    column_mapping=None,
):
    loaded = load_search_results(csv_path_1, csv_path_2, column_mapping=column_mapping)
    combined_result = _build_combined_postprocess_figures(
        loaded,
        top_k=top_k,
        output_dir=output_dir,
    )

    return {
        "loaded_results": combined_result["loaded_results"],
        "topk": {
            "summary_tables": combined_result["topk_summary_tables"],
            "top_k": combined_result["top_k"],
        },
        "weighted": {
            "summary_tables": combined_result["weighted_summary_tables"],
        },
        "figures": combined_result["figures"],
        "saved_paths": combined_result["saved_paths"],
        "output_dir": combined_result["output_dir"],
    }


def plot_noisy_optimum_probability(sweep_result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    x_values = [record["param_value"] for record in records]
    y_values = [record["mean_classical_optimum_prob"] for record in records]
    y_errors = [record["std_classical_optimum_prob"] for record in records]
    config = records[0]

    ax.errorbar(x_values, y_values, yerr=y_errors, marker="o", linewidth=1.8, capsize=4)
    ax.set_xlabel(sweep_xlabel(sweep_result["param_name"]))
    ax.set_ylabel("Mean classical optimum probability")
    ax.set_title(f"Noisy optimum probability ({config['n_assets']} assets, B={config['B']})")
    ax.grid(True, alpha=0.3)
    return ax


def plot_noisy_feasible_sample_rate(sweep_result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    x_values = [record["param_value"] for record in records]
    y_values = [record["mean_feasible_sample_rate"] for record in records]
    y_errors = [record["std_feasible_sample_rate"] for record in records]
    config = records[0]

    ax.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        linewidth=1.8,
        capsize=4,
        color="tab:green",
    )
    ax.set_xlabel(sweep_xlabel(sweep_result["param_name"]))
    ax.set_ylabel("Mean feasible sample rate")
    ax.set_title(f"Noisy feasible sample rate ({config['n_assets']} assets, B={config['B']})")
    ax.grid(True, alpha=0.3)
    return ax


def plot_noisy_sweep_combined(sweep_result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    x_values = [record["param_value"] for record in records]
    optimum_means = [record["mean_classical_optimum_prob"] for record in records]
    optimum_stds = [record["std_classical_optimum_prob"] for record in records]
    feasible_means = [record["mean_feasible_sample_rate"] for record in records]
    feasible_stds = [record["std_feasible_sample_rate"] for record in records]
    config = records[0]

    optimum_line = ax.errorbar(
        x_values,
        optimum_means,
        yerr=optimum_stds,
        marker="o",
        linewidth=1.8,
        capsize=4,
        label="Classical optimum probability",
    )
    twin_ax = ax.twinx()
    feasible_line = twin_ax.errorbar(
        x_values,
        feasible_means,
        yerr=feasible_stds,
        marker="s",
        linewidth=1.8,
        capsize=4,
        color="tab:green",
        label="Feasible sample rate",
    )

    ax.set_xlabel(sweep_xlabel(sweep_result["param_name"]))
    ax.set_ylabel("Mean classical optimum probability")
    twin_ax.set_ylabel("Mean feasible sample rate")
    ax.set_title(f"Noisy sweep comparison ({config['n_assets']} assets, B={config['B']})")
    ax.grid(True, alpha=0.3)
    ax.legend(
        [optimum_line[0], feasible_line[0]],
        ["Classical optimum probability", "Feasible sample rate"],
        loc="best",
    )
    return ax, twin_ax


def plot_optimization_convergence(run_result_or_runs, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if isinstance(run_result_or_runs, dict) and "runs" in run_result_or_runs:
        runs = run_result_or_runs["runs"]
    elif isinstance(run_result_or_runs, (list, tuple)):
        runs = run_result_or_runs
    else:
        runs = [run_result_or_runs]

    for run in runs:
        history = run["optimization"]["history"]
        label = f"p={run['config']['p']}"
        ax.plot(range(len(history)), history, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Expected cost")
    ax.set_title("QAOA optimization convergence")
    ax.grid(True, alpha=0.3)
    if len(runs) > 1:
        ax.legend()
    return ax


def plot_bitstring_distribution(run_result, top_k=10, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    records = run_result["samples"]["top_sampled"][: int(top_k)]
    labels = [record["bitstring_str"] for record in records]
    counts = [record["count"] for record in records]
    colors = ["tab:green" if record["is_feasible"] else "tab:red" for record in records]

    bars = ax.bar(labels, counts, color=colors, alpha=0.85)
    ax.set_xlabel("Sampled bitstring")
    ax.set_ylabel("Count")
    ax.set_title("Most common sampled portfolios")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, record in zip(bars, records):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{record['count']}\nobj={record['penalized_objective']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return ax


def plot_feasible_rate_vs_lambda(sweep_result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    lambdas = [record["param_value"] for record in records]
    rates = [record["feasible_sample_rate"] for record in records]

    ax.plot(lambdas, rates, marker="o", linewidth=1.8)
    ax.set_xlabel("Penalty strength lambda")
    ax.set_ylabel("Feasible sample rate")
    ax.set_title("Constraint handling vs lambda")
    ax.grid(True, alpha=0.3)
    return ax


def plot_depth_study(sweep_result, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    depths = [record["param_value"] for record in records]
    best_values = [
        np.nan
        if record["best_feasible_sampled_objective"] is None
        else record["best_feasible_sampled_objective"]
        for record in records
    ]
    optimum_probs = [record["classical_optimum_prob"] for record in records]

    axes[0].plot(depths, best_values, marker="o", linewidth=1.8)
    axes[0].set_xlabel("QAOA depth p")
    axes[0].set_ylabel("Best feasible sampled objective")
    axes[0].set_title("Depth study: sampled objective")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(depths, optimum_probs, marker="o", linewidth=1.8)
    axes[1].set_xlabel("QAOA depth p")
    axes[1].set_ylabel("Probability of classical optimum")
    axes[1].set_title("Depth study: optimum probability")
    axes[1].grid(True, alpha=0.3)

    return axes


def plot_risk_aversion_study(sweep_result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    q_values = [record["param_value"] for record in records]
    objectives = [
        np.nan
        if record["best_feasible_sampled_objective"] is None
        else record["best_feasible_sampled_objective"]
        for record in records
    ]

    ax.plot(q_values, objectives, marker="o", linewidth=1.8)
    for q_value, objective, record in zip(q_values, objectives, records):
        if record["best_feasible_sampled_bitstring"] is not None:
            ax.annotate(
                record["best_feasible_sampled_bitstring"],
                (q_value, objective),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Risk-aversion q")
    ax.set_ylabel("Best feasible sampled objective")
    ax.set_title("Risk-aversion study")
    ax.grid(True, alpha=0.3)
    return ax


def plot_asset_count_study(sweep_result, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))

    records = sorted(sweep_result["records"], key=lambda record: record["param_value"])
    counts = [record["param_value"] for record in records]
    runtimes = [record["runtime_sec"] for record in records]
    feasible_rates = [record["feasible_sample_rate"] for record in records]
    budgets = [record["B"] for record in records]
    budget_note = " (B = ceil(n/2))" if len(set(budgets)) > 1 else f" (B = {budgets[0]})"

    axes[0].plot(counts, runtimes, marker="o", linewidth=1.8)
    for count, runtime, budget in zip(counts, runtimes, budgets):
        axes[0].annotate(
            f"B={budget}",
            (count, runtime),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[0].set_xlabel("Number of assets")
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title(f"Runtime vs asset count{budget_note}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(counts, feasible_rates, marker="o", linewidth=1.8)
    for count, rate, budget in zip(counts, feasible_rates, budgets):
        axes[1].annotate(
            f"B={budget}",
            (count, rate),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Number of assets")
    axes[1].set_ylabel("Feasible sample rate")
    axes[1].set_title(f"Constraint satisfaction vs asset count{budget_note}")
    axes[1].grid(True, alpha=0.3)

    return axes


def plot_risk_return_scatter(classical_result, highlight_bitstring=None, show_frontier=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    feasible_records = classical_result["feasible_records"]
    x = [record["stdev"] for record in feasible_records]
    y = [record["expected_return"] for record in feasible_records]
    ax.scatter(x, y, color="tab:blue", alpha=0.7, label="Feasible portfolios")

    if show_frontier and classical_result["efficient_frontier_records"]:
        frontier = classical_result["efficient_frontier_records"]
        ax.plot(
            [record["stdev"] for record in frontier],
            [record["expected_return"] for record in frontier],
            color="tab:orange",
            linewidth=2.0,
            marker="o",
            label="Efficient frontier",
        )

    target_bits = (
        classical_result["best_feasible"]["bitstring"]
        if highlight_bitstring is None
        else highlight_bitstring
    )
    target_tuple = bitstring_to_tuple(target_bits)
    target_record = next(
        record for record in feasible_records if record["bitstring"] == target_tuple
    )

    for record in feasible_records:
        is_target = record["bitstring"] == target_tuple
        label = f"{record['bitstring_str']}\nobj={record['penalized_objective']:.3f}"
        ax.annotate(
            label,
            (record["stdev"], record["expected_return"]),
            textcoords="offset points",
            xytext=(8, -20 if is_target else -10),
            fontsize=9 if is_target else 8,
            color="tab:green" if is_target else "black",
            fontweight="bold" if is_target else "normal",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

    ax.scatter(
        [target_record["stdev"]],
        [target_record["expected_return"]],
        color="tab:red",
        s=90,
        label=f"Highlighted: {target_record['bitstring_str']}",
    )

    ax.set_xlabel("Portfolio risk (standard deviation)")
    ax.set_ylabel("Expected return")
    ax.set_title("Risk-return view of feasible portfolios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
