from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

from portfolio_qaoa_core import (
    bruteforce_portfolio_baseline,
    extract_sampling_metrics,
    portfolio_qubo_coeffs,
    qubo_to_ising,
    run_noisy_cartesian_sweep_with_runner,
    sweep_noisy_hyperparam_with_runner,
    sweep_parameter_with_runner,
    validate_portfolio_inputs,
)


def build_cost_hamiltonian(c0, z_coeffs, zz_coeffs, tol=1e-12):
    """Build a PennyLane Hamiltonian from Ising coefficients."""
    coeffs = []
    ops = []

    if abs(c0) >= tol:
        coeffs.append(float(c0))
        ops.append(qml.Identity(0))

    for i, coeff in enumerate(np.asarray(z_coeffs, dtype=float)):
        if abs(coeff) >= tol:
            coeffs.append(float(coeff))
            ops.append(qml.PauliZ(i))

    for (i, j), coeff in sorted(zz_coeffs.items()):
        if abs(coeff) >= tol:
            coeffs.append(float(coeff))
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if not coeffs:
        coeffs = [0.0]
        ops = [qml.Identity(0)]

    return qml.Hamiltonian(coeffs, ops)


def make_x_mixer(n_wires):
    """Build the standard X mixer Hamiltonian for QAOA."""
    return qml.qaoa.x_mixer(wires=range(int(n_wires)))


def normalize_noise_mode(noise_mode):
    """Normalize user-facing noise-mode values."""
    if isinstance(noise_mode, str):
        normalized = noise_mode.strip().lower()
    elif noise_mode is True:
        normalized = "layer_depolarizing"
    elif noise_mode in (None, False):
        normalized = "ideal"
    else:
        normalized = str(noise_mode).strip().lower()

    if normalized in {"ideal", "false", "none", "0"}:
        return "ideal"
    if normalized in {"layer_depolarizing", "depolarizing", "noisy", "true", "1"}:
        return "layer_depolarizing"

    raise ValueError("noise_mode must be False/'ideal' or True/'layer_depolarizing'.")


def apply_layer_noise(noise_mode, noise_strength, n_wires):
    """Apply per-wire depolarizing noise for one QAOA layer."""
    normalized_mode = normalize_noise_mode(noise_mode)
    probability = float(noise_strength)

    if normalized_mode == "ideal" or probability <= 0.0:
        return
    if probability > 1.0:
        raise ValueError("noise_strength must lie in [0, 1] for depolarizing noise.")

    for wire in range(int(n_wires)):
        qml.DepolarizingChannel(probability, wires=wire)


def apply_qaoa_ansatz(
    params,
    p,
    cost_h,
    mixer_h,
    n_wires,
    noise_mode="ideal",
    noise_strength=0.0,
):
    """Apply the alternating QAOA cost and mixer layers."""
    if params.shape[0] != 2 * int(p):
        raise ValueError(f"Expected {2 * int(p)} parameters for p={p}, got {params.shape[0]}.")

    for wire in range(int(n_wires)):
        qml.Hadamard(wires=wire)

    gammas = params[:p]
    betas = params[p:]
    for layer in range(int(p)):
        qml.qaoa.cost_layer(gammas[layer], cost_h)
        apply_layer_noise(noise_mode, noise_strength, n_wires)
        qml.qaoa.mixer_layer(betas[layer], mixer_h)
        apply_layer_noise(noise_mode, noise_strength, n_wires)


def create_qaoa_qnodes(
    cost_h,
    mixer_h,
    n_wires,
    shots,
    noise_mode="ideal",
    noise_strength=0.0,
    device_seed=None,
):
    """Create optimization and sampling QNodes for one QAOA problem."""
    normalized_noise_mode = normalize_noise_mode(noise_mode)
    device_name = (
        "default.mixed" if normalized_noise_mode == "layer_depolarizing" else "default.qubit"
    )
    device_kwargs = {}
    if device_seed is not None:
        device_kwargs["seed"] = int(device_seed)

    dev_expval = qml.device(device_name, wires=int(n_wires), **device_kwargs)
    dev_sample = qml.device(device_name, wires=int(n_wires), **device_kwargs)

    @qml.qnode(dev_expval)
    def expectation_qnode(params, p):
        apply_qaoa_ansatz(
            params,
            p,
            cost_h,
            mixer_h,
            n_wires,
            noise_mode=normalized_noise_mode,
            noise_strength=noise_strength,
        )
        return qml.expval(cost_h)

    @qml.qnode(dev_sample)
    def sample_qnode_base(params, p):
        apply_qaoa_ansatz(
            params,
            p,
            cost_h,
            mixer_h,
            n_wires,
            noise_mode=normalized_noise_mode,
            noise_strength=noise_strength,
        )
        return qml.sample(wires=range(int(n_wires)))

    sample_qnode = qml.set_shots(sample_qnode_base, shots=int(shots))
    return expectation_qnode, sample_qnode


def build_optimizer(optimizer_name, stepsize):
    """Create the requested PennyLane optimizer."""
    name = str(optimizer_name).lower()
    if name == "adam":
        return qml.AdamOptimizer(stepsize=stepsize)
    if name == "gradient_descent":
        return qml.GradientDescentOptimizer(stepsize=stepsize)
    raise ValueError("optimizer_name must be 'adam' or 'gradient_descent'.")


def optimize_qaoa(
    cost_h,
    mixer_h,
    n_wires,
    p,
    n_steps,
    stepsize,
    seed=None,
    optimizer_name="adam",
    noise_mode="ideal",
    noise_strength=0.0,
    device_seed=None,
):
    """Optimize QAOA parameters on a PennyLane simulator."""
    expectation_qnode, _ = create_qaoa_qnodes(
        cost_h,
        mixer_h,
        n_wires,
        shots=1,
        noise_mode=noise_mode,
        noise_strength=noise_strength,
        device_seed=device_seed,
    )

    if int(p) == 0:
        params = qnp.array([], requires_grad=True)
        history = [float(expectation_qnode(params, 0))]
        return {
            "initial_params": np.asarray(params, dtype=float),
            "optimized_params": np.asarray(params, dtype=float),
            "history": history,
            "final_expected_cost": history[-1],
            "n_objective_evals": len(history),
            "iterations_run": 0,
        }

    rng = np.random.default_rng(seed)
    initial_params = rng.uniform(0.0, np.pi, size=2 * int(p))
    params = qnp.array(initial_params, requires_grad=True)
    optimizer = build_optimizer(optimizer_name, stepsize)
    objective = lambda values: expectation_qnode(values, int(p))

    history = [float(objective(params))]
    for _ in range(int(n_steps)):
        params = optimizer.step(objective, params)
        history.append(float(objective(params)))

    optimized_params = np.asarray(params, dtype=float)
    return {
        "initial_params": np.asarray(initial_params, dtype=float),
        "optimized_params": optimized_params,
        "history": history,
        "final_expected_cost": history[-1],
        "n_objective_evals": len(history),
        "iterations_run": int(n_steps),
    }


def run_qaoa_experiment(
    mu,
    Sigma,
    q,
    B,
    lam,
    p,
    shots=1000,
    n_steps=80,
    stepsize=0.1,
    seed=None,
    optimizer_name="adam",
    noise_mode="ideal",
    noise_strength=0.0,
    device_seed=None,
):
    """Run the full PennyLane QAOA workflow and return structured results."""
    mu_arr, sigma_arr, budget = validate_portfolio_inputs(mu, Sigma, B)
    n_assets = len(mu_arr)
    normalized_noise_mode = normalize_noise_mode(noise_mode)
    resolved_device_seed = seed if device_seed is None else device_seed

    started_at = time.perf_counter()
    classical_result = bruteforce_portfolio_baseline(mu_arr, sigma_arr, q, budget, lam)
    const, linear, quad = portfolio_qubo_coeffs(mu_arr, sigma_arr, q, budget, lam)
    c0, z_coeffs, zz_coeffs = qubo_to_ising(const, linear, quad)
    cost_h = build_cost_hamiltonian(c0, z_coeffs, zz_coeffs)
    mixer_h = make_x_mixer(n_assets)

    optimization = optimize_qaoa(
        cost_h=cost_h,
        mixer_h=mixer_h,
        n_wires=n_assets,
        p=int(p),
        n_steps=int(n_steps),
        stepsize=float(stepsize),
        seed=seed,
        optimizer_name=optimizer_name,
        noise_mode=normalized_noise_mode,
        noise_strength=float(noise_strength),
        device_seed=resolved_device_seed,
    )

    _, sample_qnode = create_qaoa_qnodes(
        cost_h,
        mixer_h,
        n_assets,
        shots=shots,
        noise_mode=normalized_noise_mode,
        noise_strength=float(noise_strength),
        device_seed=resolved_device_seed,
    )
    sample_params = qnp.array(optimization["optimized_params"], requires_grad=False)
    raw_samples = sample_qnode(sample_params, int(p))
    sample_metrics = extract_sampling_metrics(
        raw_samples,
        mu_arr,
        sigma_arr,
        q,
        budget,
        lam,
        classical_result,
    )
    runtime_sec = float(time.perf_counter() - started_at)

    return {
        "config": {
            "n_assets": n_assets,
            "q": float(q),
            "B": budget,
            "lam": float(lam),
            "p": int(p),
            "shots": int(shots),
            "n_steps": int(n_steps),
            "stepsize": float(stepsize),
            "seed": seed,
            "optimizer_name": optimizer_name,
            "noise_mode": normalized_noise_mode,
            "noise_strength": float(noise_strength),
            "device_seed": resolved_device_seed,
            "framework": "pennylane",
        },
        "instance": {"mu": mu_arr, "Sigma": sigma_arr},
        "qubo": {"const": const, "linear": linear, "quad": quad},
        "ising": {
            "c0": c0,
            "z_coeffs": z_coeffs,
            "zz_coeffs": zz_coeffs,
            "cost_hamiltonian": cost_h,
        },
        "classical": classical_result,
        "optimization": optimization,
        "samples": sample_metrics,
        "runtime_sec": runtime_sec,
    }


def pennylane_runner_kwargs(config):
    """Extract PennyLane runner keyword arguments from a config dictionary."""
    return {
        "noise_mode": config.get("noise_mode", "ideal"),
        "noise_strength": config.get("noise_strength", 0.0),
        "device_seed": config.get("device_seed"),
    }


def sweep_parameter(base_config, param_name, values, seed=None):
    """Run a one-parameter PennyLane sweep."""
    return sweep_parameter_with_runner(
        run_qaoa_experiment,
        base_config,
        param_name,
        values,
        seed=seed,
        runner_kwargs_builder=pennylane_runner_kwargs,
    )


def sweep_noisy_hyperparam(base_config, param_name, values, instance_seeds=(0, 1, 2)):
    """Run a noisy PennyLane hyperparameter sweep across repeated instances."""
    return sweep_noisy_hyperparam_with_runner(
        run_qaoa_experiment,
        base_config,
        param_name,
        values,
        instance_seeds=instance_seeds,
        runner_kwargs_builder=pennylane_runner_kwargs,
    )


def run_noisy_cartesian_sweep(base_config, sweep_space, instance_seeds=(0, 1, 2)):
    """Run a Cartesian noisy hyperparameter search with the PennyLane runner."""
    return run_noisy_cartesian_sweep_with_runner(
        run_qaoa_experiment,
        base_config,
        sweep_space,
        instance_seeds=instance_seeds,
        runner_kwargs_builder=pennylane_runner_kwargs,
    )
