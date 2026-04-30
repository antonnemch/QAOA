from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from portfolio_qaoa_core import (
    bitstring_to_tuple,
    bruteforce_portfolio_baseline,
    extract_sampling_metrics,
    penalized_cost,
    portfolio_qubo_coeffs,
    qubo_to_ising,
    validate_portfolio_inputs,
)


def pauli_label(n_qubits, z_positions):
    """Build a Qiskit Pauli label with Z operators at selected qubits."""
    chars = ["I"] * int(n_qubits)
    for qubit in z_positions:
        chars[int(n_qubits) - 1 - int(qubit)] = "Z"
    return "".join(chars)


def build_qiskit_cost_operator(c0, z_coeffs, zz_coeffs, tol=1e-12, include_constant=True):
    """Build a SparsePauliOp cost Hamiltonian from Ising coefficients."""
    z_array = np.asarray(z_coeffs, dtype=float)
    n_qubits = len(z_array)
    terms = []

    if include_constant and abs(c0) >= tol:
        terms.append((pauli_label(n_qubits, ()), float(c0)))

    for i, coeff in enumerate(z_array):
        if abs(coeff) >= tol:
            terms.append((pauli_label(n_qubits, (i,)), float(coeff)))

    for (i, j), coeff in sorted(zz_coeffs.items()):
        if abs(coeff) >= tol:
            terms.append((pauli_label(n_qubits, (i, j)), float(coeff)))

    if not terms:
        terms = [(pauli_label(n_qubits, ()), 0.0)]

    return SparsePauliOp.from_list(terms).simplify()


def strip_identity_from_operator(operator, tol=1e-12):
    """Remove the identity term from a cost operator for circuit evolution."""
    filtered_terms = []
    for label, coeff in operator.to_list():
        if set(label) == {"I"}:
            continue
        if abs(coeff) >= tol:
            filtered_terms.append((label, coeff))

    if not filtered_terms:
        return None
    return SparsePauliOp.from_list(filtered_terms).simplify()


def build_qaoa_ansatz_circuit(cost_operator, n_qubits, p, add_measurements=False):
    """Construct a parameterized Qiskit QAOA ansatz circuit."""
    circuit = QuantumCircuit(int(n_qubits), int(n_qubits) if add_measurements else 0)
    gammas = ParameterVector("gamma", int(p))
    betas = ParameterVector("beta", int(p))
    evolution_operator = strip_identity_from_operator(cost_operator)

    for qubit in range(int(n_qubits)):
        circuit.h(qubit)

    for layer in range(int(p)):
        if evolution_operator is not None:
            circuit.append(PauliEvolutionGate(evolution_operator, time=gammas[layer]), range(int(n_qubits)))
        for qubit in range(int(n_qubits)):
            circuit.rx(2.0 * betas[layer], qubit)

    if add_measurements:
        circuit.measure(range(int(n_qubits)), range(int(n_qubits)))

    return {
        "circuit": circuit,
        "parameter_order": [*gammas, *betas],
        "gammas": list(gammas),
        "betas": list(betas),
    }


def build_bound_measured_qaoa_circuit(cost_operator, n_qubits, p, parameter_values):
    """Bind QAOA parameters and add measurements for sampling."""
    measured_ansatz = build_qaoa_ansatz_circuit(
        cost_operator,
        n_qubits,
        p,
        add_measurements=True,
    )
    bound_measured_circuit = bind_qaoa_parameters(
        measured_ansatz["circuit"],
        measured_ansatz["parameter_order"],
        parameter_values,
    )
    return {
        "ansatz_info": measured_ansatz,
        "bound_circuit": bound_measured_circuit,
    }


def generate_initial_qaoa_parameters(p, seed=None):
    """Generate seeded initial gamma and beta parameters."""
    if int(p) == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, np.pi, size=2 * int(p)).astype(float)


def bind_qaoa_parameters(circuit, parameter_order, values):
    """Bind a flat parameter vector into a QAOA circuit."""
    values = np.asarray(values, dtype=float)
    if len(parameter_order) != len(values):
        raise ValueError(
            f"Expected {len(parameter_order)} parameters, received {len(values)}."
        )

    bindings = {parameter: float(value) for parameter, value in zip(parameter_order, values)}
    return circuit.assign_parameters(bindings, inplace=False)


def expectation_from_statevector(bound_circuit, cost_operator, estimator=None):
    """Estimate the cost expectation for a bound circuit using statevector primitives."""
    estimator = StatevectorEstimator() if estimator is None else estimator
    result = estimator.run([(bound_circuit, cost_operator)]).result()
    value = result[0].data.evs
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0])


def extract_counts_from_pub_result(pub_result):
    """Extract measurement counts from a runtime primitive result."""
    for register_data in pub_result.data.values():
        if hasattr(register_data, "get_counts"):
            try:
                counts = register_data.get_counts()
            except TypeError:
                counts = register_data.get_counts(0)
            return {str(bitstring).replace(" ", ""): int(count) for bitstring, count in counts.items()}
    raise ValueError("Could not extract counts from the sampler result.")


def counts_to_samples(counts, n_qubits):
    """Expand bitstring counts into repeated sample tuples."""
    rows = []
    for bitstring, count in counts.items():
        bit_tuple = tuple(int(bit) for bit in str(bitstring)[::-1])
        if len(bit_tuple) != int(n_qubits):
            raise ValueError(
                f"Expected {n_qubits} bits in sampled string, received {bitstring!r}."
            )
        rows.extend([bit_tuple] * int(count))

    if not rows:
        return np.empty((0, int(n_qubits)), dtype=int)
    return np.asarray(rows, dtype=int)


def normalized_probability_vector(counts, ordered_bitstrings):
    """Return probabilities for ordered bitstrings from raw counts."""
    total_shots = sum(int(count) for count in counts.values())
    if total_shots <= 0:
        return np.zeros(len(ordered_bitstrings), dtype=float)
    return np.asarray(
        [float(counts.get(bitstring, 0)) / float(total_shots) for bitstring in ordered_bitstrings],
        dtype=float,
    )


def sample_counts_statevector(bound_measured_circuit, shots, seed=None):
    """Sample a measured circuit with the local statevector sampler."""
    sampler = StatevectorSampler(seed=seed)
    result = sampler.run([bound_measured_circuit], shots=int(shots)).result()
    return extract_counts_from_pub_result(result[0])


def get_ibm_runtime_service(channel=None, instance=None):
    """Create an IBM Runtime service using optional channel and instance settings."""
    kwargs = {}
    if channel is not None:
        kwargs["channel"] = channel
    if instance is not None:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def resolve_runtime_backend(
    backend=None,
    backend_name=None,
    service=None,
    min_num_qubits=None,
    instance=None,
):
    """Resolve an explicit backend or backend name for runtime execution."""
    if backend is not None:
        return backend
    service = get_ibm_runtime_service(instance=instance) if service is None else service
    if backend_name is not None:
        return service.backend(backend_name)
    return service.least_busy(
        min_num_qubits=min_num_qubits,
        instance=instance,
        simulator=False,
        operational=True,
    )


def select_ibm_backend(service=None, backend_name=None, min_num_qubits=None, instance=None):
    """Select an IBM backend that satisfies the requested qubit count."""
    return resolve_runtime_backend(
        backend=None,
        backend_name=backend_name,
        service=service,
        min_num_qubits=min_num_qubits,
        instance=instance,
    )

def backend_name_or_str(backend):
    """Return a readable backend name."""
    name = getattr(backend, "name", None)
    return name() if callable(name) else name


def count_two_qubit_gates(circuit):
    """Count two-qubit gates in a transpiled circuit."""
    return sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)


def json_safe(value):
    """Convert nested run metadata into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def circuit_resource_summary(circuit):
    """Summarize circuit size, depth, width, and operation counts."""
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "count_ops": dict(circuit.count_ops()),
        "two_qubit_gate_count": int(count_two_qubit_gates(circuit)),
    }


def summarize_transpiled_circuit(circuit, backend):
    """Transpile a circuit and summarize the hardware-level resources."""
    transpiled_circuit = transpile(circuit, backend=backend)
    gate_counts = {str(name): int(count) for name, count in transpiled_circuit.count_ops().items()}
    two_qubit_gate_breakdown = {
        gate_name: int(gate_counts.get(gate_name, 0))
        for gate_name in ("cx", "ecr", "cz")
        if int(gate_counts.get(gate_name, 0)) > 0
    }
    return {
        "backend_name": backend_name_or_str(backend),
        "logical_n_qubits": int(circuit.num_qubits),
        "circuit_width": int(transpiled_circuit.num_qubits),
        "transpiled_depth": int(transpiled_circuit.depth()),
        "transpiled_size": int(transpiled_circuit.size()),
        "total_gate_counts": gate_counts,
        "two_qubit_gate_count": int(count_two_qubit_gates(transpiled_circuit)),
        "two_qubit_gate_breakdown": two_qubit_gate_breakdown,
        "resource_summary": circuit_resource_summary(transpiled_circuit),
        "transpiled_circuit": transpiled_circuit,
    }


def summarize_transpiled_bound_qaoa_circuit(cost_operator, n_qubits, p, parameter_values, backend):
    """Summarize a bound QAOA circuit after backend transpilation."""
    bound_info = build_bound_measured_qaoa_circuit(
        cost_operator=cost_operator,
        n_qubits=n_qubits,
        p=p,
        parameter_values=parameter_values,
    )
    diagnostics = summarize_transpiled_circuit(bound_info["bound_circuit"], backend)
    diagnostics["parameter_values"] = np.asarray(parameter_values, dtype=float).tolist()
    return diagnostics


def timestamped_run_id(prefix="ibm_run"):
    """Create a timestamped identifier for saved run artifacts."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_output_dir(output_dir):
    """Create and return an output directory path."""
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json_artifact(payload, output_dir, filename):
    """Save a JSON artifact and return its path."""
    output_path = ensure_output_dir(output_dir)
    artifact_path = output_path / filename
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    return str(artifact_path.resolve())


def save_figure_artifact(fig, output_dir, filename):
    """Save a Matplotlib figure artifact and return its path."""
    output_path = ensure_output_dir(output_dir)
    artifact_path = output_path / filename
    fig.savefig(artifact_path, bbox_inches="tight")
    return str(artifact_path.resolve())


def summarize_run_for_logging(result, run_id, mode_label=None):
    """Build a compact JSON-friendly summary for a completed run."""
    best_sampled = result["samples"].get("best_sampled")
    best_feasible_sampled = result["samples"].get("best_feasible_sampled")
    return {
        "run_id": run_id,
        "mode": mode_label or result["config"]["execution_mode"],
        "backend": result["hardware"].get("backend_name"),
        "final_expected_cost": result["optimization"].get("final_expected_cost"),
        "best_sampled_bitstring": None if best_sampled is None else best_sampled["bitstring_str"],
        "best_sampled_objective": None
        if best_sampled is None
        else best_sampled["penalized_objective"],
        "best_feasible_sampled_bitstring": None
        if best_feasible_sampled is None
        else best_feasible_sampled["bitstring_str"],
        "best_feasible_sampled_objective": None
        if best_feasible_sampled is None
        else best_feasible_sampled["penalized_objective"],
        "feasible_sample_rate": result["samples"].get("feasible_sample_rate"),
        "classical_optimum_probability": result["samples"].get("classical_best_feasible_prob"),
        "runtime_sec": result.get("runtime_sec"),
    }


def hardware_loop_warnings(n_assets, p, shots, n_steps):
    """Return warnings for expensive or risky hardware-loop settings."""
    warnings = []
    if int(n_assets) > 3:
        warnings.append("n_assets > 3 can be expensive on IBM Open Plan.")
    if int(p) > 2:
        warnings.append("QAOA depth p > 2 may consume Open Plan budget quickly.")
    if int(shots) > 200:
        warnings.append("shots > 200 may be too expensive for repeated hardware objective evaluations.")
    if int(n_steps) > 10:
        warnings.append("n_steps > 10 may be too expensive for hardware-in-loop optimization.")
    return warnings


def estimate_hardware_loop_cost(p, shots, n_steps):
    """Estimate runtime job count and shot usage for a hardware optimization loop."""
    n_params = 2 * int(p)
    if int(p) == 0:
        objective_evaluations = 1
    else:
        objective_evaluations = 1 + 3 * int(n_steps)
    return {
        "n_parameters": int(n_params),
        "estimated_objective_evaluations": int(objective_evaluations),
        "estimated_ibm_jobs": int(objective_evaluations),
        "estimated_total_shots": int(objective_evaluations) * int(shots),
    }


def sample_counts_runtime(
    bound_measured_circuit,
    shots,
    backend=None,
    backend_name=None,
    service=None,
    sampler=None,
    instance=None,
    return_metadata=False,
):
    """Run final circuit sampling through IBM Runtime sampler primitives."""
    runtime_backend = resolve_runtime_backend(
        backend=backend,
        backend_name=backend_name,
        service=service,
        min_num_qubits=bound_measured_circuit.num_qubits,
        instance=instance,
    )
    diagnostics = summarize_transpiled_circuit(bound_measured_circuit, runtime_backend)
    transpiled = diagnostics["transpiled_circuit"]
    runtime_sampler = SamplerV2(mode=runtime_backend) if sampler is None else sampler
    job = runtime_sampler.run([transpiled], shots=int(shots))
    result = job.result()
    counts = extract_counts_from_pub_result(result[0])

    if not return_metadata:
        return counts

    return counts, {
        "backend_name": backend_name_or_str(runtime_backend),
        "job_id": None
        if not hasattr(job, "job_id")
        else (job.job_id() if callable(job.job_id) else job.job_id),
        "transpiled_circuit": transpiled,
        "resource_summary": diagnostics["resource_summary"],
        "diagnostics": diagnostics,
    }


def empirical_average_penalized_cost_from_counts(counts, mu, Sigma, q, B, lam):
    """Compute average penalized cost from measured counts."""
    total_shots = sum(int(count) for count in counts.values())
    if total_shots <= 0:
        raise ValueError("counts must contain at least one sampled bitstring.")

    weighted_total = 0.0
    for bitstring, count in counts.items():
        bits = bitstring_to_tuple(int(bit) for bit in str(bitstring)[::-1])
        weighted_total += float(count) * penalized_cost(bits, mu, Sigma, q, B, lam)
    return float(weighted_total / total_shots)


def sample_fixed_qaoa_circuit(
    cost_operator,
    n_qubits,
    p,
    parameter_values,
    shots,
    sampling_mode="statevector",
    seed=None,
    backend=None,
    backend_name=None,
    runtime_service=None,
    runtime_instance=None,
):
    """Sample fixed QAOA parameters locally or through IBM Runtime."""
    bound_info = build_bound_measured_qaoa_circuit(
        cost_operator=cost_operator,
        n_qubits=n_qubits,
        p=p,
        parameter_values=parameter_values,
    )
    bound_circuit = bound_info["bound_circuit"]
    resolved_sampling_mode = str(sampling_mode).strip().lower()
    if resolved_sampling_mode == "statevector":
        counts = sample_counts_statevector(bound_circuit, shots=shots, seed=seed)
        metadata = {
            "backend_name": "local_statevector",
            "job_id": None,
            "resource_summary": circuit_resource_summary(bound_circuit),
            "diagnostics": None,
            "transpiled_circuit": None,
        }
    elif resolved_sampling_mode == "runtime":
        counts, metadata = sample_counts_runtime(
            bound_circuit,
            shots=shots,
            backend=backend,
            backend_name=backend_name,
            service=runtime_service,
            instance=runtime_instance,
            return_metadata=True,
        )
    else:
        raise ValueError("sampling_mode must be 'statevector' or 'runtime'.")

    return {
        "sampling_mode": resolved_sampling_mode,
        "bound_circuit": bound_circuit,
        "counts": counts,
        "metadata": metadata,
    }


def ordered_bitstrings_from_counts(*counts_dicts):
    """Return a stable bitstring ordering for probability comparison plots."""
    bitstrings = sorted({bitstring for counts in counts_dicts for bitstring in counts.keys()})
    return bitstrings


def plot_local_vs_ibm_probability_comparison(
    local_counts,
    ibm_counts,
    ax=None,
    title="Local vs IBM sampled probabilities",
):
    """Plot local and IBM sampled probabilities for matching bitstrings."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ordered_bitstrings = ordered_bitstrings_from_counts(local_counts, ibm_counts)
    x_positions = np.arange(len(ordered_bitstrings), dtype=float)
    width = 0.38
    local_probs = normalized_probability_vector(local_counts, ordered_bitstrings)
    ibm_probs = normalized_probability_vector(ibm_counts, ordered_bitstrings)

    local_bars = ax.bar(
        x_positions - width / 2.0,
        local_probs,
        width=width,
        label="Local final sampling",
        color="tab:blue",
        alpha=0.85,
    )
    ibm_bars = ax.bar(
        x_positions + width / 2.0,
        ibm_probs,
        width=width,
        label="IBM final sampling",
        color="tab:orange",
        alpha=0.85,
    )

    ax.bar_label(local_bars, labels=[f"{value:.3f}" for value in local_probs], padding=2, fontsize=8)
    ax.bar_label(ibm_bars, labels=[f"{value:.3f}" for value in ibm_probs], padding=2, fontsize=8)
    ax.set_xticks(x_positions, ordered_bitstrings, rotation=45, ha="right")
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Sample probability")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return ax


def build_optimizer_state(optimizer_name, n_params):
    """Initialize state storage for the selected classical optimizer."""
    name = str(optimizer_name).lower()
    if name not in {"adam", "gradient_descent"}:
        raise ValueError("optimizer_name must be 'adam' or 'gradient_descent'.")

    state = {"name": name, "t": 0}
    if name == "adam":
        state["m"] = np.zeros(int(n_params), dtype=float)
        state["v"] = np.zeros(int(n_params), dtype=float)
    return state


def finite_difference_gradient(objective, params, epsilon=1e-4):
    """Estimate a gradient with central finite differences."""
    gradient = np.zeros_like(params, dtype=float)
    for idx in range(len(params)):
        offset = np.zeros_like(params, dtype=float)
        offset[idx] = float(epsilon)
        gradient[idx] = (
            objective(params + offset) - objective(params - offset)
        ) / (2.0 * float(epsilon))
    return gradient


def optimizer_step(params, gradient, stepsize, state):
    """Apply one classical optimizer update step."""
    name = state["name"]
    if name == "gradient_descent":
        return params - float(stepsize) * gradient

    state["t"] += 1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * gradient
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (gradient**2)
    m_hat = state["m"] / (1.0 - beta1 ** state["t"])
    v_hat = state["v"] / (1.0 - beta2 ** state["t"])
    return params - float(stepsize) * m_hat / (np.sqrt(v_hat) + epsilon)


def optimize_qaoa_qiskit(
    cost_operator,
    n_qubits,
    p,
    n_steps,
    stepsize,
    seed=None,
    optimizer_name="adam",
):
    """Optimize QAOA parameters with local statevector expectation values."""
    ansatz_info = build_qaoa_ansatz_circuit(cost_operator, n_qubits, p, add_measurements=False)
    circuit = ansatz_info["circuit"]
    parameter_order = ansatz_info["parameter_order"]
    estimator = StatevectorEstimator()

    objective_evals = 0

    def objective(values):
        nonlocal objective_evals
        bound = bind_qaoa_parameters(circuit, parameter_order, values)
        objective_evals += 1
        return expectation_from_statevector(bound, cost_operator, estimator=estimator)

    if int(p) == 0:
        params = np.asarray([], dtype=float)
        history = [float(objective(params))]
        return {
            "ansatz_info": ansatz_info,
            "initial_params": params.copy(),
            "optimized_params": params.copy(),
            "history": history,
            "final_expected_cost": history[-1],
            "n_objective_evals": objective_evals,
            "iterations_run": 0,
        }

    params = generate_initial_qaoa_parameters(p, seed=seed)
    initial_params = params.copy()
    history = [float(objective(params))]
    optimizer_state = build_optimizer_state(optimizer_name, len(params))

    for _ in range(int(n_steps)):
        gradient = finite_difference_gradient(objective, params)
        params = optimizer_step(params, gradient, stepsize, optimizer_state)
        params = np.mod(params, 2.0 * np.pi)
        history.append(float(objective(params)))

    return {
        "ansatz_info": ansatz_info,
        "initial_params": initial_params,
        "optimized_params": params.copy(),
        "history": history,
        "final_expected_cost": history[-1],
        "n_objective_evals": objective_evals,
        "iterations_run": int(n_steps),
    }


def optimize_qaoa_qiskit_hardware_in_loop(
    cost_operator,
    n_qubits,
    p,
    mu,
    Sigma,
    q,
    B,
    lam,
    shots,
    n_steps,
    stepsize,
    seed=None,
    backend=None,
    backend_name=None,
    runtime_service=None,
    instance=None,
    spsa_perturbation=0.15,
):
    """Optimize QAOA parameters using hardware-sampled objective estimates."""
    ansatz_info = build_qaoa_ansatz_circuit(cost_operator, n_qubits, p, add_measurements=True)
    circuit = ansatz_info["circuit"]
    parameter_order = ansatz_info["parameter_order"]
    rng = np.random.default_rng(seed)
    runtime_backend = resolve_runtime_backend(
        backend=backend,
        backend_name=backend_name,
        service=runtime_service,
        min_num_qubits=n_qubits,
        instance=instance,
    )
    sampler = SamplerV2(mode=runtime_backend)
    objective_evals = 0
    transpilation_reference = None

    def objective(values):
        nonlocal objective_evals, transpilation_reference
        bound_circuit = bind_qaoa_parameters(circuit, parameter_order, values)
        counts, metadata = sample_counts_runtime(
            bound_circuit,
            shots=shots,
            backend=runtime_backend,
            sampler=sampler,
            return_metadata=True,
        )
        if transpilation_reference is None:
            transpilation_reference = metadata
        objective_evals += 1
        return empirical_average_penalized_cost_from_counts(
            counts,
            mu,
            Sigma,
            q,
            B,
            lam,
        )

    if int(p) == 0:
        params = np.asarray([], dtype=float)
        history = [float(objective(params))]
        return {
            "ansatz_info": ansatz_info,
            "initial_params": params.copy(),
            "optimized_params": params.copy(),
            "history": history,
            "final_expected_cost": history[-1],
            "n_objective_evals": objective_evals,
            "iterations_run": 0,
            "backend_name": backend_name_or_str(runtime_backend),
            "transpilation_reference": transpilation_reference,
            "optimizer_method": "hardware_spsa",
        }

    params = generate_initial_qaoa_parameters(p, seed=seed)
    initial_params = params.copy()
    history = [float(objective(params))]

    for step in range(int(n_steps)):
        ck = float(spsa_perturbation) / ((step + 1) ** 0.101)
        ak = float(stepsize) / np.sqrt(step + 1.0)
        delta = rng.choice([-1.0, 1.0], size=len(params))

        plus_params = np.mod(params + ck * delta, 2.0 * np.pi)
        minus_params = np.mod(params - ck * delta, 2.0 * np.pi)
        plus_value = objective(plus_params)
        minus_value = objective(minus_params)

        gradient = ((plus_value - minus_value) / (2.0 * ck)) * delta
        params = np.mod(params - ak * gradient, 2.0 * np.pi)
        history.append(float(objective(params)))

    return {
        "ansatz_info": ansatz_info,
        "initial_params": initial_params,
        "optimized_params": params.copy(),
        "history": history,
        "final_expected_cost": history[-1],
        "n_objective_evals": objective_evals,
        "iterations_run": int(n_steps),
        "backend_name": backend_name_or_str(runtime_backend),
        "transpilation_reference": transpilation_reference,
        "optimizer_method": "hardware_spsa",
    }


def run_qaoa_experiment_qiskit(
    mu,
    Sigma,
    q,
    B,
    lam,
    p,
    shots=1000,
    n_steps=40,
    stepsize=0.1,
    seed=None,
    optimizer_name="adam",
    execution_mode="local_optimize_hardware_sample",
    sampling_mode="statevector",
    final_sampling_mode=None,
    backend=None,
    backend_name=None,
    runtime_service=None,
    runtime_instance=None,
    allow_hardware_loop=False,
    spsa_perturbation=0.15,
):
    """Run the full Qiskit QAOA workflow and return structured results."""
    mu_arr, sigma_arr, budget = validate_portfolio_inputs(mu, Sigma, B)
    n_assets = len(mu_arr)
    execution_mode = str(execution_mode).strip().lower()
    if execution_mode not in {"local_optimize_hardware_sample", "hardware_in_loop"}:
        raise ValueError(
            "execution_mode must be 'local_optimize_hardware_sample' or 'hardware_in_loop'."
        )

    resolved_final_sampling_mode = sampling_mode if final_sampling_mode is None else final_sampling_mode
    resolved_final_sampling_mode = str(resolved_final_sampling_mode).strip().lower()
    if resolved_final_sampling_mode not in {"statevector", "runtime"}:
        raise ValueError("final_sampling_mode must be 'statevector' or 'runtime'.")
    if execution_mode == "hardware_in_loop" and not allow_hardware_loop:
        raise ValueError(
            "Set allow_hardware_loop=True to run the full variational loop on IBM hardware."
        )

    started_at = time.perf_counter()
    classical_result = bruteforce_portfolio_baseline(mu_arr, sigma_arr, q, budget, lam)
    const, linear, quad = portfolio_qubo_coeffs(mu_arr, sigma_arr, q, budget, lam)
    c0, z_coeffs, zz_coeffs = qubo_to_ising(const, linear, quad)
    cost_operator = build_qiskit_cost_operator(c0, z_coeffs, zz_coeffs, include_constant=True)

    runtime_backend = None
    selected_backend_name = None
    using_hardware = execution_mode == "hardware_in_loop" or resolved_final_sampling_mode == "runtime"
    hardware_warnings = hardware_loop_warnings(n_assets, p, shots, n_steps) if using_hardware else []
    if using_hardware:
        runtime_backend = resolve_runtime_backend(
            backend=backend,
            backend_name=backend_name,
            service=runtime_service,
            min_num_qubits=n_assets,
            instance=runtime_instance,
        )
        selected_backend_name = backend_name_or_str(runtime_backend)

    if execution_mode == "hardware_in_loop":
        optimization = optimize_qaoa_qiskit_hardware_in_loop(
            cost_operator=cost_operator,
            n_qubits=n_assets,
            p=int(p),
            mu=mu_arr,
            Sigma=sigma_arr,
            q=q,
            B=budget,
            lam=lam,
            shots=int(shots),
            n_steps=int(n_steps),
            stepsize=float(stepsize),
            seed=seed,
            backend=runtime_backend,
            backend_name=selected_backend_name,
            runtime_service=runtime_service,
            instance=runtime_instance,
            spsa_perturbation=float(spsa_perturbation),
        )
        optimization_mode = "hardware_in_loop"
    else:
        optimization = optimize_qaoa_qiskit(
            cost_operator=cost_operator,
            n_qubits=n_assets,
            p=int(p),
            n_steps=int(n_steps),
            stepsize=float(stepsize),
            seed=seed,
            optimizer_name=optimizer_name,
        )
        optimization_mode = "local_statevector_estimator"

    measured_ansatz = build_qaoa_ansatz_circuit(cost_operator, n_assets, p, add_measurements=True)
    bound_measured_circuit = bind_qaoa_parameters(
        measured_ansatz["circuit"],
        measured_ansatz["parameter_order"],
        optimization["optimized_params"],
    )

    final_sampling_metadata = None
    if resolved_final_sampling_mode == "statevector":
        counts = sample_counts_statevector(bound_measured_circuit, shots=shots, seed=seed)
    else:
        counts, final_sampling_metadata = sample_counts_runtime(
            bound_measured_circuit,
            shots=shots,
            backend=runtime_backend,
            backend_name=selected_backend_name,
            service=runtime_service,
            instance=runtime_instance,
            return_metadata=True,
        )

    raw_samples = counts_to_samples(counts, n_assets)
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
            "execution_mode": execution_mode,
            "sampling_mode": resolved_final_sampling_mode,
            "backend_name": selected_backend_name,
            "runtime_instance": runtime_instance,
            "allow_hardware_loop": bool(allow_hardware_loop),
            "framework": "qiskit",
        },
        "instance": {"mu": mu_arr, "Sigma": sigma_arr},
        "qubo": {"const": const, "linear": linear, "quad": quad},
        "ising": {
            "c0": c0,
            "z_coeffs": z_coeffs,
            "zz_coeffs": zz_coeffs,
            "cost_operator": cost_operator,
        },
        "classical": classical_result,
        "optimization": {
            **optimization,
            "parameter_names": [str(parameter) for parameter in optimization["ansatz_info"]["parameter_order"]],
            "mode": optimization_mode,
        },
        "samples": {
            **sample_metrics,
            "counts_from_backend": counts,
        },
        "qiskit": {
            "ansatz_circuit": optimization["ansatz_info"]["circuit"],
            "measured_circuit": measured_ansatz["circuit"],
        },
        "hardware": {
            "backend_name": selected_backend_name,
            "execution_mode": execution_mode,
            "sampling_mode": resolved_final_sampling_mode,
            "warnings": hardware_warnings,
            "initial_transpilation": optimization.get("transpilation_reference"),
            "final_sampling_transpilation": final_sampling_metadata,
        },
        "runtime_sec": runtime_sec,
    }
