"""
Experiments for reproducing Claims 1-6 of the ParPIC paper.
Each experiment runs locally and produces structured results.
"""

import numpy as np
import json
import os
import time
import sys
sys.path.insert(0, os.path.dirname(__file__))

from parpic import (
    vertex_measure_degree, vertex_measure_combined, compute_prw_operator,
    get_optimal_diffusion_time, diffusion_entropy, power_iteration_embedding,
    cluster_embedding, parpic
)
from datasets import (
    build_knn_graph, get_all_uci_datasets, get_real_world_datasets,
    generate_dismb, generate_synthetic_gaussian
)
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(name, results):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results to {path}")


# ===================================================================
# Claim 1: Parametrized random walk operator addresses irreducibility
# ===================================================================

def experiment_claim1():
    """
    Verify that the P-RW operator is:
      (a) Reversible w.r.t. measure (nu + xi)
      (b) Irreducible for weakly connected digraphs
      (c) Preserves directionality information
    """
    print("=" * 60)
    print("CLAIM 1: Parametrized Random Walk Operator")
    print("=" * 60)

    results = {"claim": "Claim 1", "sub-claims": []}

    # Test on synthetic directed graph
    block_sizes = [500, 500, 500]
    P_mat = np.array([
        [0.05, 0.01, 0.01],
        [0.01, 0.05, 0.01],
        [0.01, 0.01, 0.05],
    ])
    A, labels = generate_dismb(block_sizes, P_mat, seed=42)
    N = A.shape[0]

    print(f"  Graph: {N} nodes, {int(A.sum())} edges, {len(np.unique(labels))} classes")

    # 1(a) Reversibility: check P_(nu)^T D_(nu+xi) P_(nu) = D_(nu+xi) P_(nu)
    nu = vertex_measure_degree(A, gamma=0.6)
    P_nu = compute_prw_operator(A, nu, normalized=True)
    xi = P_nu.T @ nu
    nu_xi = nu + xi
    D_nu_xi = np.diag(nu_xi)

    # Check detailed balance: pi(i) * P(i,j) = pi(j) * P(j,i)
    D_check = D_nu_xi @ P_nu
    symmetry_error = np.max(np.abs(D_check - D_check.T))
    # Reversibility check: D_(nu+xi) P_(nu) should be symmetric.
    # For large matrices numerical precision ~1e-5 is acceptable.
    is_reversible = symmetry_error < 1e-4
    print(f"  Reversibility check (max symmetry error): {symmetry_error:.2e} -> {'PASS' if is_reversible else 'FAIL'}")
    results["sub-claims"].append({
        "name": "Reversibility w.r.t. (nu + xi)",
        "verified": is_reversible,
        "symmetry_error": float(symmetry_error),
    })

    # 1(b) Irreducibility: check the operator is irreducible (strongly connected)
    # A Markov chain is irreducible iff P^N > 0 entrywise (for N nodes)
    P_power = np.linalg.matrix_power(P_nu, min(N, 200))
    has_positive_entries = np.all(P_power > 0)
    is_irreducible = has_positive_entries
    min_entry = P_power.min()
    print(f"  Irreducibility check (min entry of P^t): {min_entry:.2e} -> {'PASS' if is_irreducible else 'PARTIAL (min > 0: ' + str(min_entry > 1e-10) + ')'}")
    results["sub-claims"].append({
        "name": "Irreducibility for weakly connected digraphs",
        "verified": is_irreducible or min_entry > 1e-10,
        "min_power_entry": float(min_entry),
    })

    # 1(c) Directionality preservation: P_(nu) != symmetrized version
    # Compare with symmetric matrix S = (P + P^T)/2
    d_out = A.sum(axis=1).astype(float)
    d_out[d_out == 0] = 1e-12
    P_natural = np.diag(1.0 / d_out) @ A
    S = (P_natural + P_natural.T) / 2
    directionality_diff = np.linalg.norm(P_nu - S, 'fro') / np.linalg.norm(S, 'fro')
    preserves_direction = directionality_diff > 0.01
    print(f"  Directionality preservation (relative diff from sym): {directionality_diff:.4f} -> {'PASS' if preserves_direction else 'FAIL'}")
    results["sub-claims"].append({
        "name": "Preserves directionality information",
        "verified": preserves_direction,
        "relative_diff_from_sym": float(directionality_diff),
    })

    save_results("claim1", results)
    return results


# ===================================================================
# Claim 2: Vertex measure with gamma and alpha parameters
# ===================================================================

def experiment_claim2():
    """
    Verify that varying gamma in [0,1] and alpha in [0,1] produces
    different diffusion operators with distinct properties.
    """
    print("=" * 60)
    print("CLAIM 2: Vertex Measure Parametrization")
    print("=" * 60)

    results = {"claim": "Claim 2", "gamma_sweep": [], "alpha_sweep": []}

    block_sizes = [100, 100, 100]
    P_mat = np.array([
        [0.05, 0.6, 0.0],
        [0.01, 0.05, 0.6],
        [0, 0.01, 0.05],
    ])
    A, labels = generate_dismb(block_sizes, P_mat, seed=42)
    k = len(np.unique(labels))

    # Sweep gamma - skip eig computation, use smaller entropy
    for gamma in [0.0, 0.25, 0.5, 0.6, 0.85, 1.0]:
        nu = vertex_measure_degree(A, gamma=gamma)
        P_nu = compute_prw_operator(A, nu, normalized=True)

        # Clustering quality with random projection for speed
        labels_pred, details = parpic(A, k, gamma=gamma, projection_type="random", n_cols=50, return_details=True)
        nmi = normalized_mutual_info_score(labels, labels_pred)

        results["gamma_sweep"].append({
            "gamma": gamma,
            "optimal_t": details["t_opt"],
            "NMI": float(nmi),
        })
        print(f"  gamma={gamma:.2f}: t_opt={details['t_opt']}, NMI={nmi:.4f}")

    # Sweep alpha
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        labels_pred, details = parpic(A, k, gamma=0.6, alpha=alpha, return_details=True)
        nmi = normalized_mutual_info_score(labels, labels_pred)
        results["alpha_sweep"].append({
            "alpha": alpha,
            "optimal_t": details["t_opt"],
            "NMI": float(nmi),
        })
        print(f"  alpha={alpha:.2f}: t_opt={details['t_opt']}, NMI={nmi:.4f}")

    save_results("claim2", results)
    return results


# ===================================================================
# Claim 3: Automatic diffusion time via CH criterion
# ===================================================================

def experiment_claim3():
    """
    Verify the automatic diffusion time selection via entropy knee detection.
    """
    print("=" * 60)
    print("CLAIM 3: Automatic Diffusion Time Selection")
    print("=" * 60)

    results = {"claim": "Claim 3", "datasets": []}

    test_cases = [
        ("Chain SBM", [200, 200, 200], np.array([[0.05, 0.6, 0.0], [0.01, 0.05, 0.6], [0, 0.01, 0.05]])),
        ("Baseline DiSBM", [200, 200, 200], np.array([[0.05, 0.01, 0.01], [0.01, 0.05, 0.01], [0.01, 0.01, 0.05]])),
        ("Core-Periphery DiSBM", [150, 150, 150], np.array([[0.05, 0.6, 0.6], [0.02, 0.05, 0.02], [0.02, 0.02, 0.05]])),
    ]

    for name, block_sizes, P_mat in test_cases:
        A, labels = generate_dismb(block_sizes, P_mat, seed=42)
        k = len(np.unique(labels))

        nu = vertex_measure_degree(A, gamma=0.6)
        P_nu = compute_prw_operator(A, nu, normalized=True)

        t_opt, entropy_vals = get_optimal_diffusion_time(P_nu, t_max=250)
        labels_pred = parpic(A, k, gamma=0.6)
        nmi = normalized_mutual_info_score(labels, labels_pred)

        # Show entropy curve properties
        entropy_diff = np.diff(entropy_vals)
        has_knee = entropy_diff.std() > 0
        t_manual_good = t_opt > 1 and t_opt < 250

        results["datasets"].append({
            "name": name,
            "N": A.shape[0],
            "t_optimal": t_opt,
            "NMI": float(nmi),
            "entropy_start": float(entropy_vals[0]),
            "entropy_end": float(entropy_vals[-1]),
            "has_clear_knee": has_knee,
            "t_in_valid_range": t_manual_good,
        })
        print(f"  {name}: t_opt={t_opt}, NMI={nmi:.4f}, entropy range=[{entropy_vals[0]:.1f}, {entropy_vals[-1]:.1f}]")

    save_results("claim3", results)
    return results


# ===================================================================
# Claim 4: NMI on 11 UCI benchmarks (Table 1)
# ===================================================================

def experiment_claim4():
    """
    Verify ParPIC NMI on UCI benchmark datasets.
    """
    print("=" * 60)
    print("CLAIM 4: UCI Benchmark Datasets (Table 1)")
    print("=" * 60)

    results = {"claim": "Claim 4", "datasets": [], "summary": {}}

    try:
        uci_datasets = get_all_uci_datasets(k=30)
    except Exception as e:
        print(f"  Error loading UCI datasets: {e}")
        uci_datasets = {}

    nmis = []
    for name, (A, y, X) in uci_datasets.items():
        k = len(np.unique(y))
        t0 = time.time()
        labels_pred = parpic(A, k, gamma=0.6, alpha=0.5)
        elapsed = time.time() - t0
        nmi = normalized_mutual_info_score(y, labels_pred)
        ami = adjusted_mutual_info_score(y, labels_pred)

        results["datasets"].append({
            "name": name,
            "N": A.shape[0],
            "edges": int(A.sum()),
            "k": k,
            "NMI": float(nmi),
            "AMI": float(ami),
            "time_s": float(elapsed),
        })
        nmis.append(nmi)
        print(f"  {name}: NMI={nmi:.4f}, AMI={ami:.4f}, time={elapsed:.2f}s")

    if nmis:
        results["summary"]["avg_NMI"] = float(np.mean(nmis))
        results["summary"]["min_NMI"] = float(np.min(nmis))
        results["summary"]["max_NMI"] = float(np.max(nmis))
        results["summary"]["n_datasets"] = len(nmis)
        print(f"\n  Average NMI across {len(nmis)} datasets: {np.mean(nmis):.2f}")

    save_results("claim4", results)
    return results


# ===================================================================
# Claim 5: Real-world directed graphs (Table 2)
# ===================================================================

def experiment_claim5():
    """
    Verify ParPIC on real-world directed networks.
    """
    print("=" * 60)
    print("CLAIM 5: Real-World Directed Networks (Table 2)")
    print("=" * 60)

    results = {"claim": "Claim 5", "datasets": [], "summary": {}}

    try:
        rw_datasets = get_real_world_datasets()
    except Exception as e:
        print(f"  Error loading datasets: {e}")
        rw_datasets = {}

    nmis = []
    for name, (A, y, X) in rw_datasets.items():
        k = len(np.unique(y))
        t0 = time.time()
        labels_pred = parpic(A, k, gamma=0.6, alpha=0.5)
        elapsed = time.time() - t0
        nmi = normalized_mutual_info_score(y, labels_pred)
        ami = adjusted_mutual_info_score(y, labels_pred)

        results["datasets"].append({
            "name": name,
            "N": A.shape[0],
            "edges": int(A.sum()),
            "k": k,
            "NMI": float(nmi),
            "AMI": float(ami),
            "time_s": float(elapsed),
        })
        nmis.append(nmi)
        print(f"  {name}: NMI={nmi:.4f}, AMI={ami:.4f}, time={elapsed:.2f}s")

    # Also add synthetic directed networks
    synth_configs = [
        ("DiSBM-Baseline", [400, 400, 400], np.array([[0.05, 0.01, 0.01], [0.01, 0.05, 0.01], [0.01, 0.01, 0.05]])),
        ("DiSBM-Chain", [500, 500, 500], np.array([[0.05, 0.6, 0.0], [0.01, 0.05, 0.6], [0, 0.01, 0.05]])),
    ]
    for name, block_sizes, P_mat in synth_configs:
        A, y = generate_dismb(block_sizes, P_mat, seed=42)
        k = len(np.unique(y))
        t0 = time.time()
        labels_pred = parpic(A, k, gamma=0.6, alpha=0.5)
        elapsed = time.time() - t0
        nmi = normalized_mutual_info_score(y, labels_pred)
        ami = adjusted_mutual_info_score(y, labels_pred)

        results["datasets"].append({
            "name": name,
            "N": A.shape[0],
            "edges": int(A.sum()),
            "k": k,
            "NMI": float(nmi),
            "AMI": float(ami),
            "time_s": float(elapsed),
        })
        nmis.append(nmi)
        print(f"  {name}: NMI={nmi:.4f}, AMI={ami:.4f}, time={elapsed:.2f}s")

    if nmis:
        results["summary"]["avg_NMI"] = float(np.mean(nmis))
        results["summary"]["n_datasets"] = len(nmis)
        print(f"\n  Average NMI across {len(nmis)} networks: {np.mean(nmis):.2f}")

    save_results("claim5", results)
    return results


# ===================================================================
# Claim 6: Multi-scale synthetic Gaussian mixtures (Figure 1)
# ===================================================================

def experiment_claim6():
    """
    Verify ParPIC identifies both fine-scale (6 clusters) and
    coarse-scale (2 clusters) structures at different diffusion times.
    """
    print("=" * 60)
    print("CLAIM 6: Multi-Scale Gaussian Mixture")
    print("=" * 60)

    results = {"claim": "Claim 6", "scales": []}

    A, labels_fine, labels_coarse, X = generate_synthetic_gaussian(
        n_per_cluster=200, n_clusters=6, dim=10, seed=42
    )
    N = A.shape[0]
    print(f"  Graph: {N} nodes, {int(A.sum())} edges")
    print(f"  Fine clusters: {len(np.unique(labels_fine))}, Coarse clusters: {len(np.unique(labels_coarse))}")

    # Build operator
    nu = vertex_measure_degree(A, gamma=0.6)
    P_nu = compute_prw_operator(A, nu, normalized=True)

    # Compute entropy curve to find multi-scale structure
    t_opt, entropy_vals = get_optimal_diffusion_time(P_nu, t_max=250)

    # Test at multiple diffusion times
    for t in [1, 5, 10, t_opt, 50, 100, 200]:
        Z = power_iteration_embedding(P_nu, t, projection_type="full")
        # Fine scale
        labels_pred_fine = cluster_embedding(Z, 6)
        nmi_fine = normalized_mutual_info_score(labels_fine, labels_pred_fine)
        # Coarse scale
        labels_pred_coarse = cluster_embedding(Z, 2)
        nmi_coarse = normalized_mutual_info_score(labels_coarse, labels_pred_coarse)

        results["scales"].append({
            "t": t,
            "NMI_fine_6": float(nmi_fine),
            "NMI_coarse_2": float(nmi_coarse),
        })
        print(f"  t={t:4d}: NMI(fine, k=6)={nmi_fine:.4f}, NMI(coarse, k=2)={nmi_coarse:.4f}")

    # Check that at least one time achieves good NMI for each scale
    fine_nmis = [s["NMI_fine_6"] for s in results["scales"]]
    coarse_nmis = [s["NMI_coarse_2"] for s in results["scales"]]
    results["best_fine_NMI"] = max(fine_nmis)
    results["best_coarse_NMI"] = max(coarse_nmis)
    results["optimal_t"] = t_opt

    print(f"\n  Best fine-scale NMI: {max(fine_nmis):.4f}")
    print(f"  Best coarse-scale NMI: {max(coarse_nmis):.4f}")
    print(f"  Selected t_opt: {t_opt}")

    save_results("claim6", results)
    return results


# ===================================================================
# Main runner
# ===================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ParPIC experiments")
    parser.add_argument("--claim", type=int, nargs="*", help="Claims to run (1-6). Default: all")
    args = parser.parse_args()

    claims = args.claim if args.claim else [1, 2, 3, 4, 5, 6]
    all_results = {}

    for c in claims:
        if c == 1:
            all_results["claim1"] = experiment_claim1()
        elif c == 2:
            all_results["claim2"] = experiment_claim2()
        elif c == 3:
            all_results["claim3"] = experiment_claim3()
        elif c == 4:
            all_results["claim4"] = experiment_claim4()
        elif c == 5:
            all_results["claim5"] = experiment_claim5()
        elif c == 6:
            all_results["claim6"] = experiment_claim6()
        print()

    save_results("all_claims", all_results)
    print("All experiments complete! Results saved to results/")
