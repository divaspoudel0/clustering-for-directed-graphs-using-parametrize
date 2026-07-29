"""
ParPIC: Parametrized Power-Iteration Clustering for Directed Graphs
====================================================================
Independent re-implementation from scratch, following:
  Debaussart-Joniec et al., "Parametrized Power-Iteration Clustering
  for Directed Graphs", arXiv 2210.00310 (ICML 2026).

Key components:
  1. Vertex measure: degree-based (gamma) and PageRank-based (alpha)
  2. P-RW operator (Definition 3.1)
  3. Entropy-based diffusion time selection (Algorithm 2)
  4. Power iteration embedding + KMeans clustering
"""

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from kneed import KneeLocator


# ---------------------------------------------------------------------------
# Vertex measures (Section 5.1, Eq. 6-7)
# ---------------------------------------------------------------------------

def vertex_measure_degree(A, gamma=0.6):
    """
    Degree-based vertex measure: nu = gamma * d_in + (1 - gamma) * d_out
    with normalized degrees summing to 1.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix (A[i,j] > 0 means edge i -> j).
    gamma : float, in [0, 1]
        Forward/backward flow mixing parameter.
        gamma=1 -> incoming edges dominate; gamma=0 -> outgoing dominate.

    Returns
    -------
    nu : ndarray, shape (N,)
        Vertex measure vector (non-negative, sums to 1).
    """
    d_in = A.sum(axis=0).astype(float)
    d_out = A.sum(axis=1).astype(float)
    d_in_sum, d_out_sum = d_in.sum(), d_out.sum()
    if d_in_sum > 0:
        d_in = d_in / d_in_sum
    if d_out_sum > 0:
        d_out = d_out / d_out_sum
    nu = gamma * d_in + (1 - gamma) * d_out
    nu_sum = nu.sum()
    if nu_sum > 0:
        nu = nu / nu_sum
    return nu


def vertex_measure_pagerank(A, damping=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank vector as a vertex measure component.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix.
    damping : float
        Damping factor (default 0.85).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    pr : ndarray, shape (N,)
        PageRank vector (non-negative, sums to 1).
    """
    N = A.shape[0]
    d_out = A.sum(axis=1).astype(float)
    # Handle dangling nodes
    dangling = np.where(d_out == 0)[0]
    dangling_mass = 0.0
    pr = np.ones(N) / N

    for _ in range(max_iter):
        pr_new = np.zeros(N)
        # Distribute dangling node mass uniformly
        dangling_mass = pr[dangling].sum() if len(dangling) > 0 else 0.0
        pr_new += dangling_mass / N

        for i in range(N):
            if d_out[i] > 0:
                pr_new += damping * pr[i] * A[i, :] / d_out[i]
            else:
                pr_new += damping * pr[i] / N

        # Add teleportation
        pr_new += (1 - damping) / N

        if np.linalg.norm(pr_new - pr, 1) < tol:
            return pr_new / pr_new.sum()
        pr = pr_new

    return pr / pr.sum()


def vertex_measure_combined(A, gamma=0.6, alpha=0.5):
    """
    Combined vertex measure from Section 3.3:
      nu(i) = alpha * c(i) + (1 - alpha) * s(i)
    where:
      c(i) = degree-based measure
      s(i) = PageRank-based measure

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix.
    gamma : float, in [0, 1]
        Forward/backward flow mixing parameter for degree component.
    alpha : float, in [0, 1]
        Balance between degree-based (alpha=1) and PageRank-based (alpha=0).

    Returns
    -------
    nu : ndarray, shape (N,)
        Combined vertex measure vector.
    """
    c = vertex_measure_degree(A, gamma=gamma)
    A_T = A.T
    s = vertex_measure_pagerank(A_T, damping=0.85)
    nu = alpha * c + (1 - alpha) * s
    nu_sum = nu.sum()
    if nu_sum > 0:
        nu = nu / nu_sum
    return nu


# ---------------------------------------------------------------------------
# P-RW operator (Definition 3.1, Eq. 3)
# ---------------------------------------------------------------------------

def compute_prw_operator(A, nu, normalized=True):
    """
    Compute the parametrized random walk operator (Definition 3.1):
      P_(nu) = (D_nu + D_xi)^{-1} (D_nu P + P^T D_nu)
    where:
      D_nu = diag(nu), xi = P^T nu, D_xi = diag(xi), P = D_out^{-1} W

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix.
    nu : ndarray, shape (N,)
        Vertex measure (non-negative).
    normalized : bool
        Whether to row-normalize the result.

    Returns
    -------
    P_nu : ndarray, shape (N, N)
        The parametrized random walk transition matrix.
    """
    N = A.shape[0]
    # Compute natural transition matrix P = D_out^{-1} W
    d_out = A.sum(axis=1).astype(float)
    d_out[d_out == 0] = 1e-12
    D_out_inv = np.diag(1.0 / d_out)
    P = D_out_inv @ A

    # D_nu = diag(nu)
    D_nu = np.diag(nu)
    # xi = P^T nu
    xi = P.T @ nu
    D_xi = np.diag(xi)

    # Build the operator
    numerator = D_nu @ P + P.T @ D_nu
    denominator = D_nu + D_xi

    row_sums = denominator.sum(axis=1)
    row_sums[row_sums == 0] = 1e-12

    if normalized:
        D_denom_inv = np.diag(1.0 / row_sums)
        P_nu = D_denom_inv @ numerator
    else:
        P_nu = numerator

    return P_nu


# ---------------------------------------------------------------------------
# Entropy computation for diffusion time selection (Section 5.2, Algorithm 2)
# ---------------------------------------------------------------------------

def diffusion_entropy(P, max_t=250, n_probes=0):
    """
    Compute diffusion-based entropy trajectory H(t) (Eq. 7).

    Parameters
    ----------
    P : ndarray or sparse matrix, shape (N, N)
        Transition matrix.
    max_t : int
        Maximum diffusion time.
    n_probes : int
        Number of probe nodes for stochastic estimation.
        If 0, exact computation.

    Returns
    -------
    entropies : ndarray, shape (max_t,)
        Entropy values for t = 1, ..., max_t.
    """
    N = P.shape[0]
    if not sp.issparse(P):
        P = sp.csr_matrix(P)

    entropies = []
    if n_probes > 0:
        source_indices = np.random.choice(N, size=min(n_probes, N), replace=False)
        distributions = np.zeros((len(source_indices), N))
        for idx, i in enumerate(source_indices):
            distributions[idx, i] = 1.0

        for _ in range(max_t):
            dist_clipped = np.clip(distributions, 1e-15, 1.0)
            H_estimates = -np.sum(dist_clipped * np.log(dist_clipped), axis=1)
            H_total = np.sum(H_estimates) * (N / n_probes)
            entropies.append(H_total)
            distributions = distributions @ P.T
    else:
        Pt = sp.eye(N, format="csr")
        for _ in range(1, max_t + 1):
            Pt = Pt @ P
            data_clipped = np.clip(Pt.data, 1e-15, 1.0)
            H = -np.sum(Pt.data * np.log(data_clipped))
            entropies.append(H)

    return np.array(entropies)


def get_optimal_diffusion_time(P, t_max=250, n_probes=None):
    """
    Algorithm 2: Automatic diffusion time selection via Calinski-Harabasz-like
    knee detection on the entropy curve.

    Parameters
    ----------
    P : ndarray, shape (N, N)
        The P-RW operator.
    t_max : int
        Maximum diffusion time to consider.
    n_probes : int or None
        Number of probes for stochastic entropy estimation.

    Returns
    -------
    t_opt : int
        Optimal diffusion time.
    entropy_vals : ndarray
        Full entropy trajectory.
    """
    N = P.shape[0]
    if n_probes is None:
        n_probes = max(min(N, 100), int(np.sqrt(N)))

    entropy_vals = diffusion_entropy(P, max_t=t_max, n_probes=n_probes)

    knee_locator = KneeLocator(
        range(1, t_max + 1),
        entropy_vals,
        curve="concave",
        direction="increasing",
    )

    t_opt = knee_locator.knee
    if t_opt is None:
        t_opt = int(np.log(N))

    return t_opt, entropy_vals


# ---------------------------------------------------------------------------
# Power iteration embedding
# ---------------------------------------------------------------------------

def power_iteration_embedding(P_nu, t, projection_type="full", n_cols=None):
    """
    Compute the power iteration embedding Z^(t) approximating P^t_(nu).

    Parameters
    ----------
    P_nu : ndarray, shape (N, N)
        P-RW operator.
    t : int
        Number of iterations (diffusion time).
    projection_type : str
        "full" for exact matrix power, "random" for random projection.
    n_cols : int or None
        Number of columns for random projection.

    Returns
    -------
    Z : ndarray
        Embedding matrix of shape (N, N) for full, or (N, n_cols) for random.
    """
    if projection_type == "full":
        return np.linalg.matrix_power(P_nu, t)
    elif projection_type == "random":
        N = P_nu.shape[0]
        if n_cols is None:
            n_cols = max(min(N, 100), int(np.sqrt(N)))
        Z = np.random.rand(N, n_cols)
        Z = Z / np.linalg.norm(Z, axis=0, keepdims=True)
        for _ in range(t):
            Z = P_nu @ Z
        return Z
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_embedding(embedding, k):
    """
    KMeans clustering of the embedding rows.

    Parameters
    ----------
    embedding : ndarray, shape (N, d)
        Node embedding matrix.
    k : int
        Number of clusters.

    Returns
    -------
    labels : ndarray, shape (N,)
        Cluster labels.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(embedding)
    return km.labels_


# ---------------------------------------------------------------------------
# Full ParPIC pipeline
# ---------------------------------------------------------------------------

def parpic(A, k, gamma=0.6, alpha=0.5, t_max=250, projection_type="full",
           n_cols=None, return_details=False):
    """
    Full ParPIC algorithm.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix of the directed graph.
    k : int
        Number of clusters.
    gamma : float
        Forward/backward flow mixing parameter.
    alpha : float
        Degree vs PageRank balance.
    t_max : int
        Maximum diffusion time for automatic selection.
    projection_type : str
        "full" for exact, "random" for random projection.
    n_cols : int or None
        Dimension for random projection.
    return_details : bool
        If True, also return intermediate results.

    Returns
    -------
    labels : ndarray, shape (N,)
        Predicted cluster labels.
    """
    # Step 1: Compute vertex measure
    nu = vertex_measure_combined(A, gamma=gamma, alpha=alpha)

    # Step 2: Compute P-RW operator
    P_nu = compute_prw_operator(A, nu, normalized=True)

    # Step 3: Select diffusion time automatically
    t_opt, entropy_vals = get_optimal_diffusion_time(P_nu, t_max=t_max)

    # Step 4: Power iteration embedding
    Z = power_iteration_embedding(P_nu, t_opt, projection_type=projection_type,
                                   n_cols=n_cols)

    # Step 5: Cluster
    labels = cluster_embedding(Z, k)

    if return_details:
        return labels, {
            "nu": nu,
            "P_nu": P_nu,
            "t_opt": t_opt,
            "entropy_vals": entropy_vals,
            "Z": Z,
        }
    return labels
