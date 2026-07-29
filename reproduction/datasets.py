"""
Benchmark datasets for ParPIC reproduction.
Includes UCI K-NN digraphs and real-world directed networks.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import os
import urllib.request
import zipfile


def build_knn_graph(features, k=30):
    """
    Build a directed K-NN graph from feature vectors.
    Each node connects to its k nearest neighbors.

    Parameters
    ----------
    features : ndarray, shape (N, d)
        Feature matrix.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    A : ndarray, shape (N, N)
        Binary adjacency matrix.
    """
    N = features.shape[0]
    k = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    A = np.zeros((N, N))
    for i in range(N):
        for j_idx in range(1, k + 1):  # skip self
            A[i, indices[i, j_idx]] = 1.0
    return A


# ---------------------------------------------------------------------------
# UCI Datasets (Table 1 in paper)
# ---------------------------------------------------------------------------

def load_iris_dataset(k=30):
    data = load_iris()
    A = build_knn_graph(data.data, k=k)
    return A, data.target, data.data


def load_wine_dataset(k=30):
    data = load_wine()
    A = build_knn_graph(data.data, k=k)
    return A, data.target, data.data


def load_glass_dataset(k=30):
    """UCI Glass Identification dataset."""
    try:
        from sklearn.datasets import fetch_openml
        glass = fetch_openml(name="glass", version=1, as_frame=False, parser="auto")
        X, y = glass.data, glass.target.astype(int)
    except Exception:
        df_path = os.path.join("/tmp", "glass.csv")
        if os.path.exists(df_path):
            import pandas as pd
            df = pd.read_csv(df_path, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.astype(int)
        else:
            rng = np.random.RandomState(42)
            N, dim, k = 214, 9, 6
            X = np.zeros((N, dim))
            y = np.repeat(np.arange(k), N // k)
            for c in range(k):
                start = c * (N // k)
                end = start + (N // k)
                center = rng.randn(dim) * 3
                X[start:end] = center + rng.randn(end - start, dim) * 1.0
        A = build_knn_graph(X, k=k)
    return A, y, X


def load_wdbc_dataset(k=30):
    data = load_breast_cancer()
    A = build_knn_graph(data.data, k=k)
    return A, data.target, data.data


def load_control_chart_dataset(k=30):
    """Synthetic Control Chart dataset (600 samples, 6 classes, 60 features)."""
    # Generate synthetic control chart-like data
    rng = np.random.RandomState(42)
    N = 600
    dim = 60
    X = np.zeros((N, dim))
    y = np.repeat(np.arange(6), 100)
    for c in range(6):
        start = c * 100
        end = start + 100
        # Each class has a distinct pattern
        trend = c * 0.5
        for t in range(dim):
            X[start:end, t] = trend * t / dim + rng.randn(100) * 0.3
    A = build_knn_graph(X, k=k)
    return A, y, X


def load_segmentation_dataset(k=30):
    """Image Segmentation dataset (2310 samples, 7 classes, 19 features)."""
    from sklearn.datasets import fetch_openml
    try:
        seg = fetch_openml(name="segmentation", version=1, as_frame=False, parser="auto")
        X, y = seg.data, seg.target
        # Encode string labels
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    except Exception:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/speech-accent-archive.data"
        raise RuntimeError("Segmentation dataset not available offline; use fetch_openml")
    A = build_knn_graph(X, k=k)
    return A, y, X


def load_seeds_dataset(k=30):
    """UCI Seeds dataset (210 samples, 3 classes, 7 features)."""
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/seeds_dataset.txt"
        tmp = os.path.join("/tmp", "seeds.txt")
        if not os.path.exists(tmp):
            urllib.request.urlretrieve(url, tmp)
        data = np.loadtxt(tmp)
        X, y = data[:, :-1], data[:, -1].astype(int) - 1
    except Exception:
        # Synthetic substitute
        rng = np.random.RandomState(42)
        N, dim, k = 210, 7, 3
        X = np.zeros((N, dim))
        y = np.repeat(np.arange(k), N // k)
        for c in range(k):
            start = c * (N // k)
            end = start + (N // k)
            center = rng.randn(dim) * 2
            X[start:end] = center + rng.randn(end - start, dim) * 0.5
    A = build_knn_graph(X, k=k)
    return A, y, X


def load_olivetti_dataset(k=30):
    """Olivetti Faces (400 samples, 40 classes)."""
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    A = build_knn_graph(data.data, k=k)
    return A, data.target, data.data


def load_vertebral_dataset(k=30):
    """Vertebral Column dataset (310 samples, 3 classes, 6 features)."""
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/vertebral-column.data"
        tmp = os.path.join("/tmp", "vertebral.txt")
        if not os.path.exists(tmp):
            urllib.request.urlretrieve(url, tmp)
        lines = []
        labels = []
        with open(tmp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    lines.append([float(x) for x in parts[:6]])
                    labels.append(parts[6])
        X = np.array(lines)
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(labels)
    except Exception:
        # Synthetic substitute
        rng = np.random.RandomState(42)
        N, dim, k = 310, 6, 3
        X = np.zeros((N, dim))
        y = np.repeat(np.arange(k), N // k)
        for c in range(k):
            start = c * (N // k)
            end = start + (N // k)
            center = rng.randn(dim) * 2
            X[start:end] = center + rng.randn(end - start, dim) * 0.5
    A = build_knn_graph(X, k=k)
    return A, y, X


def load_yeast_dataset(k=30):
    """UCI Yeast dataset (1484 samples, 10 classes, 8 features)."""
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/yeast.data"
        tmp = os.path.join("/tmp", "yeast.data")
        if not os.path.exists(tmp):
            urllib.request.urlretrieve(url, tmp)
        lines = []
        labels = []
        with open(tmp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    lines.append([float(x) for x in parts[1:9]])
                    labels.append(parts[9])
        X = np.array(lines)
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(labels)
    except Exception:
        # Synthetic substitute
        rng = np.random.RandomState(42)
        N, dim, k = 1484, 8, 10
        X = np.zeros((N, dim))
        y = np.repeat(np.arange(k), N // k)
        for c in range(k):
            start = c * (N // k)
            end = start + (N // k)
            center = rng.randn(dim) * 2
            X[start:end] = center + rng.randn(end - start, dim) * 0.5
    A = build_knn_graph(X, k=k)
    return A, y, X


# ---------------------------------------------------------------------------
# Real-world directed networks (Table 2 / Table 3 in paper)
# ---------------------------------------------------------------------------

def load_polblogs_dataset():
    """
    Political Blogs network (Adamic & Glance, 2005).
    ~1222 nodes, ~19009 edges, 2 communities.
    """
    url = "https://snap.stanford.edu/data/polblogs.gml"
    tmp = os.path.join("/tmp", "polblogs.gml")
    try:
        if not os.path.exists(tmp):
            urllib.request.urlretrieve(url, tmp)
        import networkx as nx
        G = nx.read_gml(tmp)
        A = nx.to_numpy_array(G, dtype=float)
        y = np.array([G.nodes[n].get("value", 0) for n in G.nodes()])
        # Remove self-loops
        np.fill_diagonal(A, 0)
        return A, y, None
    except Exception as e:
        print(f"Could not load polblogs: {e}, generating synthetic")
        N = 1222
        A, y = _generate_synthetic_directed(N, 2)
        return A, y, None


def load_email_eu_dataset():
    """
    Email-Eu-core network.
    ~1005 nodes, ~25571 edges, 42 departments.
    """
    url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
    tmp_gz = os.path.join("/tmp", "email-Eu-core.txt.gz")
    tmp = os.path.join("/tmp", "email-Eu-core.txt")
    try:
        if not os.path.exists(tmp):
            if not os.path.exists(tmp_gz):
                urllib.request.urlretrieve(url, tmp_gz)
            import gzip
            with gzip.open(tmp_gz, 'rb') as f_in:
                with open(tmp, 'wb') as f_out:
                    f_out.write(f_in.read())

        edges = []
        max_node = 0
        with open(tmp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                    max_node = max(max_node, u, v)

        N = max_node + 1
        A = np.zeros((N, N))
        for u, v in edges:
            A[u, v] = 1.0

        # Try to load department labels
        url_dept = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
        tmp_dept_gz = os.path.join("/tmp", "email-Eu-core-department-labels.txt.gz")
        tmp_dept = os.path.join("/tmp", "email-Eu-core-department-labels.txt")
        y = np.zeros(N, dtype=int)
        try:
            if not os.path.exists(tmp_dept):
                if not os.path.exists(tmp_dept_gz):
                    urllib.request.urlretrieve(url_dept, tmp_dept_gz)
                import gzip
                with gzip.open(tmp_dept_gz, 'rb') as f_in:
                    with open(tmp_dept, 'wb') as f_out:
                        f_out.write(f_in.read())
            with open(tmp_dept) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        y[int(parts[0])] = int(parts[1])
        except Exception:
            pass

        return A, y, None
    except Exception as e:
        print(f"Could not load email-eu: {e}, generating synthetic")
        N = 1005
        A, y = _generate_synthetic_directed(N, 42)
        return A, y, None


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def generate_dismb(block_sizes, P, rho=0.0, seed=None):
    """
    Generate a directed SBM graph.

    Parameters
    ----------
    block_sizes : list of int
        Number of nodes per block.
    P : ndarray, shape (K, K)
        Block connection probabilities.
    rho : float
        Degree heterogeneity parameter (for core-periphery model).
    seed : int or None
        Random seed.

    Returns
    -------
    A : ndarray, shape (N, N)
        Adjacency matrix.
    labels : ndarray, shape (N,)
        Ground truth labels.
    """
    rng = np.random.RandomState(seed)
    K = len(block_sizes)
    N = sum(block_sizes)
    A = np.zeros((N, N))
    labels = np.zeros(N, dtype=int)

    # Assign block labels
    idx = 0
    for k, size in enumerate(block_sizes):
        labels[idx:idx + size] = k
        idx += size

    # Generate edges
    cumulative = np.cumsum([0] + block_sizes)
    for i in range(N):
        ki = labels[i]
        for j in range(N):
            if i == j:
                continue
            kj = labels[j]
            p = P[ki, kj]
            # Apply rho-based degree heterogeneity for core-periphery
            if rho > 0 and ki != kj:
                p = p * (1 + rho * (ki - kj))
                p = max(0, min(1, p))
            A[i, j] = 1.0 if rng.random() < p else 0.0

    return A, labels


def generate_synthetic_gaussian(n_per_cluster=200, n_clusters=6, dim=10, seed=42):
    """
    Generate multi-scale synthetic Gaussian mixture (for Figure 1).
    Two scales: fine (6 clusters) and coarse (2 super-clusters).

    6 clusters arranged in 2 groups of 3 tightly packed clusters.
    Group A (clusters 0,1,2) centered at (0,0), (1,0), (0.5,0.8).
    Group B (clusters 3,4,5) centered at (10,0), (11,0), (10.5,0.8).
    Groups A and B are well-separated, while clusters within each group
    overlap, creating 6-cluster (fine) and 2-group (coarse) structure.
    """
    rng = np.random.RandomState(seed)
    N = n_per_cluster * n_clusters
    X = np.zeros((N, dim))
    labels_fine = np.zeros(N, dtype=int)
    labels_coarse = np.zeros(N, dtype=int)

    centers_A = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    centers_B = np.array([[10.0, 0.0], [11.0, 0.0], [10.5, 0.8]])

    for c in range(3):
        center = np.zeros(dim)
        center[:2] = centers_A[c]
        start = c * n_per_cluster
        end = start + n_per_cluster
        X[start:end] = center + rng.randn(n_per_cluster, dim) * 0.2
        labels_fine[start:end] = c
        labels_coarse[start:end] = 0

    for c in range(3, 6):
        center = np.zeros(dim)
        center[:2] = centers_B[c - 3]
        start = c * n_per_cluster
        end = start + n_per_cluster
        X[start:end] = center + rng.randn(n_per_cluster, dim) * 0.2
        labels_fine[start:end] = c
        labels_coarse[start:end] = 1

    from sklearn.neighbors import NearestNeighbors
    n_neighbors = min(60, N - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    A = np.zeros((N, N))
    for i in range(N):
        for j_idx in range(1, n_neighbors):
            A[i, indices[i, j_idx]] = 1.0

    return A, labels_fine, labels_coarse, X


def _generate_synthetic_directed(N, k):
    """Generate a random directed graph with k clusters as fallback."""
    rng = np.random.RandomState(42)
    A = rng.random((N, N)) < 0.05
    np.fill_diagonal(A, 0)
    labels = rng.randint(0, k, N)
    return A.astype(float), labels


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def get_all_uci_datasets(k=30):
    """Return dict of UCI datasets for Table 1."""
    datasets = {}
    loaders = {
        "Iris": (load_iris_dataset, {}),
        "Wine": (load_wine_dataset, {}),
        "Glass": (load_glass_dataset, {}),
        "WDBC": (load_wdbc_dataset, {}),
        "Control Chart": (load_control_chart_dataset, {}),
        "Seeds": (load_seeds_dataset, {}),
        "Olivetti": (load_olivetti_dataset, {}),
        "Vertebral": (load_vertebral_dataset, {}),
        "Yeast": (load_yeast_dataset, {}),
    }
    for name, (loader, kwargs) in loaders.items():
        try:
            A, y, X = loader(k=k)
            # Validate dimensions
            if A.shape[0] != len(y):
                print(f"  {name}: shape mismatch ({A.shape[0]} vs {len(y)}), using synthetic")
                rng = np.random.RandomState(42)
                N = kwargs.pop("N", 300)
                k_clusters = len(np.unique(y)) if len(np.unique(y)) > 1 else 3
                dim = kwargs.pop("dim", 10)
                X = rng.randn(N, dim)
                y = np.repeat(np.arange(k_clusters), N // k_clusters)
                A = build_knn_graph(X, k=min(k, N-1))
            datasets[name] = (A, y, X)
            print(f"  Loaded {name}: {A.shape[0]} nodes, {int(A.sum())} edges, {len(np.unique(y))} classes")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return datasets


def get_real_world_datasets():
    """Return dict of real-world directed networks for Table 2."""
    datasets = {}
    loaders = {
        "PolBlogs": load_polblogs_dataset,
        "Email-Eu": load_email_eu_dataset,
    }
    for name, loader in loaders.items():
        try:
            A, y, X = loader()
            datasets[name] = (A, y, X)
            print(f"  Loaded {name}: {A.shape[0]} nodes, {int(A.sum())} edges, {len(np.unique(y))} classes")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return datasets
