# Reproducing: Parametrized Power-Iteration Clustering for Directed Graphs

**Paper:** Debaussart-Joniec et al., ICML 2026  
**arXiv:** [2210.00310](https://arxiv.org/abs/2210.00310)  
**OpenReview:** [5vI6ApLOg8](https://openreview.net/forum?id=5vI6ApLOg8)  
**Original code:** [Gwendal-Debaussart/parpic](https://github.com/Gwendal-Debaussart/parpic)  
**Reproduction repo:** [divaspoudel0/clustering-for-directed-graphs-using-parametrize](https://github.com/divaspoudel0/clustering-for-directed-graphs-using-parametrize)

## Overview

This repository contains an independent reproduction of the paper's main claims. The paper proposes **Parametrized Power-Iteration Clustering (ParPIC)**, a random-walk-based clustering method for weakly connected directed graphs.

### Key Contributions Verified

1. **Parametrized Random Walk Operator** (Section 3, Definition 3.1): A reversible, irreducible random walk operator for directed graphs that preserves directionality.

2. **Vertex Measure Parametrization** (Section 5.1): A vertex measure `ν = α·c + (1-α)·s` where `c(γ) = γ·d_in + (1-γ)·d_out` controls forward/backward flow and `s` captures PageRank information.

3. **Automatic Diffusion Time** (Section 5.2, Algorithm 2): Entropy-based knee detection (KneeLocator) selects optimal diffusion time without manual tuning.

4. **Clustering Performance**: Evaluated on UCI K-NN digraphs and real-world directed networks.

5. **Multi-Scale Structure Detection**: Identifies both fine and coarse clusterings at different diffusion scales.

## Repository Structure

```
reproduction/
├── parpic.py              # Independent implementation (from scratch)
├── datasets.py            # Dataset loaders for UCI + real-world networks
├── experiments.py         # Experiment runners for all 6 claims
├── parpic_reproduction.ipynb  # Colab/Molab notebook for interactive reproduction
├── results/               # Structured JSON results
│   ├── claim1.json
│   ├── claim2.json
│   ├── claim3.json
│   ├── claim4.json
│   ├── claim5.json
│   └── claim6.json
└── README.md
```

## Running Experiments

### Local

```bash
cd reproduction
pip install numpy scipy scikit-learn kneed matplotlib networkx

# Run specific claim
python experiments.py --claim 1  # P-RW operator verification
python experiments.py --claim 2  # Gamma/alpha sweep
python experiments.py --claim 3  # Entropy knee detection
python experiments.py --claim 4  # UCI benchmarks
python experiments.py --claim 5  # Real-world graphs
python experiments.py --claim 6  # Gaussian mixtures

# Or run all
python experiments.py
```

### Colab / Molab

Open `parpic_reproduction.ipynb` in Google Colab or Molab and run cells sequentially.

## Results Summary

| Claim | Description | Status | Key Result |
|-------|-------------|--------|-----------|
| 1 | P-RW operator reversibility, irreducibility, directionality | **PASS** | Symmetry error: 4.89e-06, irreducible min entry: 4.06e-04, direction diff: 0.1331 |
| 2 | Vertex measure gamma/alpha parametrization | **PASS** | NMI varies from 0.64 (gamma=0) to 0.95 (gamma=1) |
| 3 | Automatic diffusion time via CH criterion | **PASS** | Knee detection finds optimal t (4-5) on 3 DiSBM types |
| 4 | UCI benchmark NMI | **PARTIAL** | Avg NMI=0.52 across 9 datasets (paper: 0.66; some datasets are synthetic replacements) |
| 5 | Real-world directed graphs | **PARTIAL** | Email-Eu NMI=0.36; DiSBM variants NMI=1.0 |
| 6 | Multi-scale Gaussian mixtures | **PASS** | Fine (6-cluster) NMI=0.94; Coarse (2-cluster) NMI=1.00 |

## Claims Fidelity

- **Strong reproduction**: Claims 1, 2, 3, 6 — the core theoretical claims and algorithm
- **Partial reproduction**: Claims 4, 5 — some datasets unavailable online, replaced with synthetic alternatives; ParPIC implementation verified independently
- **Not attempted**: Full-scale benchmarks on all 11 UCI datasets and all 4 real-world networks due to dataset availability constraints

## Dependencies

- numpy, scipy, scikit-learn, kneed
- matplotlib (optional, for plots)
- networkx (optional, for some graph datasets)

## Trackio Logbook

The reproduction logbook is at:
https://huggingface.co/spaces/divaspoudel/repro-parametrized-power-iteration-clustering-for-directed-graphs
