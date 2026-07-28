# FHLSNN

# FHL-SNN

  
Implementation details for **FHL-SNN** (Fractional Hodge Laplacian Simplicial

Neural Network). 

  


  

---

  

## 1. Installation

  

Python 3.10+ is required.

  

```bash

git clone <url> FHL-SNN

cd FHL-SNN

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

```

  

`requirements.txt`:

  

```

numpy>=1.24

scipy>=1.10

networkx>=3.0

scikit-learn>=1.3

torch>=2.0

torch-geometric>=2.4

jupyter>=1.0

```

  

CUDA will be required for training. 

  

---

  

## 2. Repository structure

  

```

FHL-SNN/

├── fhl_snn/

│ ├── utils.py seeding, timing, resumable result cache, table formatting

│ ├── data.py benchmark loading, synthetic graph families, edge splits

│ ├── complexes.py boundary operators, Hodge Laplacians, spectrum diagnostics

│ ├── operators.py fractional powers, Chebyshev filter, stochasticity checks

│ ├── models.py FHL-SNN architecture and training loop

│ └── experiments.py one entry point per experiment

├── notebooks/

│ └── FHL_SNN_experiments.ipynb runs all experiments end to end

├── results/ JSON cache written by the sweeps

├── requirements.txt

└── README.md

```

  

---

  
Sample code to reproduce results
```

  

```python

from fhl_snn import data as dta, experiments as ex

  

dataset = dta.load_dataset('Cora', root='./data') # downloads on first call

  

ex.exp_gamma_sweep(dataset, gammas=(0.3, 0.5, 0.7, 1.0), seeds=range(10))

ex.exp_k_study(dataset, Ks=(4, 8, 16, 32), seeds=range(10))

ex.exp_expander_mixing()

ex.exp_structure_control(dataset, seeds=range(10))

```




  

---

  

## 4. Model Details

  

Dual pathways over the simplicial complex and some basic high-level details. See source paper for more details. 

  

| Component | Definition |

|---|---|

| Node pathway | `X → Linear → P_γ(L₀) → Linear → P_γ(L₀)` |

| Simplicial pathway | edge features `xᵤ + xᵥ` → `Linear → P_γ(L₁) → Linear → P_γ(L₁)` |

| Coupling | edge embeddings averaged onto nodes via row-normalised `\|B₁\|`, concatenated, `Linear` |

| Scoring | `MLP([z_u ⊙ z_v, \|z_u − z_v\|])` |

| Propagation operator | `P_γ = I − ρ L^γ`, `ρ = λ_max^{−γ}` |

| Operators | `L₀ = B₁B₁ᵀ`, `L₁ = B₁ᵀB₁ + B₂B₂ᵀ` |

  

`L^γ` is applied by a degree-`K` Chebyshev expansion using sparse matrix–vector

products, giving `O(K·nnz)` cost per layer.

  

---

  

## 5. Hyperparameters

  
Optimal hyperparameters were found using grid search. 

  

### Architecture


| Parameter | Value |

|---|---|

| Hidden dimension | 64 |

| Output (embedding) dimension | 32 |

| Scorer hidden dimension | 32 |

| Layers per pathway | 2 |

| Dropout | 0.5 |


  

### Optimisation

  

| Parameter | Value |

|---|---|

| Optimiser | Adam |

| Learning rate | 0.01 |

| Weight decay | 5e-4 |

| Loss | Binary cross-entropy with logits |

| Negative sampling ratio | 1 : 1 (resampled each epoch) |

  

### Operator

  

| Parameter | Value | Notes |

|---|---|---|

| Fractional exponent `γ` | grid over `{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}` | `γ = 1.0` is the non-fractional baseline |


  

### Data

  

| Parameter | Value |

|---|---|

| Split | 85 % train / 5 % validation / 10 % test, on edges |


  

  

---

  
