# VAGRU — Volatility-Aware Gated Recurrent Unit

> A drop-in GRU replacement for **irregular time-series** — handles variable time gaps natively.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)

---

## What is VAGRU?

Standard GRUs assume your data is **evenly spaced in time**. Real-world data isn't — sensor readings drop out, events arrive in bursts, and gaps vary wildly. VAGRU fixes this by making the recurrent cell **time-aware**: you feed in the time gap `Δt` alongside your features, and the cell adapts its gating, memory decay, and update behavior accordingly.

**Key features:**
- 🕐 **Time-gap aware** — Takes `Δt` as an explicit input, no preprocessing needed
- 🔌 **Drop-in replacement** — Works as a standard Keras RNN layer
- 🧠 **Smart memory decay** — Learns to forget old info across large gaps while retaining a safety floor
- 📈 **Volatility detection** — Tracks velocity and acceleration of input changes to detect regime shifts
- ⚡ **XLA compatible** — Supports `jit_compile=True` for GPU acceleration

---

## Installation

```bash
git clone https://github.com/yourusername/VAGRU.git
cd VAGRU
pip install -r requirements.txt
```

**Dependencies:** `numpy >= 1.26.0`, `tensorflow >= 2.15.0`

---

## Quick Start

### 1. Basic Usage

```python
from vagru_lib import FastVAGRU
from tensorflow import keras

# Build a simple model
inp = keras.layers.Input(shape=(None, 5))  # 4 features + 1 Δt column (last)
x = FastVAGRU(units=32)(inp)
out = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='mse')
```

### 2. Prepare Your Data

VAGRU expects the **time gap `Δt` as the last feature column**. Just concatenate it:

```python
import numpy as np

# Your data
X_features = np.random.randn(1000, 60, 4)      # (samples, timesteps, features)
delta_t = np.random.uniform(0.01, 1.0, (1000, 60))  # time gaps between steps

# Combine: Δt goes as the LAST column
X_input = np.concatenate([X_features, delta_t[..., np.newaxis]], axis=-1)
# X_input.shape → (1000, 60, 5)

# Train as usual
y = np.random.randn(1000, 60, 1)  # your targets
model.fit(X_input, y, epochs=10, batch_size=64)
```

> **⚠️ Important:** The last column of your input **must** be `Δt`. VAGRU splits it internally — everything except the last column is treated as features, and the last column is treated as the time gap.

### 3. Using the Cell Directly

If you need more control (e.g., custom RNN loops), use the cell:

```python
from vagru_lib import FastVAGRUCell
from tensorflow import keras

cell = FastVAGRUCell(units=64)
rnn = keras.layers.RNN(cell, return_sequences=True, return_state=False)

inp = keras.layers.Input(shape=(None, 5))
x = rnn(inp)
out = keras.layers.Dense(1)(x)
model = keras.Model(inp, out)
```

---

## Run the Benchmarks

Reproduce the VAGRU vs. standard GRU comparison:

```bash
python main.py
```

This runs both models across three temporal regimes and prints the results:

```
----------------------------------------------------------------------
EMPIRICAL BENCHMARK: VAGRU vs. STANDARD GRU
Framework: Continuous-Time Non-Stationary Trajectories
----------------------------------------------------------------------

[INFO] Initializing evaluation for regime: NOISE
  [RESULT] Standard GRU Loss : 0.00039
  [RESULT] Refined VAGRU Loss: 0.00031
  [STATUS] VAGRU outperformed baseline by 19.54%

[INFO] Initializing evaluation for regime: VOID
  [RESULT] Standard GRU Loss : 0.09181
  [RESULT] Refined VAGRU Loss: 0.07924
  [STATUS] VAGRU outperformed baseline by 13.70%

[INFO] Initializing evaluation for regime: MIXED
  [RESULT] Standard GRU Loss : 0.00649
  [RESULT] Refined VAGRU Loss: 0.00656
  [STATUS] Baseline parity not broken.
```

### What the Regimes Test

| Regime | What it simulates | Δt range |
|:---|:---|:---|
| **NOISE** | High-frequency burst sampling (e.g., rapid sensor pings) | 0.005 – 0.05 |
| **VOID** | Deep temporal gaps (e.g., missing data, downtime) | 0.5 – 3.0 |
| **MIXED** | Real-world irregular sampling — mostly fast, occasional big gaps | 0.01 – 4.0 |

---

## API Reference

### `FastVAGRU(units, **kwargs)`

A ready-to-use Keras RNN layer. Drop-in replacement for `keras.layers.GRU`.

| Parameter | Type | Description |
|:---|:---|:---|
| `units` | `int` | Number of hidden units |
| `**kwargs` | — | Passed to `keras.layers.RNN` |

**Input shape:** `(batch_size, timesteps, features + 1)` — last column is `Δt`
**Output shape:** `(batch_size, timesteps, units)` — always returns full sequences

---

### `FastVAGRUCell(units, rho=0.1, gamma_init=0.01, tau_init=0.1, epsilon=1e-6)`

The raw recurrent cell, for advanced use.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `units` | `int` | — | Number of hidden units |
| `rho` | `float` | `0.1` | Memory floor — minimum fraction of hidden state preserved across any gap. Higher = more memory retention. |
| `gamma_init` | `float` | `0.01` | Initial decay rate. Controls how fast memory fades per unit time. |
| `tau_init` | `float` | `0.1` | Initial time constant for all temporal gates. |
| `epsilon` | `float` | `1e-6` | Numerical stability constant. |

**Tuning tips:**
- Increase `rho` (e.g., `0.3`) if your data has very long gaps and you want the model to remember more
- Decrease `rho` (e.g., `0.01`) if old observations become irrelevant quickly
- `gamma_init` and `tau_init` are learned during training — the defaults work well in most cases

---

### `generate_fast_physics_data(n_samples, seq_len, feature_dim, scenario)`

Generates synthetic irregular time-series data for testing.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `n_samples` | `int` | `2000` | Number of sequences |
| `seq_len` | `int` | `60` | Length of each sequence |
| `feature_dim` | `int` | `4` | Number of feature channels |
| `scenario` | `str` | `'mixed'` | One of `'noise'`, `'void'`, or `'mixed'` |

**Returns:** `(X, X_combined, y)`
- `X` — Feature matrix `(n_samples, seq_len, feature_dim)` — for baseline GRU
- `X_combined` — Features + Δt column `(n_samples, seq_len, feature_dim + 1)` — for VAGRU
- `y` — Targets `(n_samples, seq_len, 1)` — one-step-ahead prediction

```python
from vagru_lib import generate_fast_physics_data

X, X_combined, y = generate_fast_physics_data(1000, 60, 4, 'noise')
# X.shape       → (1000, 60, 4)
# X_combined.shape → (1000, 60, 5)
# y.shape       → (1000, 60, 1)
```

---

## Example: Building a Deeper Model

```python
from vagru_lib import FastVAGRU, FastVAGRUCell
from tensorflow import keras

inp = keras.layers.Input(shape=(None, 5))  # 4 features + Δt

# Stack two VAGRU layers
x = FastVAGRU(64)(inp)

# Add Δt back for the second VAGRU layer
# (the first layer outputs hidden states without Δt)
dt = keras.layers.Lambda(lambda t: t[..., -1:])(inp)  # extract Δt
x = keras.layers.Concatenate()([x, dt])                # re-attach Δt
x = FastVAGRU(32)(x)

out = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')

model.summary()
```

---

## Project Structure

```
VAGRU/
├── vagru_lib.py        # Core: FastVAGRUCell, FastVAGRU, data generator
├── main.py             # Benchmark script (VAGRU vs GRU comparison)
├── requirements.txt    # Dependencies
├── LICENSE             # MIT License
└── README.md           # You are here
```

---

## How It Works (Short Version)

Standard GRU treats every timestep the same, regardless of how much time has passed. VAGRU adds three things:

1. **Memory decay** — The hidden state fades based on how much time has passed (`Δt`). Big gap? More forgetting. Small gap? State stays mostly intact. A safety floor (`rho`) prevents complete memory loss.

2. **Volatility sensing** — Tracks how fast inputs are changing (velocity) and how that change itself is changing (acceleration). This helps detect sudden shifts in the data pattern.

3. **Adaptive update strength** — The amount of state update scales with `Δt`, input strength, and detected volatility — all through a smooth exponential function that plays nicely with gradients.

---

## Citation

```bibtex
@article{saha2026vagru,
  title   = {Volatility-Aware Gated Recurrent Units: Continuously Differentiable
             Contractive Sequence Modeling with GPU Stabilization},
  author  = {Saha, Pulak},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.