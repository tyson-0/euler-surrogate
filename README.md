# Euler — Physics-Informed Surrogate Modeling

**By Tyson Physics**

Euler is an open-source Python library that trains a neural surrogate model on your simulation data — then predicts outputs instantly for any new input, while enforcing your governing physics equation.

---

## The Problem

Running engineering simulations is slow and expensive. A single FEA or CFD run can take hours. Testing hundreds of design variations takes days.

**Euler fixes this.**

Train once on your simulation data. Query instantly forever.

---

## How It Works

| | Traditional Simulation | Euler |
|---|---|---|
| Speed | Hours per run | Milliseconds |
| Cost | Expensive software | Free and open source |
| Physics | Solved numerically | Enforced by neural network |
| Scalability | One run at a time | Million queries instantly |

---

## Quick Start

### 1. Prepare your simulation data as a CSV

```
T1, T2, x, k, f, u
100, 50, 0.1, 200, 50000, 67.3
100, 50, 0.3, 200, 50000, 54.1
...
```

### 2. Define your config.yaml

```yaml
inputs: [T1, T2, x, k, f]
output: u
```

### 3. Train and predict

```python
from core import Euler

def my_pde(vars, real_vars, diff):
    d2u_dx2 = diff(vars["u"], vars["x"], order=2)
    k = real_vars["k"]
    f = real_vars["f"]
    return (k / f) * d2u_dx2 + 1

model = Euler("data.csv")
model.set_pde(my_pde)
model.fit(epochs=5000)

result = model.predict([100, 50, 0.5, 200, 50000])
print(f"Predicted: {result:.4f}")
```

---

## Web App

Don't want to write code? Use the Euler web app:

**[Launch Euler App →](https://euler-surrogate.streamlit.app)**

Upload your CSV, define your inputs and outputs, train with one click, and query predictions instantly — no coding required.

---

## Benchmark

Tested on 1D steady-state heat equation with source term:

`k * d²u/dx² + f = 0`

| Data Points | Analytical | Euler | Error |
|-------------|-----------|-------|-------|
| 10000 | 43.75 | 43.25 | 0.50° |
| 100 | 43.75 | 43.07 | 0.68° |

Sub-degree accuracy on thermal problems with sparse data.

---

## Installation

```bash
git clone https://github.com/tyson-0/euler-surrogate.git
cd euler-surrogate
pip install -r requirements.txt
```

---

## Roadmap

- **v0.2** — Generic PDE interface, CSV data loading, Streamlit web app ← you are here
- **v0.3** — Pip package, transient problems, benchmark suite
- **v0.4** — Uncertainty quantification, adaptive collocation
- **v0.5** — arXiv paper, community physics library
- **v1.0** — Desktop app, cloud platform, enterprise integrations

---

## About Tyson Physics

Tyson Physics builds open-source scientific AI tools for engineers and researchers. We believe simulation AI should be what NumPy is to data science — open, free, and owned by the community.

---

## License

MIT License — free for everyone.

---

## Contributing

Contributions welcome. Open an issue or submit a pull request.

*Built by an engineer who got tired of waiting hours for simulation results.*