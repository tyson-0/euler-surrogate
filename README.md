# Euler

Open-source surrogate modeling framework for engineers, with physics-informed learning.

By Tyson Physics

---

## The idea

Every time you change a design parameter, you run the simulation again. That takes hours. Euler learns from your existing simulation data and predicts instantly for any new input — no simulation needed.

Train once. Predict forever.

---

## Try it

No-code web app: https://euler-surrogate.streamlit.app

Upload your CSV, pick your input and output columns, hit train, and start predicting. No Python required.

---

## API

```python
from core import Euler

model = Euler("data.csv")
model.fit(epochs=5000)
model.save("my_model.pt")

print(model.predict([100, 50, 0.5, 200, 50000]))
```

Loading a saved model later:

```python
model = Euler.from_saved("my_model.pt")
print(model.predict([100, 50, 0.5, 200, 50000]))
```

---

## Two modes

Surrogate mode is pure data-driven and works reliably for any simulation dataset. This is what you should use.

Physics Informed mode lets you define your governing differential equation so the model enforces it during training. This is experimental in v0.1 — physics scaling with parametric inputs is still being fixed. It will work properly in v0.2.

---

## Your data format

Euler works with any simulation CSV. Rows are simulation runs, columns are your parameters and output:

```
T1,  T2,  x,   k,   f,      u
100, 50,  0.1, 200, 50000,  67.3
100, 50,  0.3, 200, 50000,  54.1
```

---

## Installation

```bash
git clone https://github.com/tyson-0/euler-surrogate.git
cd euler-surrogate
pip install -r requirements.txt
```

---

## What is working and what is not

Surrogate mode works well. You can train on any simulation CSV and get accurate predictions. Save and load models. Use the web app without writing code.

Physics enforcement is integrated but experimental. The framework accepts any PDE you write. The issue is scaling — derivatives computed in normalized input space don't match real physical constants, causing the physics loss to not enforce correctly. This is a known problem in parametric PINNs and the fix is being implemented for v0.2.

---

## Roadmap

v0.1.1 — surrogate mode, save and load, Streamlit web app (current)

v0.2 — physics enforcement fixed, pip installable package

v0.3 — transient problems, uncertainty quantification

v1.0 — desktop app, cloud platform

---

## Contributing

Open an issue if something breaks. Pull requests are welcome.

MIT License. Built by an engineer who got tired of waiting hours for simulation results.