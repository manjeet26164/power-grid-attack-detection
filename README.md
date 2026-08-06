# Power Grid Attack Detection

![tests](https://github.com/manjeet26164/power-grid-attack-detection/actions/workflows/tests.yml/badge.svg)

LSTM-based cyber-attack detection for power grids — covering **attack occurrence detection**, **attack localization**, and **state estimation** from partial grid observations.

This project is an independent, small-scale implementation of the approach described in *"Deep Learning for Cyber-Attack Detection in Power Grids"* (Zhai, Moradi, Lai — PRX Energy, 2025). The architecture and methodology follow the paper, while the code, preprocessing pipeline, baselines, and interactive dashboard are built from scratch. Evaluated on the **IEEE Case14 test grid** (one of the three grids used in the original paper).

---

## Overview

Modern power grids rely on state estimation to monitor system health, but this makes them vulnerable to **false data injection attacks (FDIA)** that can corrupt sensor readings without triggering traditional bad-data detectors. This project trains an LSTM-based model to:

1. **Detect** whether an attack is occurring (binary classification)
2. **Localize** which transmission line(s) are under attack
3. **Estimate** the true grid state despite corrupted/partial observations

The model is trained and evaluated with only **6 of 20 lines observed** (`Po = 6/20`), simulating realistic partial-observability conditions in a real grid.

---

## Results

Evaluated on the IEEE 14-bus (Case14) held-out test set:

| Model | F1 Score |
|---|---|
| **LSTM (proposed)** | **0.9895** |
| Random Forest (baseline) | 0.9627 |
| FNN (baseline) | 0.9821 |

- All results fall within the paper's reported range (0.92–0.99) for F1 score.
- **State estimation MSE ≈ 1.5 × 10⁻⁴** — better than the original paper's reported figure.
- Training curves, noise-robustness tests, and Po-sweep experiments (varying number of observed lines) all reproduce trends consistent with the paper.

### Why the numbers are trustworthy
- No data leakage in preprocessing (train/test split done before any normalization/statistics are computed).
- EarlyStopping + class weighting used during training to avoid overfitting and handle class imbalance.
- Architecture and training practices are faithful to the original paper's methodology.
- Compared against two independent baselines (Random Forest, FNN), not just reported in isolation.

---

## Live Attack Simulation (Dashboard)

The 4th tab of the dashboard, `Live Simulation`, lets you run the trained checkpoints (`best_occurrence_model.pt`, `best_location_model.pt`, `best_state_model.pt`) interactively:

- **Mode 1 — Replay a real test-set window:** step through an actual test sample and watch live model predictions.
- **Mode 2 — Manually craft an attack:** move sliders for each of the 6 observed lines (Line 2, 8, 9, 12, 13, 18) and see the model's live prediction for attack probability and location.

Each slider is scaled to that specific line's **actual training data range** (not a generic 0–1 scale), since the model was trained on raw power-flow values, not normalized ratios.

---

## Learnings / Debugging Journey

Some of the more interesting bugs and discoveries while building the live simulation tab:

1. **Chart x-axis sorting bug:** Line labels were strings, so the x-axis was sorting alphabetically ("Line 1", "Line 10", "Line 11"...) instead of numerically. Fixed by converting to integer index before plotting.

2. **Missing function bug:** The `run_inference` function definition was accidentally deleted during an earlier edit, breaking the live prediction pipeline. Restored it.

3. **Slider range bug (the interesting one):** Sliders were initially bounded to (0–1.5), assuming the underlying feature was a normalized ratio (ρ). In reality, the preprocessing pipeline's `choose_capacity_channel` heuristic selects a **raw power-flow value** as the feature, with a training range closer to 0–1900+. This meant the sliders couldn't reach realistic input values at all. Fixed by having each slider dynamically pull its min/max from that line's actual training data.

4. **Unresolved edge case:** When given uniform/flat input (all sliders at similar values), the location model becomes overconfident about a specific line (e.g., always predicting "Line 1") without a clear logical basis for it. Root cause isn't fully understood yet — likely related to how the model handles out-of-distribution or degenerate inputs. This is an honest, documented limitation rather than a hidden one, and worth discussing directly in interviews as an example of real debugging and model-behavior analysis (not just "it works").

5. Added a **debug panel** to the dashboard showing: raw slider value, training min/max for that line, scaled value fed to the model, and whether the value is in-range — makes future debugging of similar issues much faster.

---

## Setup & How to Run

1. **Check your environment** (verifies Python version and that all dependencies import correctly):
   ```bash
   python setup_check.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the raw data** (builds sliding-window sequences, applies the train-only scaler, saves arrays to disk):
   ```bash
   python preprocess_data.py
   ```
   Optional flags: `--data-dir`, `--output-dir`, `--sequence-length`, `--po`, `--selected-lines` (see `python preprocess_data.py --help`).

4. **(Optional) Explore/visualize the preprocessed data:**
   ```bash
   python explore_data.py
   python visualize_data.py
   ```

5. **Train the models:**
   ```bash
   python train_models.py
   ```
   This trains the LSTM occurrence/location/state models and saves the best checkpoints (`best_occurrence_model.pt`, `best_location_model.pt`, `best_state_model.pt`).

6. **Train baselines for comparison:**
   ```bash
   python baseline_model.py
   python comparison_models.py
   ```

7. **Evaluate the trained models:**
   ```bash
   python evaluate_models.py
   ```

8. **(Optional) Run robustness experiments:**
   ```bash
   python noise_test.py     # noise-robustness sweep
   python po_analysis.py    # observability (Po) sweep
   ```

9. **Launch the interactive dashboard:**
   ```bash
   streamlit run final_dashboard.py
   ```

---

## Project Structure

```
├── setup_check.py              # Verifies Python/dependency setup
├── preprocess_data.py          # Builds sequences, scales data, train/test split
├── explore_data.py             # Data exploration / summary stats
├── visualize_data.py           # Data visualization utilities
├── train_utils.py              # Shared training helpers (EarlyStopping, checkpointing, etc.)
├── build_lstm_model.py         # LSTM model architecture
├── train_models.py             # Trains occurrence / location / state LSTM models
├── baseline_model.py           # Simple baseline classifiers
├── comparison_models.py        # Random Forest / FNN baselines for comparison
├── evaluate_models.py          # Evaluation metrics on the held-out test set
├── noise_test.py               # Noise-robustness experiments
├── po_analysis.py              # Partial-observability (Po) sweep experiments
├── final_dashboard.py          # Main Streamlit dashboard app (4 tabs, incl. Live Simulation)
├── results/
│   └── baseline_metrics.json   # Saved baseline evaluation metrics
├── requirements.txt
└── README.md

# Generated locally when you run the pipeline (not committed to git):
├── best_occurrence_model.pt    # Trained attack-occurrence detection model
├── best_location_model.pt      # Trained attack-localization model
├── best_state_model.pt         # Trained state-estimation model
├── data/                       # Preprocessed arrays
└── plots/                      # Training curves, evaluation plots
```

---

## Tech Stack

- PyTorch (LSTM architecture, training, inference)
- Scikit-learn (Random Forest baseline)
- Streamlit (dashboard/UI)
- IEEE Case14 test grid (via PandaPower or equivalent)

---

## Known Limitations & Future Work

- Only 1 of the 3 grids from the original paper is implemented (Case14); Case30/Case118 not yet replicated.
- Location model overconfidence on flat/uniform input is unresolved (see Learnings above).
- No adversarial robustness testing against adaptive attackers who know the detection model.

---

## Reference

Zhai, Moradi, Lai — *"Deep Learning for Cyber-Attack Detection in Power Grids"*, PRX Energy, 2025.