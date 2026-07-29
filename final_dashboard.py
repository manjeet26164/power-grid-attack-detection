import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# so the model/utils modules import correctly no matter where streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Power Grid Attack Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; color: #00ffcc; }
    .centered-title { text-align: center; }
    .centered-subtitle { text-align: center; color: #a1a1a1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# plot file paths used across tabs
model_img_path = PLOTS_DIR / "model_comparison.png"
po_img_path = PLOTS_DIR / "po_analysis.png"
confusion_img_path = PLOTS_DIR / "confusion_matrix.png"
if not confusion_img_path.exists():
    confusion_img_path = PLOTS_DIR / "confusion_lstm.png"
state_scatter_path = PLOTS_DIR / "state_scatter.png"
metrics_file_path = PLOTS_DIR / "metrics_backup.pkl"

# load saved F1 scores from comparison_models.py, if available
if metrics_file_path.exists():
    try:
        with open(metrics_file_path, "rb") as f:
            saved_metrics = pickle.load(f)
        f1_lstm = round(saved_metrics["lstm_f1"], 4)
        f1_rf = round(saved_metrics["rf_f1"], 4)
        f1_fnn = round(saved_metrics["fnn_f1"], 4)
        status_msg = "Case14 grid · 6 of 20 lines observed"
    except Exception as e:
        f1_lstm = f1_rf = f1_fnn = 0.0
        status_msg = f"Could not read metrics file: {e}"
else:
    f1_lstm = f1_rf = f1_fnn = 0.0
    status_msg = "metrics_backup.pkl not found — run comparison_models.py first"

with st.sidebar:
    st.markdown("## Model Performance")
    st.markdown("---")
    st.metric("LSTM F1 Score", f1_lstm)
    st.metric("Random Forest F1 Score", f1_rf)
    st.metric("FNN F1 Score", f1_fnn)
    st.markdown("---")
    st.caption(status_msg)

st.markdown("<h1 class='centered-title'>⚡ Power Grid Attack Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='centered-subtitle'>LSTM-based attack detection, localization and state estimation "
    "from partial grid observations</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Results Overview",
    "📈 Training Curves",
    "🔬 Robustness Tests",
    "🎯 Live Simulation",
])

# ==================== TAB 1: RESULTS OVERVIEW ====================
with tab1:
    st.markdown("### Model Comparison")
    metrics_df = pd.DataFrame({
        "Model": ["LSTM", "Random Forest", "FNN"],
        "F1 Score": [f1_lstm, f1_rf, f1_fnn],
        "Paper Reference Range": ["0.92 - 0.99", "0.92 - 0.99", "0.92 - 0.99"],
    })
    st.table(metrics_df)

    st.divider()
    st.markdown("### F1 Score Comparison")
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        if model_img_path.exists():
            st.image(str(model_img_path), use_container_width=True)
        else:
            st.error("model_comparison.png not found. Run comparison_models.py to generate it.")

    st.divider()
    st.markdown("### Confusion Matrix & State Estimation")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix (Attack Detection)")
        if confusion_img_path.exists():
            st.image(str(confusion_img_path), use_container_width=True)
        else:
            st.info("Confusion matrix image not found in plots/.")

    with col2:
        st.markdown("#### Predicted vs Real (State Estimation)")
        if state_scatter_path.exists():
            st.image(str(state_scatter_path), use_container_width=True)
        else:
            st.info("state_scatter.png not found in plots/.")

# ==================== TAB 2: TRAINING CURVES ====================
with tab2:
    st.subheader("Training & Validation Curves")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Occurrence Detection")
        occ_curve_path = PLOTS_DIR / "occurrence_training_curves.png"
        if occ_curve_path.exists():
            st.image(str(occ_curve_path), use_container_width=True)
        else:
            st.info("occurrence_training_curves.png not found.")

    with col2:
        st.markdown("#### Location Detection")
        loc_curve_path = PLOTS_DIR / "location_training_curves.png"
        if loc_curve_path.exists():
            st.image(str(loc_curve_path), use_container_width=True)
        else:
            st.info("location_training_curves.png not found.")

    st.divider()
    st.markdown("#### State Estimation")
    state_curve_path = PLOTS_DIR / "state_training_curves.png"
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        if state_curve_path.exists():
            st.image(str(state_curve_path), use_container_width=True)
        else:
            st.info("state_training_curves.png not found.")

# ==================== TAB 3: ROBUSTNESS TESTS ====================
with tab3:
    st.subheader("Noise Robustness & Partial Observation Effects")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Robustness to Sensor Noise")
        noise_plot_path = PLOTS_DIR / "noise_robustness.png"
        if noise_plot_path.exists():
            st.image(str(noise_plot_path), use_container_width=True)
        else:
            st.info("noise_robustness.png not found. Run noise_test.py first.")

    with col2:
        st.markdown("#### Effect of Partial Observation (Po)")
        if po_img_path.exists():
            st.image(str(po_img_path), use_container_width=True)
        else:
            st.info("po_analysis.png not found. Run po_analysis.py first.")

    st.divider()
    st.markdown("### State Estimation Diagnostics")
    st.write("Real vs predicted capacity from the trained state-estimation LSTM, from evaluate_models.py.")

    state_series_path = PLOTS_DIR / "state_estimation.png"
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='text-align:center;font-weight:bold;'>Real vs Predicted (per line)</p>", unsafe_allow_html=True)
        if state_series_path.exists():
            st.image(str(state_series_path), use_container_width=True)
        else:
            st.info("state_estimation.png not found. Run evaluate_models.py first.")

    with col2:
        st.markdown("<p style='text-align:center;font-weight:bold;'>Predicted vs Real Scatter</p>", unsafe_allow_html=True)
        if state_scatter_path.exists():
            st.image(str(state_scatter_path), use_container_width=True)
        else:
            st.info("state_scatter.png not found. Run evaluate_models.py first.")

# ==================== TAB 4: LIVE SIMULATION ====================
with tab4:
    st.subheader("Run the trained models on real or manually crafted input")
    st.caption(
        "Loads the saved checkpoints (best_occurrence_model.pt, best_location_model.pt, "
        "best_state_model.pt) and runs live inference."
    )
    st.divider()

    PREPROC_DIR = BASE_DIR / "data" / "preprocessed"
    MODELS_DIR = BASE_DIR / "models"

    @st.cache_resource(show_spinner="Loading trained models...")
    def load_assets():
        import torch
        from build_lstm_model import AttackOccurrenceModel, AttackLocationModel, StateEstimationModel

        required = {
            "X_test": PREPROC_DIR / "X_test.npy",
            "y_test_occur": PREPROC_DIR / "y_test_occur.npy",
            "y_test_loc": PREPROC_DIR / "y_test_loc.npy",
            "selected_lines": PREPROC_DIR / "selected_lines.npy",
            "scaler": PREPROC_DIR / "scaler.pkl",
            "occurrence_ckpt": MODELS_DIR / "best_occurrence_model.pt",
            "location_ckpt": MODELS_DIR / "best_location_model.pt",
            "state_ckpt": MODELS_DIR / "best_state_model.pt",
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

        x_test = np.load(required["X_test"])
        y_test_occur = np.load(required["y_test_occur"])
        y_test_loc = np.load(required["y_test_loc"])
        selected_lines = np.load(required["selected_lines"])

        with open(required["scaler"], "rb") as f:
            scaler = pickle.load(f)

        occ_ckpt = torch.load(required["occurrence_ckpt"], map_location="cpu", weights_only=False)
        occ_model = AttackOccurrenceModel(occ_ckpt["input_dim"])
        occ_model.load_state_dict(occ_ckpt["state_dict"])
        occ_model.eval()

        loc_ckpt = torch.load(required["location_ckpt"], map_location="cpu", weights_only=False)
        loc_model = AttackLocationModel(loc_ckpt["input_dim"])
        loc_model.load_state_dict(loc_ckpt["state_dict"])
        loc_model.eval()

        state_ckpt = torch.load(required["state_ckpt"], map_location="cpu", weights_only=False)
        state_model = StateEstimationModel(state_ckpt["input_dim"], output_dim=20)
        state_model.load_state_dict(state_ckpt["state_dict"])
        state_model.eval()

        return {
            "x_test": x_test,
            "y_test_occur": y_test_occur,
            "y_test_loc": y_test_loc,
            "selected_lines": selected_lines,
            "scaler": scaler,
            "occ_model": occ_model,
            "loc_model": loc_model,
            "state_model": state_model,
        }

    try:
        assets = load_assets()
    except FileNotFoundError as exc:
        st.error(
            "Live inference needs the preprocessed data and trained checkpoints on disk. "
            "Run preprocess_data.py, train_models.py and build_lstm_model.py first, then reload this tab.\n\n"
            + str(exc)
        )
        assets = None

    if assets is not None:
        import torch

        sim_mode = st.radio(
            "Simulation mode",
            ["Replay a real test-set window", "Manually craft an attack"],
            horizontal=True,
        )
        st.divider()

        selected_lines = assets["selected_lines"]
        num_observed = len(selected_lines)

        # bar chart of the 21-way location softmax — use an int index so it sorts
        # numerically (string labels like "10" sort before "2" alphabetically)
        def show_location_chart(loc_probs):
            labels = ["No Attack" if i == 0 else f"Line {i}" for i in range(len(loc_probs))]
            loc_df = pd.DataFrame({"probability": loc_probs}, index=range(len(loc_probs)))
            loc_df.index.name = "class (0 = No Attack)"
            st.bar_chart(loc_df)
            top3 = np.argsort(loc_probs)[::-1][:3]
            st.caption("Top-3: " + ", ".join(f"{labels[i]} ({loc_probs[i]:.3f})" for i in top3))

        def run_inference(window):
            x = torch.from_numpy(window).float().unsqueeze(0)
            with torch.no_grad():
                occ_prob = torch.sigmoid(assets["occ_model"](x)).item()
                loc_probs = torch.softmax(assets["loc_model"](x), dim=-1).squeeze(0).numpy()
                state_pred = assets["state_model"](x).squeeze(0).numpy()
            return occ_prob, loc_probs, state_pred

        if sim_mode == "Replay a real test-set window":
            x_test = assets["x_test"]
            y_occur = assets["y_test_occur"]
            y_loc = assets["y_test_loc"]

            idx = st.slider("Test-set window index", 0, x_test.shape[0] - 1, 0)
            occ_prob, loc_probs, state_pred = run_inference(x_test[idx])
            predicted_line = int(np.argmax(loc_probs))

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted attack probability", f"{occ_prob:.3f}")
            c2.metric("Ground truth", "Attack" if y_occur[idx] == 1 else "No attack")
            c3.metric(
                "Predicted location",
                "No attack" if predicted_line == 0 else f"Line {predicted_line}",
                delta=f"true: {'no attack' if y_loc[idx] == 0 else 'line ' + str(int(y_loc[idx]))}",
            )

            st.markdown("#### Attack location probability")
            show_location_chart(loc_probs)

        else:
            scaler = assets["scaler"]
            data_min = getattr(scaler, "data_min_", None)
            data_max = getattr(scaler, "data_max_", None)
            has_range = data_min is not None and data_max is not None

            st.write(
                f"Trained with Po = {num_observed}/20 lines observed "
                f"(lines: {', '.join(str(int(i) + 1) for i in selected_lines)}). "
                "Each slider is scaled to that line's actual training data range, shown below."
            )

            raw_values = []
            cols = st.columns(num_observed)
            for i, col in enumerate(cols):
                with col:
                    if has_range:
                        lo, hi = float(data_min[i]), float(data_max[i])
                        span = hi - lo if hi > lo else 1.0
                        default_val = lo + 0.3 * span
                        val = st.slider(f"Line {int(selected_lines[i]) + 1}", lo, hi, default_val, span / 200)
                    else:
                        val = st.slider(f"Line {int(selected_lines[i]) + 1}", 0.0, 1.5, 0.3, 0.01)
                    raw_values.append(val)

            raw_array = np.array(raw_values, dtype=np.float32).reshape(1, -1)
            scaled = scaler.transform(raw_array).reshape(-1)
            window = np.tile(scaled, (5, 1))  # same value repeated across the 5-step window

            occ_prob, loc_probs, state_pred = run_inference(window)
            predicted_line = int(np.argmax(loc_probs))

            c1, c2 = st.columns(2)
            c1.metric("Predicted attack probability", f"{occ_prob:.3f}")
            c2.metric("Predicted location", "No attack" if predicted_line == 0 else f"Line {predicted_line}")

            st.markdown("#### Attack location probability")
            show_location_chart(loc_probs)

            with st.expander("Debug: is this input in-distribution?"):
                if not has_range:
                    st.warning("Scaler has no data_min_/data_max_ — cannot show training range.")
                else:
                    rows = []
                    out_of_range = False
                    for i in range(num_observed):
                        lo, hi = float(data_min[i]), float(data_max[i])
                        in_range = lo <= raw_values[i] <= hi
                        out_of_range = out_of_range or not in_range
                        rows.append({
                            "Line": f"Line {int(selected_lines[i]) + 1}",
                            "Your value": round(raw_values[i], 3),
                            "Training min": round(lo, 3),
                            "Training max": round(hi, 3),
                            "Scaled value": round(float(scaled[i]), 3),
                            "In range?": "yes" if in_range else "no",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    if out_of_range:
                        st.warning("One or more values fall outside the training range — the model is extrapolating here.")

            st.caption(
                "Note: repeating a flat value across all 5 timesteps means there is no temporal "
                "pattern for the LSTM to read — this mode is meant for quick sanity checks, not a "
                "realistic attack trajectory."
            )