import streamlit as st
import pandas as pd
import yaml
import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from core import Euler

st.set_page_config(page_title="Euler — Physics Informed Surrogate", layout="wide")

st.title("Euler")
st.caption("Physics-Informed Surrogate Modeling Framework — by Tyson Physics")

# ── LOAD EXISTING MODEL ──
with st.expander("📂 Load a previously saved model"):
    uploaded_model = st.file_uploader("Upload saved model (.pt file)", type="pt")
    if uploaded_model:
        with open("loaded_model.pt", "wb") as f:
            f.write(uploaded_model.getbuffer())
        try:
            model = Euler.from_saved("loaded_model.pt")
            st.session_state["model"] = model
            st.session_state["input_cols"] = model.input_df_columns
            st.session_state["output_col"] = "output"
            st.session_state["trained"] = True
            st.success(f"Model loaded! Inputs: {model.input_df_columns}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

st.divider()

# ── TRAIN NEW MODEL ──
st.subheader("1. Upload your simulation data")
uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    with open("uploaded_data.csv", "wb") as f:
        f.write(uploaded.getbuffer())

    st.dataframe(df.head())
    st.write(f"**{len(df)} rows** | **{len(df.columns)} columns** | Columns: {list(df.columns)}")

    # ── COLUMNS ──
    st.subheader("2. Define inputs and output")
    col1, col2 = st.columns(2)
    with col1:
        input_cols = st.multiselect("Select input columns", df.columns)
    with col2:
        output_col = st.selectbox("Select output column", df.columns)

    if input_cols and output_col:
        config = {"inputs": list(input_cols), "output": output_col}
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        st.success(f"Config saved — inputs: {input_cols}, output: {output_col}")

        # ── MODE ──
        st.subheader("3. Select mode")
        mode = st.radio(
            "Training mode",
            ["Surrogate Only (Recommended)", "Physics Informed (Experimental)"],
            help="Surrogate mode is pure data-driven and works reliably. Physics mode enforces your PDE but is experimental in v0.1."
        )
        physics_mode = mode == "Physics Informed (Experimental)"

        if physics_mode:
            st.warning("⚠️ Physics enforcement is experimental in v0.1. Proper scaling fix coming in v0.2.")
            st.info("""
**How to write your PDE:**
- `vars["column"]` — scaled, use for differentiation
- `real_vars["column"]` — real physical values, use for constants
- `diff(vars["u"], vars["x"], order=2)` — computes d²u/dx²
- Column names must match selected input columns exactly
- Return the residual — expression that equals zero when physics is satisfied
            """)
            pde_code = st.text_area(
                "Write your PDE residual function",
                value="""def my_pde(vars, real_vars, diff):
    # Example: 1D steady state heat equation
    d2u_dx2 = diff(vars["u"], vars["x"], order=2)
    k = real_vars["k"]
    f = real_vars["f"]
    return (k / f) * d2u_dx2 + 1""",
                height=150
            )
        else:
            st.success("✅ Surrogate mode — pure data driven, works reliably for any simulation dataset.")
            pde_code = None

        # ── TRAINING SETTINGS ──
        st.subheader("4. Training settings")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 1000, 10000, 5000, step=500)
        with col2:
            lambda_physics = st.slider("Physics loss weight", 0.0, 0.1, 0.01, step=0.005) if physics_mode else 0.0

        # ── TRAIN ──
        st.subheader("5. Train")
        if st.button("Train Model", type="primary"):
            try:
                pde_fn = None
                if physics_mode and pde_code:
                    exec_globals = {}
                    exec(pde_code, exec_globals)
                    pde_fn = exec_globals["my_pde"]

                model = Euler("uploaded_data.csv", physics_loss=physics_mode)

                if pde_fn:
                    model.set_pde(pde_fn)

                progress_bar = st.progress(0)
                loss_display = st.empty()
                chart_data = {"Total Loss": [], "Data Loss": []}
                if physics_mode:
                    chart_data["Physics Loss"] = []
                chart = st.line_chart(chart_data)

                def on_epoch(epoch, loss, data_loss, physics_loss):
                    progress_bar.progress(epoch / epochs)
                    if physics_mode:
                        loss_display.markdown(f"**Epoch {epoch}/{epochs}** | Loss `{loss:.6f}` | Data `{data_loss:.6f}` | Physics `{physics_loss:.6f}`")
                        chart.add_rows({"Total Loss": [loss], "Data Loss": [data_loss], "Physics Loss": [physics_loss]})
                    else:
                        loss_display.markdown(f"**Epoch {epoch}/{epochs}** | Loss `{loss:.6f}` | Data `{data_loss:.6f}`")
                        chart.add_rows({"Total Loss": [loss], "Data Loss": [data_loss]})

                model.fit(epochs=epochs, lambda_physics=lambda_physics, callback=on_epoch)

                st.session_state["model"] = model
                st.session_state["input_cols"] = input_cols
                st.session_state["output_col"] = output_col
                st.session_state["trained"] = True
                st.success("Training complete!")

            except Exception as e:
                st.error(f"Error during training: {e}")

        # ── SAVE ──
        if st.session_state.get("trained"):
            if st.button("💾 Save Model"):
                try:
                    st.session_state["model"].save("saved_model.pt")
                    with open("saved_model.pt", "rb") as f:
                        st.download_button(
                            label="Download saved_model.pt",
                            data=f,
                            file_name="saved_model.pt",
                            mime="application/octet-stream"
                        )
                    st.success("Model saved!")
                except Exception as e:
                    st.error(f"Error saving: {e}")

# ── PREDICT ──
if st.session_state.get("trained") and "model" in st.session_state:
    st.divider()
    st.subheader("6. Predict")
    st.caption("Enter values for each input variable")

    input_values = []
    cols = st.columns(len(st.session_state["input_cols"]))
    for i, col in enumerate(st.session_state["input_cols"]):
        with cols[i]:
            val = st.number_input(f"{col}", value=0.0, format="%.4f")
            input_values.append(val)

    if st.button("Predict", type="primary"):
        try:
            model = st.session_state["model"]
            result = model.predict(input_values)
            st.success(f"Predicted **{st.session_state['output_col']}**: `{result:.4f}`")
        except Exception as e:
            st.error(f"Prediction error: {e}")