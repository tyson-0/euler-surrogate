import streamlit as st
import pandas as pd
import yaml
import os
import sys

sys.path.append(os.path.dirname(__file__))
from core import Euler

st.set_page_config(page_title="Euler — Physics Informed Surrogate", layout="wide")
st.title("Euler")
st.caption("Physics Informed Surrogate Modeling — by Tyson Physics")

# user data input field
st.subheader("1. Upload your simulation data")
uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    with open("uploaded_data.csv", "wb") as f:
        f.write(uploaded.getbuffer())
    
    st.dataframe(df.head())
    st.write(f"**{len(df)} rows** | **{len(df.columns)} columns** | Columns: {list(df.columns)}")

    # select training stuff
    
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

        # section to write PDE
        st.subheader("3. Define your PDE")
        st.caption("Use diff(vars['u'], vars['x'], order=2) for derivatives. Variables must match your input column names.")

        st.info("""
            **How to write your PDE:**
            - `vars["column_name"]` — use for variables you want to differentiate (spatial/temporal like x, t)
            - `real_vars["column_name"]` — use for physical constants and parameters (like k, f, T1, T2)
            - `diff(vars["u"], vars["x"], order=2)` — computes d²u/dx²
            - All column names must match your selected input columns exactly
            - Return the residual — the expression that should equal zero
            """)


        pde_code = st.text_area(
            "Write your PDE residual function (should return 0 when physics is satisfied)",
            value="""def my_pde(vars, real_vars, diff):
    d2u_dx2 = diff(vars["u"], vars["x"], order=2)
    k = real_vars["k"]
    f = real_vars["f"]
    return (k / f) * d2u_dx2 + 1""",
            height=150
        )

        # to setup training
        st.subheader("4. Training settings")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 1000, 10000, 5000, step=500)
        with col2:
            lambda_physics = st.slider("Physics loss weight", 0.0, 1.0, 0.01)

        # to train. to modify this, please take care of callback argument in core.py
        st.subheader("5. Train")
        if st.button("Train Model", type="primary"):
            try:
                exec_globals = {}
                exec(pde_code, exec_globals)
                pde_fn = exec_globals["my_pde"]

                model = Euler("uploaded_data.csv")
                model.set_pde(pde_fn)

                # live display
                progress_bar = st.progress(0)
                loss_display = st.empty()
                chart_data = {"Total Loss": [], "Data Loss": [], "Physics Loss": []}
                chart = st.line_chart(chart_data)

                def on_epoch(epoch, loss, data_loss, physics_loss):
                    progress_bar.progress(epoch / epochs)
                    loss_display.markdown(f"**Epoch {epoch}/{epochs}** | Loss `{loss:.6f}` | Data `{data_loss:.6f}` | Physics `{physics_loss:.6f}`")
                    chart.add_rows({
                        "Total Loss": [loss],
                        "Data Loss": [data_loss],
                        "Physics Loss": [physics_loss]
                    })

                model.fit(epochs=epochs, lambda_physics=lambda_physics, callback=on_epoch)
                model.save("saved_model.pt")

                st.session_state["model"] = model
                st.session_state["input_cols"] = input_cols
                st.session_state["output_col"] = output_col
                st.success("Training complete! Model saved.")

            except Exception as e:
                st.error(f"Error during training: {e}")

        # prediction section
        if "model" in st.session_state:
            st.subheader("6. Predict")
            st.caption("Enter values for each input variable")
            
            input_values = []
            cols = st.columns(len(st.session_state["input_cols"]))
            for i, col in enumerate(st.session_state["input_cols"]):
                with cols[i]:
                    val = st.number_input(f"{col}", value=0.0, format="%.4f")
                    input_values.append(val)

            if st.button("Predict", type="primary"):
                model = st.session_state["model"]
                result = model.predict(input_values)
                st.success(f"Predicted {st.session_state['output_col']}: **{result:.4f}**")