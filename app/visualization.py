import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Load Data & Model
# -------------------------

@st.cache_data
def load_data():
    """Load the cleaned delivery dataset."""
    return pd.read_csv("data/cleaned_delivery_data.csv")


@st.cache_resource
def load_model():
    """Load the trained Random Forest model if available."""
    model_path = "models/delivery_eta_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# -------------------------
# Visualization Page
# -------------------------

def show_visualization():
    st.title("📈 Data Visualization & Model Diagnostics")
    st.write(
        "Explore key patterns in the delivery dataset and evaluate the performance "
        "of the trained Random Forest model."
    )

    df = load_data()

    # -------------------------
    # Dataset Overview Metrics
    # -------------------------
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Average Delivery Time", f"{df['Time_taken'].mean():.1f} min")

    st.divider()

    # -------------------------
    # 1. Delivery Time Distribution
    # -------------------------
    st.subheader("1. Delivery Time Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Time_taken"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Delivery Time (minutes)")
    ax.set_ylabel("Number of Deliveries")
    ax.set_title("Distribution of Delivery Time")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 2. Distance vs Delivery Time
    # -------------------------
    st.subheader("2. Distance vs Delivery Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["Distance_km"], df["Time_taken"], alpha=0.3)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.set_title("Distance vs Delivery Time")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 3. Traffic Density vs Delivery Time
    # -------------------------
    st.subheader("3. Traffic Density vs Delivery Time")
    traffic_order = ["Low", "Medium", "High", "Jam"]
    traffic_data = (
        df.groupby("Road_traffic_density")["Time_taken"]
        .mean()
        .reindex(traffic_order)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    traffic_data.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_xlabel("Road Traffic Density")
    ax.set_ylabel("Average Delivery Time (minutes)")
    ax.set_title("Average Delivery Time by Traffic Density")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 4. Weather vs Delivery Time
    # -------------------------
    st.subheader("4. Weather vs Delivery Time")
    weather_data = (
        df.groupby("Weatherconditions")["Time_taken"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    weather_data.plot(kind="bar", ax=ax, color="orange", edgecolor="black")
    ax.set_xlabel("Weather Conditions")
    ax.set_ylabel("Average Delivery Time (minutes)")
    ax.set_title("Average Delivery Time by Weather")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 5. Preparation Time vs Delivery Time
    # -------------------------
    st.subheader("5. Preparation Time vs Delivery Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["Preparation_Time_min"], df["Time_taken"], alpha=0.3, color="green")
    ax.set_xlabel("Preparation Time (minutes)")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.set_title("Preparation Time vs Delivery Time")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 6. Actual vs. Predicted Delivery Time
    # -------------------------
    st.divider()
    st.subheader("6. Actual vs. Predicted Delivery Time")

    model = load_model()

    if model is not None:
        if "Time_taken" in df.columns:
            X = df.drop(columns=["Time_taken"])
            y_actual = df["Time_taken"]

            try:
                y_pred = model.predict(X)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y_actual, y_pred, alpha=0.2, color="purple", label="Predictions")

                # Perfect prediction baseline
                min_val = min(y_actual.min(), y_pred.min())
                max_val = max(y_actual.max(), y_pred.max())
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Perfect Fit (y = x)"
                )

                ax.set_xlabel("Actual Delivery Time (minutes)")
                ax.set_ylabel("Predicted Delivery Time (minutes)")
                ax.set_title("Actual vs. Predicted Delivery Time (Random Forest)")
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                # Summary evaluation metrics
                mae = np.mean(np.abs(y_actual - y_pred))
                rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} min")
                with m_col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} min")

            except Exception as e:
                st.warning(
                    f"Could not generate predictions. Ensure `data/cleaned_delivery_data.csv` "
                    f"matches the feature pipeline expected by the saved model. Details: {e}"
                )
    else:
        st.info("Place your trained model file at `models/delivery_eta_model.joblib` to display Actual vs. Predicted results.")


if __name__ == "__main__":
    show_visualization()