import streamlit as st

from prediction import show_prediction
from model_performance import show_performance
from visualization import show_visualization


st.set_page_config(
    page_title="Delivery ETA Predictor",
    page_icon="🚚",
    layout="wide"
)


st.title("🚚 Delivery ETA Predictor")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Prediction",
        "Model Performance",
        "Visualization"
    ]
)


if page == "Prediction":
    show_prediction()

elif page == "Model Performance":
    show_performance()

elif page == "Visualization":
    show_visualization()