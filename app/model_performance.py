import streamlit as st


def show_performance():

    st.title("📊 Model Performance")

    st.write(
        "Performance of the optimized Random Forest model "
        "on the test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MAE", "3.15 min")

    with col2:
        st.metric("RMSE", "3.99 min")

    with col3:
        st.metric("R²", "0.815")

    with col4:
        st.metric("Adjusted R²", "0.815")

    st.subheader("Interpretation")

    st.write(
        "The model predicts delivery time with an average "
        "absolute error of approximately 3.15 minutes on "
        "unseen test data."
    )