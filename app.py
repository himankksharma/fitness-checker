import streamlit as st
import pickle
import pandas as pd

# Load model
with open("kmeans.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Activity Clustering App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data", df.head())

    # ⚠️ Make sure no Activity column
    if "Activity" in df.columns:
        df = df.drop("Activity", axis=1)

    # Apply scaling
    X_scaled = scaler.transform(df)

    # Predict
    clusters = model.predict(X_scaled)

    df["Cluster"] = clusters

    st.write("Results", df)