import streamlit as st
import pickle
import pandas as pd

# Load model
with open("kmeans.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Activity Clustering App", layout="wide")

st.title("🏃 Activity Clustering App")
st.markdown("Upload your dataset to see cluster results")

# Sidebar
st.sidebar.title("ℹ️ Info")
st.sidebar.info("Upload a CSV file to perform clustering")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # ✅ Keep Activity separately (for display)
    if "Activity" in df.columns:
        y = df["Activity"]
        X = df.drop("Activity", axis=1)
    else:
        y = None
        X = df

    # SAME ML logic
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)

    # Add results
    df["Cluster"] = clusters

    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    # ✅ Show Activity vs Cluster (UI feature)
    if y is not None:
        st.subheader("🔍 Activity vs Cluster")
        st.dataframe(pd.crosstab(df["Cluster"], y))

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Results",
        csv,
        "clustered_data.csv",
        "text/csv"
    )

else:
    st.info("👆 Please upload a CSV file to continue")
