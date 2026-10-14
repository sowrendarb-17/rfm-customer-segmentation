
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RFM Customer Segmentation")
st.write("Customer segmentation using RFM analysis and DBSCAN.")

# Load saved files
@st.cache_resource
def load_model():
    model = joblib.load("dbscan_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("rfm_dbscan_clustered.csv")

try:
    dbscan_model, scaler = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# Validate required columns
required = [
    "CustomerID", "Recency", "Frequency",
    "Monetary", "DBSCAN_Cluster"
]

if not all(col in df.columns for col in required):
    st.error("The CSV is missing one or more required columns.")
    st.stop()

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Customer Segments", "Try a Customer"]
)

if page == "Dashboard":
    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric(
        "Clusters",
        df.loc[df["DBSCAN_Cluster"] != -1, "DBSCAN_Cluster"].nunique()
    )
    c3.metric(
        "Noise Customers",
        f"{(df['DBSCAN_Cluster'] == -1).sum():,}"
    )

    counts = df["DBSCAN_Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Customer Distribution by DBSCAN Cluster")
    ax.set_xlabel("Cluster (-1 means noise)")
    ax.set_ylabel("Customers")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Selected DBSCAN parameters: eps = 0.837, min_samples = 4."
    )

elif page == "Customer Segments":
    st.header("Explore Customer Segments")

    clusters = sorted(df["DBSCAN_Cluster"].unique())
    selected = st.multiselect(
        "Select cluster labels",
        clusters,
        default=clusters
    )

    filtered = df[df["DBSCAN_Cluster"].isin(selected)]

    st.dataframe(filtered, use_container_width=True)

    summary = (
        filtered[filtered["DBSCAN_Cluster"] != -1]
        .groupby("DBSCAN_Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Median_Recency=("Recency", "median"),
            Median_Frequency=("Frequency", "median"),
            Median_Monetary=("Monetary", "median")
        )
        .round(2)
    )

    st.subheader("RFM Summary")
    st.dataframe(summary, use_container_width=True)

    st.download_button(
        "Download selected customers",
        filtered.to_csv(index=False).encode("utf-8"),
        "selected_rfm_customers.csv",
        "text/csv"
    )

elif page == "Try a Customer":
    st.header("Explore a New Customer")
    st.write(
        "Enter RFM values to find the nearest DBSCAN core sample. "
        "This is an approximate assignment, not native DBSCAN prediction."
    )

    recency = st.number_input(
        "Recency (days)", min_value=0.0, value=30.0
    )
    frequency = st.number_input(
        "Frequency (purchases)", min_value=1.0, value=5.0
    )
    monetary = st.number_input(
        "Monetary (total spend)", min_value=0.0, value=500.0
    )

    if st.button("Find Approximate Segment"):
        if len(dbscan_model.core_sample_indices_) == 0:
            st.error("The saved DBSCAN model has no core samples.")
        else:
            new_rfm = pd.DataFrame(
                [[recency, frequency, monetary]],
                columns=["Recency", "Frequency", "Monetary"]
            )

            # This assumes the scaler was fitted on raw RFM columns.
            # If your training used log1p before scaling, apply it here too.
            new_scaled = scaler.transform(new_rfm)

            core_points = dbscan_model.components_
            core_labels = dbscan_model.labels_[
                dbscan_model.core_sample_indices_
            ]

            distances = pairwise_distances(new_scaled, core_points)[0]
            nearest = np.argmin(distances)

            nearest_distance = distances[nearest]
            predicted_cluster = int(core_labels[nearest])

            if nearest_distance <= dbscan_model.eps:
                st.success(
                    f"Approximate cluster assignment: {predicted_cluster}"
                )
                st.write(
                    f"Distance to nearest core sample: "
                    f"{nearest_distance:.4f}"
                )
            else:
                st.warning(
                    "This customer is not within eps of any DBSCAN core "
                    "sample. Treat the customer as unassigned/noise."
                )
