
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RFM Customer Segmentation Dashboard")
st.write(
    "Explore customer segments created using "
    "RFM analysis and DBSCAN clustering."
)

# Load the previously clustered dataset
@st.cache_data
def load_data():
    data = pd.read_csv("rfm_dbscan_clustered.csv")

    required_columns = [
        "CustomerID", "Recency", "Frequency",
        "Monetary", "DBSCAN_Cluster"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return data

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# Convert cluster labels to integers
df["DBSCAN_Cluster"] = pd.to_numeric(
    df["DBSCAN_Cluster"], errors="coerce"
)
df = df.dropna(subset=["DBSCAN_Cluster"])
df["DBSCAN_Cluster"] = df["DBSCAN_Cluster"].astype(int)

# Sidebar filters
st.sidebar.header("Filters")

include_noise = st.sidebar.checkbox(
    "Include noise customers (-1)", value=True
)

available_clusters = sorted(
    df["DBSCAN_Cluster"].unique().tolist()
)

if not include_noise:
    available_clusters = [
        c for c in available_clusters if c != -1
    ]

selected_clusters = st.sidebar.multiselect(
    "Select clusters",
    options=available_clusters,
    default=available_clusters
)

filtered_df = df[
    df["DBSCAN_Cluster"].isin(selected_clusters)
]

# Main metrics
col1, col2, col3 = st.columns(3)

col1.metric("Customers", f"{len(filtered_df):,}")

col2.metric(
    "Clusters",
    filtered_df.loc[
        filtered_df["DBSCAN_Cluster"] != -1,
        "DBSCAN_Cluster"
    ].nunique()
)

noise_count = (
    filtered_df["DBSCAN_Cluster"] == -1
).sum()

col3.metric("Noise customers", f"{noise_count:,}")

st.divider()

# Cluster distribution
st.subheader("Customer Distribution by Cluster")

if filtered_df.empty:
    st.info("Select at least one cluster from the sidebar.")
else:
    counts = (
        filtered_df["DBSCAN_Cluster"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("DBSCAN Cluster (-1 = Noise)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Cluster")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # RFM summary
    st.subheader("RFM Segment Analysis")

    main_customers = filtered_df[
        filtered_df["DBSCAN_Cluster"] != -1
    ]

    if not main_customers.empty:
        summary = (
            main_customers
            .groupby("DBSCAN_Cluster")
            .agg(
                Customer_Count=("CustomerID", "count"),
                Recency_Median=("Recency", "median"),
                Frequency_Median=("Frequency", "median"),
                Monetary_Median=("Monetary", "median")
            )
            .round(2)
        )

        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No non-noise clusters are selected.")

    # Customer table
    st.subheader("Customer Records")
    st.dataframe(filtered_df, use_container_width=True)

    # Download filtered results
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Customer Data",
        data=csv_data,
        file_name="filtered_rfm_customers.csv",
        mime="text/csv"
    )
