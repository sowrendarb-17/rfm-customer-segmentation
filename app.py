
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RFM Customer Segmentation")
st.write(
    "An interactive customer segmentation dashboard "
    "using Recency, Frequency, Monetary analysis and DBSCAN."
)


# --------------------------------------------------
# 2. LOAD SAVED FILES
# --------------------------------------------------

@st.cache_resource
def load_model_files():
    model = joblib.load("dbscan_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    return model, scaler


@st.cache_data
def load_customer_data():
    return pd.read_csv("rfm_dbscan_clustered.csv")


try:
    dbscan_model, scaler = load_model_files()
    df = load_customer_data()

except Exception as error:
    st.error(
        "Unable to load the project files. "
        "Check that the model, scaler and CSV are in "
        "the same GitHub repository folder as app.py."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 3. VALIDATE THE DATA
# --------------------------------------------------

required_columns = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "DBSCAN_Cluster"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

for column in ["Recency", "Frequency", "Monetary"]:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

df["DBSCAN_Cluster"] = pd.to_numeric(
    df["DBSCAN_Cluster"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Recency",
        "Frequency",
        "Monetary",
        "DBSCAN_Cluster"
    ]
).copy()

df["DBSCAN_Cluster"] = df["DBSCAN_Cluster"].astype(int)

if df.empty:
    st.error("The customer dataset has no valid records.")
    st.stop()


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Customer Segments",
        "Try a New Customer"
    ]
)


# --------------------------------------------------
# 5. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Project Overview")

    total_customers = len(df)

    number_of_clusters = df.loc[
        df["DBSCAN_Cluster"] != -1,
        "DBSCAN_Cluster"
    ].nunique()

    noise_count = (
        df["DBSCAN_Cluster"] == -1
    ).sum()

    noise_percentage = (
        noise_count / total_customers * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Number of Clusters",
        number_of_clusters
    )

    col3.metric(
        "Noise Customers",
        f"{noise_count:,}",
        f"{noise_percentage:.2f}% of customers"
    )

    st.divider()

    st.subheader("DBSCAN Model Configuration")

    col1, col2 = st.columns(2)

    col1.info("eps = 0.837")
    col2.info("min_samples = 4")

    st.caption(
        "Best silhouette score from your parameter search: "
        "0.64513, calculated with noise excluded."
    )

    st.subheader("Customer Distribution by Cluster")

    cluster_counts = (
        df["DBSCAN_Cluster"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    cluster_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Cluster label (-1 means noise)")
    ax.set_ylabel("Number of customers")
    ax.set_title("DBSCAN Cluster Distribution")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.download_button(
        label="Download Complete Clustered Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="rfm_dbscan_clustered.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 6. CUSTOMER SEGMENT ANALYSIS
# --------------------------------------------------

elif page == "Customer Segments":

    st.header("Customer Segment Analysis")

    cluster_options = sorted(
        df["DBSCAN_Cluster"].unique().tolist()
    )

    selected_clusters = st.multiselect(
        "Select clusters to view",
        options=cluster_options,
        default=cluster_options
    )

    filtered_df = df[
        df["DBSCAN_Cluster"].isin(selected_clusters)
    ].copy()

    st.write(
        f"Showing {len(filtered_df):,} customer records."
    )

    if filtered_df.empty:
        st.warning("Select at least one cluster.")
        st.stop()

    st.subheader("RFM Statistics by Cluster")

    # Exclude noise from the main segment summary
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
                Monetary_Median=("Monetary", "median"),
                Recency_Mean=("Recency", "mean"),
                Frequency_Mean=("Frequency", "mean"),
                Monetary_Mean=("Monetary", "mean")
            )
            .round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.caption(
            "Lower Recency generally indicates a more recent purchase. "
            "Higher Frequency and Monetary values indicate more "
            "frequent purchasing and higher spending."
        )

        st.subheader("Median RFM Comparison")

        median_summary = summary[
            [
                "Recency_Median",
                "Frequency_Median",
                "Monetary_Median"
            ]
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        median_summary.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("DBSCAN Cluster")
        ax.set_ylabel("Median feature value")
        ax.set_title("Median RFM Values by Cluster")
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info(
            "No regular clusters are selected. "
            "Noise customers are excluded from this summary."
        )

    st.subheader("Customer Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.download_button(
        label="Download Selected Customers",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="selected_rfm_customers.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 7. TRY A NEW CUSTOMER
# --------------------------------------------------

elif page == "Try a New Customer":

    st.header("Test a New Customer")

    st.write(
        "Enter the customer's RFM values. The app applies "
        "the saved preprocessing and searches for the nearest "
        "DBSCAN core sample."
    )

    st.warning(
        "This is an approximate DBSCAN assignment, not a native "
        "DBSCAN prediction. A customer outside the core-sample "
        "neighbourhoods is reported as unassigned."
    )

    with st.form("new_customer_form"):

        recency = st.number_input(
            "Recency (days)",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        frequency = st.number_input(
            "Frequency (purchases)",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        monetary = st.number_input(
            "Monetary (total spend)",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

        submitted = st.form_submit_button(
            "Find Approximate Segment"
        )

    if submitted:

        if not hasattr(
            dbscan_model, "components_"
        ) or not hasattr(
            dbscan_model, "core_sample_indices_"
        ):
            st.error(
                "The saved DBSCAN model does not contain "
                "the required fitted core samples."
            )
            st.stop()

        if len(dbscan_model.components_) == 0:
            st.error(
                "The fitted DBSCAN model contains no core samples."
            )
            st.stop()

        # Preserve the original feature order
        new_rfm = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )

        try:
            # IMPORTANT:
            # The original notebook used log1p before scaling.
            new_rfm_log = np.log1p(new_rfm)

            # Use the saved fitted scaler
            new_scaled = scaler.transform(new_rfm_log)

            # DBSCAN core points are already in scaled space
            core_points = dbscan_model.components_

            # Retrieve labels belonging to the core points
            core_labels = dbscan_model.labels_[
                dbscan_model.core_sample_indices_
            ]

            # Calculate distance to each core sample
            distances = pairwise_distances(
                new_scaled,
                core_points,
                metric="euclidean"
            )[0]

            nearest_index = int(np.argmin(distances))
            nearest_distance = float(
                distances[nearest_index]
            )

            nearest_cluster = int(
                core_labels[nearest_index]
            )

            st.subheader("Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Distance to Nearest Core Sample",
                f"{nearest_distance:.4f}"
            )

            result_col2.metric(
                "DBSCAN eps Threshold",
                f"{dbscan_model.eps:.3f}"
            )

            if nearest_distance <= dbscan_model.eps:

                st.success(
                    f"Approximate segment assignment: "
                    f"Cluster {nearest_cluster}"
                )

                st.write(
                    "The customer is within eps distance of "
                    "a DBSCAN core sample."
                )

            else:

                st.warning(
                    "This customer is not within eps distance "
                    "of any DBSCAN core sample. Treat this "
                    "customer as unassigned/noise."
                )

                st.write(
                    "The model has not assigned this customer "
                    "to an existing segment."
                )

        except Exception as error:
            st.error(
                "Unable to process this customer. "
                "Check that the saved scaler matches the "
                "preprocessing used during model training."
            )
            st.code(str(error))


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "RFM Customer Segmentation | DBSCAN | "
    "Built with Python, scikit-learn and Streamlit"
)
