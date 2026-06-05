import streamlit as st
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# Load Files
df = pd.read_csv(
    "clustered_data.csv"
)

gmm = pickle.load(
    open(
        "gmm_model.pkl",
        "rb"
    )
)

scaler = pickle.load(
    open(
        "scaler.pkl",
        "rb"
    )
)

st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

st.title(
    "🛍 Mall Customer Segmentation using GMM"
)

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dataset Overview",
        "Visualizations",
        "Prediction"
    ]
)

# ==================================
# DATASET OVERVIEW
# ==================================

if menu == "Dataset Overview":

    st.header("Dataset Overview")

    st.write(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ==================================
# VISUALIZATIONS
# ==================================

elif menu == "Visualizations":

    st.header("Customer Analysis")

    fig1, ax1 = plt.subplots()

    sns.countplot(
        x='Gender',
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()

    sns.histplot(
        df['Age'],
        bins=20,
        ax=ax2
    )

    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set1',
        data=df,
        ax=ax3
    )

    st.pyplot(fig3)

# ==================================
# PREDICTION
# ==================================

else:

    st.header(
        "Customer Cluster Prediction"
    )

    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        50
    )

    spending = st.slider(
        "Spending Score (1-100)",
        1,
        100,
        50
    )

    if st.button("Predict"):

        customer = scaler.transform(
            [[income, spending]]
        )

        cluster = gmm.predict(
            customer
        )[0]

        probability = max(
            gmm.predict_proba(
                customer
            )[0]
        )

        st.success(
            f"Predicted Cluster: {cluster}"
        )

        st.info(
            f"Confidence: {probability:.2%}"
        )

        if cluster == 0:
            st.write(
                "Likely Budget Customers"
            )

        elif cluster == 1:
            st.write(
                "Likely Premium Customers"
            )

        elif cluster == 2:
            st.write(
                "Average Customers"
            )

        elif cluster == 3:
            st.write(
                "High Income Low Spending"
            )

        else:
            st.write(
                "High Spending Customers"
            )