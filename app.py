import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.sidebar.title("Filters")

contract_filter = st.sidebar.selectbox(
    "Select Contract Type",
    ["All"] + list(df["Contract"].unique())
)

if contract_filter != "All":
    df = df[df["Contract"] == contract_filter]

st.title("📊 Customer Churn Analysis Dashboard")

# KPI Calculations
total_customers = len(df)
churned = len(df[df["Churn"] == "Yes"])
retained = len(df[df["Churn"] == "No"])
churn_rate = (churned / total_customers) * 100

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Retained Customers", retained)
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Churn Distribution")
st.bar_chart(df["Churn"].value_counts())

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

st.subheader("Contract Type vs Churn")

st.bar_chart(contract_churn)

avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()

st.subheader("Average Monthly Charges by Churn")

st.bar_chart(avg_charges)

st.markdown("---")

st.header("📌 Key Business Insights")

st.success("""
• Month-to-month customers have the highest churn rate (42.71%).

• Two-year contract customers are the most loyal (2.83% churn rate).

• Churned customers pay higher monthly charges on average.

• Monthly Charges, Total Charges, and Tenure are the top churn factors identified by the Machine Learning model.

• Random Forest model achieved 78.54% prediction accuracy.
""")