import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# LOAD DATA
df = pd.read_csv("online_retail.csv", encoding="latin1")

df.dropna(subset=["CustomerID"], inplace=True)

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# LOAD MODELS
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
similarity_df = joblib.load("models/product_similarity.pkl")
product_names = joblib.load("models/product_names.pkl")


st.markdown("""
<style>

/* FONT */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Gill Sans', 'Trebuchet MS', sans-serif;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #FFB74D 0%,
        #FF9800 50%,
        #F57C00 100%
    );
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* RADIO BUTTON TEXT */
.stRadio label{
    font-size:18px !important;
    font-weight:600 !important;
}

/* BUTTONS */
.stButton > button{
    background-color:#FF8C00;
    color:white;
    border-radius:10px;
    border:none;
    font-weight:bold;
}

/* METRIC CARDS */
[data-testid="metric-container"]{
    background:white;
    border-left:5px solid #FF8C00;
    padding:15px;
    border-radius:10px;
}

/* TITLES */
h1{
    color:#E65100 !important;
}

</style>
""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.markdown(
"""
<div style='text-align:center;'>
<img src='https://cdn-icons-png.flaticon.com/512/3081/3081559.png'
width='180'>
</div>
<h1 style='
text-align:center;
color:white;
font-size:18px;
font-weight:800;
'>
🛒 Shopper Spectrum
</h1>
""",
unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "EDA Dashboard",
        "👥 Customer Segmentation",
        "🛒 Product Recommendation"
    ]
)
# HOME PAGE

# HOME PAGE
if menu == "Home":
    
    st.markdown("""
<h1 style='text-align:center;color:#2563EB;'>
🛒 Shopper Spectrum
</h1>

<h4 style='text-align:center;'>
Customer Segmentation and Product Recommendation System
</h4>
""", unsafe_allow_html=True)
    
    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.success("💎 High Value")

    with col2:
        st.info("👤 Regular")

    with col3:
        st.warning("🛒 Occasional")

    with col4:
        st.error("⚠ At Risk")


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Transactions", len(df))

    with c2:
        st.metric("Customers", df["CustomerID"].nunique())

    with c3:
        st.metric("Products", df["Description"].nunique())

    with c4:
        st.metric(
            "Revenue",
            f"£{df['TotalAmount'].sum():,.0f}"
        )

    st.markdown("---")

    monthly = (
        df.groupby(
            df["InvoiceDate"].dt.to_period("M")
        )["TotalAmount"]
        .sum()
        .reset_index()
    )

    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="TotalAmount",
        title="Revenue Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##  Statistics")

    st.write(
    f"""
    Total Transactions: {len(df):,}

    Total Revenue: £{df['TotalAmount'].sum():,.0f}

    Customers: {df['CustomerID'].nunique():,}

    Products: {df['Description'].nunique():,}
    """
    )



# CUSTOMER SEGMENTATION
elif menu == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=50
    )

    frequency = st.number_input(
        "Frequency",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        user_data = scaler.transform(
            [[recency, frequency, monetary]]
        )

        cluster = kmeans.predict(user_data)[0]

        st.markdown(
        f"""
        <div style="
        background:linear-gradient(90deg,#FFB74D,#FF8C00);
        color:white;
        padding:20px;
        border-radius:12px;
        font-size:22px;
        font-weight:bold;
        text-align:center;">
        Predicted Cluster: {cluster}
        </div>
        """,
        unsafe_allow_html=True
    )

# PRODUCT RECOMMENDATION
elif menu == "🛒 Product Recommendation":

    st.title("🛒 Product Recommendation")

    product = st.selectbox(
        "Select Product",
        product_names
    )

    if st.button("Get Recommendations"):

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.subheader("Recommended Products")

        for item in recommendations.index:

         st.markdown(
            f"""
            <div style="
            background:linear-gradient(90deg,#FFB74D,#FF8C00);
            color:white;
            padding:18px;
            border-radius:12px;
            margin-bottom:12px;
            box-shadow:0px 4px 10px rgba(0,0,0,0.15);
            font-size:18px;
            font-weight:bold;">
            🎁 {item}
            </div>
            """,
            unsafe_allow_html=True
        )

# EDA DASHBOARD
elif menu == "EDA Dashboard":

    st.title("EDA Dashboard")

    country = (
        df.groupby("Country")["InvoiceNo"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )

    fig1 = px.bar(
        country,
        title="Top Countries by Transactions"
    )

    st.plotly_chart(fig1, use_container_width=True)

    products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig2 = px.bar(
        products,
        title="Top Selling Products"
    )

    st.plotly_chart(fig2, use_container_width=True)

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly_sales = df.groupby("Month")["TotalAmount"].sum().reset_index()

    fig3 = px.line(
        monthly_sales,
        x="Month",
        y="TotalAmount",
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig3, use_container_width=True)

    top_customers = (
        df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig4 = px.bar(
        top_customers,
        x="CustomerID",
        y="TotalAmount",
        title="Top 10 Customers by Spending"
    )

    st.plotly_chart(fig4, use_container_width=True)

    daily_transactions = (
        df.groupby(df["InvoiceDate"].dt.date)["InvoiceNo"]
        .count()
        .reset_index()
    )

    daily_transactions.columns = ["Date", "Transactions"]

    fig8 = px.line(
        daily_transactions,
        x="Date",
        y="Transactions",
        title="Daily Transactions Trend"
    )

    st.plotly_chart(fig8, use_container_width=True)

    