"""Streamlit dashboard for the churn-risk API.
Run with: streamlit run streamlit_app.py
The FastAPI backend must be running separately: uvicorn app.main:app --reload
"""
import pandas as pd
import requests
import streamlit as st

from app.core.config import CUSTOMER_DATA_PATH, MAX_BATCH_SIZE

DEFAULT_BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Churn Risk Dashboard", page_icon="\U0001F4C9", layout="wide")


@st.cache_data(show_spinner=False)
def load_customer_attributes() -> pd.DataFrame:
    """Contract/PaymentMethod/tenure/MonthlyCharges aren't part of the
    API's response (see app/schemas/score.py's ScoreResponse -- it only
    returns customer_id, churn_probability, risk_tier). Pulled here from
    the same local customer data source the API itself scores from
    (CUSTOMER_DATA_PATH), then joined onto the API's output by
    customerID, purely for display/filtering in this dashboard.
    """
    df = pd.read_csv(CUSTOMER_DATA_PATH)
    return df[["customerID", "Contract", "PaymentMethod", "tenure", "MonthlyCharges"]]


def fetch_scores(backend_url: str, customer_ids: list[str]) -> pd.DataFrame:
    """POST to /batch-score."""
    response = requests.post(
        f"{backend_url}/batch-score",
        json={"customer_ids": customer_ids},
        timeout=30,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


st.title("Churn Risk Dashboard")
st.caption("Live scores from the churn-risk FastAPI backend -- not a static export.")

with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    batch_size = st.slider(
        "Customers to score",
        min_value=10,
        max_value=MAX_BATCH_SIZE,
        value=200,
        step=10,
        help=f"Capped at {MAX_BATCH_SIZE} -- the same MAX_BATCH_SIZE the API itself enforces (app/core/config.py).",
    )
    fetch_clicked = st.button("Score customers", type="primary")

if fetch_clicked or "scores_df" not in st.session_state:
    attrs = load_customer_attributes()
    sample_ids = attrs["customerID"].head(batch_size).tolist()
    try:
        with st.spinner(f"Scoring {len(sample_ids)} customers via {backend_url}/batch-score ..."):
            scores = fetch_scores(backend_url, sample_ids)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the backend at **{backend_url}**. Is it running? "
            "Start it with `uvicorn app.main:app --reload` in another terminal."
        )
        st.stop()
    except requests.exceptions.HTTPError as exc:
        st.error(f"Backend returned an error: {exc.response.status_code} -- {exc.response.text}")
        st.stop()

    merged = scores.merge(
        attrs, left_on="customer_id", right_on="customerID", how="left"
    ).drop(columns=["customerID"])
    st.session_state["scores_df"] = merged

df = st.session_state.get("scores_df")
if df is None:
    st.info("Click **Score customers** in the sidebar to fetch live scores from the API.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    tier_filter = st.multiselect(
        "Risk tier", options=["high", "medium", "low"], default=["high", "medium", "low"]
    )
    contract_filter = st.multiselect(
        "Contract", options=sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique())
    )

filtered = df[df["risk_tier"].isin(tier_filter) & df["Contract"].isin(contract_filter)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers scored", len(df))
col2.metric("Shown after filters", len(filtered))
col3.metric(
    "Avg churn probability",
    f"{filtered['churn_probability'].mean():.1%}" if len(filtered) else "--",
)
col4.metric("High risk (filtered)", int((filtered["risk_tier"] == "high").sum()))

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Avg churn probability by Contract")
    if len(filtered):
        st.bar_chart(filtered.groupby("Contract")["churn_probability"].mean().sort_values(ascending=False))
    else:
        st.caption("No customers match the current filters.")
with chart_col2:
    st.subheader("Avg churn probability by Payment Method")
    if len(filtered):
        st.bar_chart(filtered.groupby("PaymentMethod")["churn_probability"].mean().sort_values(ascending=False))
    else:
        st.caption("No customers match the current filters.")

st.subheader("Customers, sorted by churn probability")
display_df = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)
st.dataframe(
    display_df[
        ["customer_id", "churn_probability", "risk_tier", "Contract", "PaymentMethod", "tenure", "MonthlyCharges"]
    ],
    width="stretch",
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Churn Probability", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "risk_tier": st.column_config.TextColumn("Risk Tier"),
    },
)
