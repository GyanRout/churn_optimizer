import streamlit as st
import pandas as pd
import sys
import pathlib

# Ensure the src directory is in the Python path for imports
current_dir = pathlib.Path(__file__).parent.resolve()
src_dir = current_dir / 'src'
sys.path.append(str(src_dir))

from inference import ChurnInferenceEngine

# Configure page aesthetics and layout
st.set_page_config(
    page_title="Prescriptive Churn Optimizer",
    page_icon="📊",
    layout="wide"
)

# --- ENTERPRISE CACHING PATTERN ---
# Streamlit re-runs this entire script from top to bottom on EVERY user interaction.
# If we do not cache the engine, it will perform disk I/O to load the .pkl files 
# every time a slider is moved, causing massive latency.
@st.cache_resource
def load_inference_engine():
    return ChurnInferenceEngine()

try:
    engine = load_inference_engine()
except Exception as e:
    st.error(f"Critical System Failure: Unable to initialize Inference Engine. Error: {e}")
    st.stop()

# --- UI LAYOUT: SIDEBAR (INPUTS) ---
st.sidebar.header("Customer Profile")

age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
mrr = st.sidebar.number_input("Monthly Recurring Revenue (MRR) $", min_value=10.0, value=150.0)
support_tickets = st.sidebar.slider("Support Tickets (Last 30 Days)", min_value=0, max_value=20, value=2)
login_frequency = st.sidebar.slider("Login Frequency (Days/Month)", min_value=0, max_value=30, value=15)
contract_type = st.sidebar.selectbox("Contract Type", options=["Annual", "Monthly"])

st.sidebar.markdown("---")
st.sidebar.header("OpEx Parameters")
intervention_cost = st.sidebar.number_input("Intervention Budget (Cost to Retain) $", min_value=5.0, value=50.0)
retention_prob = st.sidebar.slider("Est. Retention Probability", min_value=0.01, max_value=1.0, value=0.50, help="Probability that this intervention actually stops the churn.")
roi_hurdle = st.sidebar.number_input("Minimum ROI Hurdle Rate (%)", min_value=0.0, value=15.0)

# Map UI selection to the model's expected binary format
contract_monthly = 1 if contract_type == "Monthly" else 0

# Construct the payload
payload = {
    "age": age,
    "mrr": mrr,
    "support_tickets": support_tickets,
    "login_frequency": login_frequency,
    "contract_monthly": contract_monthly
}

# --- UI LAYOUT: MAIN STAGE (OUTPUTS) ---
st.title("Prescriptive Customer Churn Optimizer")
st.markdown("""
This system evaluates the survival risk of a customer and prescribes retention strategies based on Customer Lifetime Value (CLV) and Return on Investment (ROI) targets.
""")

# Execute Inference
with st.spinner("Calculating hazard functions and financial metrics..."):
    result = engine.generate_prescription(
        payload=payload,
        intervention_cost=intervention_cost,
        retention_prob=retention_prob,
        roi_hurdle_rate=roi_hurdle
    )

prescription = result["prescription"]
metrics = prescription["metrics"]

# Display Prescriptive Action
st.subheader("Actionable Prescription")
if prescription["recommend_intervention"]:
    st.success("✅ **RECOMMENDATION: INTERVENE**")
else:
    st.error("🚫 **RECOMMENDATION: DO NOT INTERVENE (Let Churn)**")

st.info(f"**Reasoning:** {prescription['reasoning']}")

# Display Financial Metrics in a grid
st.markdown("---")
st.subheader("Financial Breakdown")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Expected Lifetime", value=f"{metrics['expected_lifetime_months']:.1f} mo")
with col2:
    st.metric(label="Expected CLV", value=f"${metrics['expected_clv']:,.2f}")
with col3:
    st.metric(label="Intervention Cost", value=f"${metrics['intervention_cost']:,.2f}")
with col4:
    # Color-code the ROI based on whether it passes the hurdle
    delta_color = "normal" if metrics['expected_roi_percentage'] >= roi_hurdle else "inverse"
    st.metric(
        label="Expected ROI", 
        value=f"{metrics['expected_roi_percentage']:,.1f}%", 
        delta=f"Hurdle: {roi_hurdle}%",
        delta_color=delta_color
    )