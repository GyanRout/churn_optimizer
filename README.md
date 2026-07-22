# churn_optimizer# Prescriptive Customer Churn & OpEx Optimizer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B.svg)](https://streamlit.io/)
[![Lifelines](https://img.shields.io/badge/Lifelines-0.28.0-lightgrey.svg)](https://lifelines.readthedocs.io/en/latest/)

## Business Objective
Standard churn prediction models formulate the problem as a binary classification task, which inherently fails to account for right-censoring and fails to prescribe actionable business strategies. 

This system optimizes operational expenditure (OpEx) by combining time-to-event survival analysis with financial integration. It evaluates a customer's baseline risk of churning and their Expected Customer Lifetime Value (CLV) to dynamically prescribe retention interventions only when the Expected ROI clears a strictly defined corporate hurdle rate.

## Mathematical Architecture

1. **Time-to-Event Modeling:** Utilizes a Cox Proportional Hazards (CPH) model with L2 regularization to estimate the survival function $S(t|X)$. 
2. **Financial Translation:** Expected remaining lifetime is computed via numerical integration of the discrete survival probabilities using the Trapezoidal rule:
   $$E[T] = \int_{0}^{\infty} S(t) dt$$
3. **Risk-Gated Prescriptive Logic:** The system strictly rejects retention budgets for accounts where the baseline risk of churn within a 3-month horizon is below 20%, preserving capital.

## System Design
The pipeline strictly enforces a Separation of Concerns (SoC).
* **`notebooks/`**: Handles synthetic data generation, preprocessing artifact creation, and model serialization (`joblib`).
* **`src/data_processor.py`**: Object-oriented payload validation and inference-time mathematical transformations.
* **`src/clv_calculator.py`**: Isolated financial logic evaluating net profitability and ROI limits.
* **`src/inference.py`**: The central orchestration engine enforcing dynamic path resolution to prevent environment drift.
* **`app.py`**: A stateless Streamlit frontend employing `@st.cache_resource` for singleton model loading, eliminating disk I/O latency during user interaction.

## Environment & Setup

To guarantee reproducible builds and avoid transitive dependency failures (specifically the deprecation of `scipy.integrate.trapz`), you must use the pinned `requirements.txt`.

```bash
# 1. Clone the repository
git clone [https://github.com/GyanRout/churn_optimizer.git](https://github.com/GyanRout/churn_optimizer.git)
cd churn_optimizer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install strict dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py