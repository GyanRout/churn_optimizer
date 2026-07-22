import logging
import pathlib
from typing import Dict, Any
import joblib

# Import our custom modules
from data_processor import DataProcessor
from clv_calculator import CLVCalculator

# Standardize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnInferenceEngine:
    """
    Orchestrates the end-to-end inference pipeline.
    Connects raw payload ingestion -> preprocessing -> ML prediction -> financial evaluation -> prescriptive action.
    """
    def __init__(self):
        """
        Initializes all pipeline components and loads model artifacts into memory.
        This ensures disk I/O happens only once at startup, not on every request.
        """
        # Dynamic path resolution to prevent environment drift
        current_dir = pathlib.Path(__file__).parent.resolve()
        project_root = current_dir.parent
        
        model_path = project_root / 'models' / 'cox_ph_model.pkl'
        scaler_path = project_root / 'models' / 'scaler.pkl'
        
        try:
            self.model = joblib.load(model_path)
            logger.info("Cox Proportional Hazards model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Critical ML artifact missing at {model_path}. Pipeline initialization failed.")
            raise

        # Initialize the processor and calculator via Dependency Composition
        self.processor = DataProcessor(scaler_path=str(scaler_path))
        self.clv_calculator = CLVCalculator()

    def generate_prescription(self, 
                              payload: Dict[str, Any], 
                              intervention_cost: float, 
                              retention_prob: float = 0.5,
                              roi_hurdle_rate: float = 10.0,
                              risk_time_horizon: float = 3.0,
                              min_churn_risk: float = 0.20) -> Dict[str, Any]:
        """
        Processes a customer profile and prescribes a business action based on Risk and ROI.
        """
        import numpy as np # Ensure this is imported at the top of inference.py
        
        logger.info(f"Generating prescription for customer payload. Intervention Cost: ${intervention_cost}")

        # 1. Validation & Transformation
        processed_df = self.processor.preprocess(payload)
        
        # 2. ML Inference (Survival Function Generation)
        survival_curves = self.model.predict_survival_function(processed_df)
        customer_curve = survival_curves.iloc[:, 0]
        
        # --- NEW: RISK GATING LOGIC ---
        # Find the closest time index in our survival curve to our target horizon (e.g., 3 months)
        closest_time_idx = np.abs(customer_curve.index - risk_time_horizon).argmin()
        closest_time = customer_curve.index[closest_time_idx]
        
        survival_prob_at_horizon = customer_curve[closest_time]
        churn_risk = 1.0 - survival_prob_at_horizon
        
        # 3. Financial Translation
        expected_lifetime = self.clv_calculator.calculate_expected_lifetime(customer_curve)
        mrr = payload.get('mrr', 0.0)
        
        financials = self.clv_calculator.calculate_roi(
            expected_lifetime_months=expected_lifetime,
            mrr=mrr,
            intervention_cost=intervention_cost,
            retention_probability=retention_prob
        )
        
        # 4. Prescriptive Business Logic
        roi = financials["expected_roi_percentage"]
        
        # A customer must BOTH be at risk of leaving AND be profitable to save
        is_at_risk = churn_risk >= min_churn_risk
        is_profitable = roi >= roi_hurdle_rate
        should_intervene = is_at_risk and is_profitable
        
        # Construct dynamic reasoning
        if not is_at_risk:
            reasoning = f"Customer is safe. Churn risk at {risk_time_horizon} months is {churn_risk:.1%}, failing the {min_churn_risk:.1%} risk threshold. Save budget."
        elif not is_profitable:
            reasoning = f"Customer is at risk ({churn_risk:.1%}), but Expected ROI ({roi}%) fails to clear the hurdle rate of {roi_hurdle_rate}%. Let churn."
        else:
            reasoning = f"Customer is at risk ({churn_risk:.1%}) and Expected ROI ({roi}%) clears the hurdle rate of {roi_hurdle_rate}%. Intervene."

        # Add risk metrics to the output
        financials["churn_risk_3_months"] = round(churn_risk * 100, 2)
        
        return {
            "status": "success",
            "prescription": {
                "recommend_intervention": should_intervene,
                "reasoning": reasoning,
                "metrics": financials
            }
        }

if __name__ == "__main__":
    # Local System Integration Test
    engine = ChurnInferenceEngine()
    
    # Simulating a high-risk customer (High support tickets, low login frequency, monthly contract)
    high_risk_payload = {
        "age": 28,
        "mrr": 300,
        "support_tickets": 5,
        "login_frequency": 2,
        "contract_monthly": 1
    }
    
    # We are considering sending a $100 discount offer
    result = engine.generate_prescription(
        payload=high_risk_payload,
        intervention_cost=100.0,
        retention_prob=0.3, # Low probability because they are highly frustrated
        roi_hurdle_rate=20.0 # Demanding at least 20% ROI to justify the spend
    )
    
    import json
    print("\n--- Final API Response ---")
    print(json.dumps(result, indent=4))