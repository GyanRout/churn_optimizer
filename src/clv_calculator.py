import numpy as np
import pandas as pd
import logging
from typing import Dict

# Configure production-standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLVCalculator:
    """
    Translates raw survival probabilities into financial metrics.
    Calculates Expected Remaining Lifetime via numerical integration and computes operational ROI.
    """
    
    def __init__(self, discount_rate_annual: float = 0.0):
        """
        Initializes the financial calculator. 
        :param discount_rate_annual: The annual discount rate to calculate Net Present Value (NPV) of future cash flows.
                                     Defaulted to 0.0 for baseline modeling.
        """
        self.discount_rate_monthly = discount_rate_annual / 12.0

    def calculate_expected_lifetime(self, survival_function: pd.Series) -> float:
        """
        Calculates expected lifetime by finding the area under the survival curve (AUC).
        
        :param survival_function: Series where the index represents time (months) and values are survival probabilities.
        :return: Expected lifetime in months.
        """
        if survival_function.empty:
            logger.error("Provided survival function is empty.")
            raise ValueError("Survival function cannot be empty.")

        # Extract time periods (x-axis) and probabilities (y-axis)
        times = survival_function.index.values
        probabilities = survival_function.values
        
        # Numerical integration using the Trapezoidal Rule
        expected_lifetime = np.trapz(y=probabilities, x=times)
        
        return float(expected_lifetime)

    def calculate_roi(self, 
                      expected_lifetime_months: float, 
                      mrr: float, 
                      intervention_cost: float, 
                      retention_probability: float = 1.0) -> Dict[str, float]:
        """
        Computes the Expected ROI of a specific marketing/retention intervention.
        
        :param expected_lifetime_months: The integrated expected remaining tenure.
        :param mrr: Monthly Recurring Revenue for this specific customer.
        :param intervention_cost: The OpEx cost of the retention campaign (e.g., $50 gift card).
        :param retention_probability: The assumed probability (0.0 to 1.0) that the intervention works.
        :return: Dictionary containing financial breakdown.
        """
        if intervention_cost <= 0:
            logger.error("Intervention cost must be strictly positive to calculate ROI.")
            raise ValueError("Intervention cost must be > 0 to avoid ZeroDivisionError.")

        if mrr < 0:
            logger.warning(f"Negative MRR detected (${mrr}). Ensure this represents a valid business state (e.g., refunds).")

        # Baseline CLV calculation
        expected_clv = expected_lifetime_months * mrr
        
        # Factoring in the probability that the intervention actually retains the customer
        retained_value = expected_clv * retention_probability
        
        # Calculate ROI: (Net Profit / Investment)
        net_profit = retained_value - intervention_cost
        roi = net_profit / intervention_cost
        
        return {
            "expected_lifetime_months": round(expected_lifetime_months, 2),
            "expected_clv": round(expected_clv, 2),
            "intervention_cost": round(intervention_cost, 2),
            "expected_roi_percentage": round(roi * 100, 2)
        }

if __name__ == "__main__":
    # Local Unit Test
    calculator = CLVCalculator()
    
    # Mocking a survival curve where probability degrades over 24 months
    mock_times = np.array([0, 6, 12, 18, 24])
    mock_probs = np.array([1.0, 0.8, 0.5, 0.2, 0.05])
    mock_survival_curve = pd.Series(data=mock_probs, index=mock_times)
    
    expected_life = calculator.calculate_expected_lifetime(mock_survival_curve)
    
    # Simulating a user with $150 MRR and a $50 retention budget
    financials = calculator.calculate_roi(
        expected_lifetime_months=expected_life,
        mrr=150.0,
        intervention_cost=50.0,
        retention_probability=0.5  # Assuming a 50% chance the intervention works
    )
    
    print("\nFinancial Evaluation:")
    for key, value in financials.items():
        print(f"{key}: {value}")