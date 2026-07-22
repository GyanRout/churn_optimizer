import pandas as pd
import joblib
import logging
from typing import Dict, Any

# Configure production-standard logging rather than print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles inference-time data validation and transformation.
    Ensures input payloads match training-time distributions via saved artifacts.
    """
    def __init__(self, scaler_path: str = 'models/scaler.pkl'):
        """
        Initializes the processor by loading the fitted StandardScaler.
        """
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Successfully loaded scaling artifact from {scaler_path}")
        except FileNotFoundError:
            logger.error(f"Critical Error: Scaler artifact not found at {scaler_path}. Ensure it is downloaded from Colab.")
            raise

        # Strictly define the schema expected by the Cox model
        self.features_to_scale = ['age', 'mrr', 'support_tickets', 'login_frequency']
        self.categorical_features = ['contract_monthly']
        self.expected_schema = self.features_to_scale + self.categorical_features

    def preprocess(self, raw_payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates and transforms a raw JSON/Dict payload into a model-ready DataFrame.
        """
        # 1. Schema Validation (Fail fast if data is missing)
        missing_features = [feat for feat in self.expected_schema if feat not in raw_payload]
        if missing_features:
            error_msg = f"Malformed payload. Missing required features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 2. Type casting to DataFrame
        try:
            df = pd.DataFrame([raw_payload])
        except Exception as e:
            logger.error(f"Failed to parse payload into DataFrame: {e}")
            raise

        # 3. Mathematical Transformation
        # We use .transform(), NEVER .fit_transform() in production.
        try:
            df[self.features_to_scale] = self.scaler.transform(df[self.features_to_scale])
        except ValueError as e:
            logger.error(f"Scaling transformation failed. Check input data types: {e}")
            raise

        # 4. Enforce Column Ordering
        # ML models rely on column order or strict names. Reordering guarantees safety.
        return df[self.expected_schema]

if __name__ == "__main__":
    import pathlib

    # 1. Dynamically locate the directory containing THIS script (src/)
    current_dir = pathlib.Path(__file__).parent.resolve()
    
    # 2. Traverse up to the project root, then into models/scaler.pkl
    project_root = current_dir.parent
    model_path = project_root / 'models' / 'scaler.pkl'
    
    # 3. Initialize processor with the absolute path
    processor = DataProcessor(scaler_path=str(model_path))
    
    test_payload = {
        "age": 45,
        "mrr": 200,
        "support_tickets": 2,
        "login_frequency": 10,
        "contract_monthly": 1
    }
    
    processed_df = processor.preprocess(test_payload)
    print("\nProcessed DataFrame ready for inference:")
    print(processed_df)