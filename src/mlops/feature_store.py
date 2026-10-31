import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from config.config import MODELS_DIR, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
from config.logging_config import get_logger

logger = get_logger("FeatureStore")

# Feature definitions for academic risk prediction
NUMERICAL_FEATURES: List[str] = [
    "age_at_enrollment",
    "admission_grade",
    "sem1_enrolled",
    "sem1_approved",
    "sem1_grade",
    "sem2_enrolled",
    "sem2_approved",
    "sem2_grade",
    "attendance_rate",
    "internal_marks_avg",
    "assignment_completion_rate",
    "lms_activity_score",
    "approval_rate_overall",
    "grade_progression"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "attendance_mode",
    "course_name"
]

TARGET_COLUMN = "is_dropout"

class AcademicFeatureStore:
    """
    Manages feature preparation, stratified splits, and the scikit-learn preprocessing pipeline.
    """

    def __init__(self):
        self.preprocessor = None
        self.feature_names_out = None

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs a robust ColumnTransformer for numerical scaling and categorical encoding."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, NUMERICAL_FEATURES),
                ("cat", cat_pipeline, CATEGORICAL_FEATURES)
            ]
        )
        self.preprocessor = preprocessor
        return preprocessor

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits dataset into stratified Train (70%), Validation (15%), and Test (15%) sets,
        fits the preprocessor on Train, and transforms all partitions.
        """
        logger.info(f"Preparing dataset with {len(df)} samples...")
        
        # Verify required columns exist
        available_num = [c for c in NUMERICAL_FEATURES if c in df.columns]
        available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        
        X = df[available_num + available_cat]
        y = df[TARGET_COLUMN].values

        # Stratified train/temp split (70% train, 30% temp)
        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
        )

        # Stratified split of temp into Val (50% of 30% = 15%) and Test (15%)
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
        )

        # Fit preprocessor on Train only
        self.build_preprocessor()
        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_val = self.preprocessor.transform(X_val_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        # Extract generated feature names
        try:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(cat_encoder.get_feature_names_out(available_cat))
            self.feature_names_out = available_num + cat_names
        except Exception:
            self.feature_names_out = [f"feat_{i}" for i in range(X_train.shape[1])]

        # Save preprocessor artifact
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor fitted and saved to {preprocessor_path}")

        logger.info(f"Data partitioning complete: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_feature_names(self) -> List[str]:
        return self.feature_names_out or []
