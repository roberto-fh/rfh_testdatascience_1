from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from rfh_testdatascience_1.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def create_xgb_pipeline(numeric_features, categorical_features,
                        use_preprocessing=True, use_balancing=False):
    """
    Crea un pipeline flexible para XGBoost con preprocesamiento y balanceo opcional.
    """
    steps = []

    # 🔹 Preprocesamiento opcional (OHE para categóricas)
    if use_preprocessing:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'  # Deja numéricas tal cual
        )
        steps.append(('preprocessing', preprocessor))

    # 🔹 Balanceo opcional
    if use_balancing:
        steps.append(('oversample', SMOTE(random_state=42)))

    # 🔹 Modelo XGBoost
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )

    steps.append(('model', model))

    return ImbPipeline(steps=steps)