from pathlib import Path

from loguru import logger
from tqdm import tqdm

from rfh_testdatascience_1.config import train_path, column_names, na_value

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pipeline import create_xgb_pipeline
from rfh_testdatascience_1.dataset import PreprocessingData



def train_and_evaluate(df, target_col: str,
                       use_preprocessing=True, use_balancing=False):
    """
    Entrena y evalúa un pipeline de XGBoost.
    """
    # 🔹 Cargar dataset
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 🔹 Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 Identificar columnas
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 🔹 Crear pipeline
    pipeline = create_xgb_pipeline(
        numeric_features, categorical_features,
        use_preprocessing=use_preprocessing,
        use_balancing=use_balancing
    )

    # 🔹 Entrenar (fit)
    pipeline.fit(X_train, y_train)

    # 🔹 Predecir
    y_pred = pipeline.predict(X_test)

    # 🔹 Evaluar
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Resultados del modelo ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return pipeline, acc, f1


# Ejemplo de uso:
if __name__ == "__main__":

    preprocess = PreprocessingData(
        csv_path=train_path,
        column_names=column_names,
        na_value=na_value
    )

    df = preprocess.execute()

    train_and_evaluate(
        df=df,
        target_col="income",
        use_preprocessing=True,
        use_balancing=True
    )