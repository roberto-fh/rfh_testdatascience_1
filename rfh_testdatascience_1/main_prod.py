from loguru import logger
import joblib
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score

from rfh_testdatascience_1.config import prod_path, column_names, na_value, dict_map_test, target_column
from rfh_testdatascience_1.dataset import PreprocessingData


def main(
        model_name_arg,
):
    with open(f"../models/{model_name_arg}.pkl", "rb") as f:
        pipeline = joblib.load(f)

    reader_data = PreprocessingData(
        csv_path=prod_path,
        column_names=column_names,
        na_value=na_value,
        dict_to_map=dict_map_test,
        target_column=target_column
    )

    df = reader_data.execute()
    df = df.iloc[1:].reset_index(drop=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    y_pred = pipeline.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("\n=== Resultados del modelo ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y, y_pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prod XGBoost Predict")

    parser.add_argument(
        "--model-name-arg",
        type=str,
        default="xgb",
        help="Model name"
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="xgb",
        help="Data to predict"
    )

    main(
        model_name_arg='test3'
    )