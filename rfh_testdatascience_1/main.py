from loguru import logger
import joblib
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score

from rfh_testdatascience_1.config import train_path, column_names, na_value, dict_map_train, target_column, keep_columns
from rfh_testdatascience_1.dataset import PreprocessingData
from rfh_testdatascience_1.train import TrainerModel


def main(
        model_name_arg,
        processing_ind,
        oversampling_ind,
        grid_search_ind_arg,
        test_arg,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200
):
    logger.info('Carga de datos')
    reader_data = PreprocessingData(
        csv_path=train_path,
        column_names=column_names,
        na_value=na_value,
        dict_to_map=dict_map_train,
        target_column=target_column
    )

    df = reader_data.execute()
    logger.info('Entrenamiento del modelo')
    trainer = TrainerModel(
        df=df,
        target_column=target_column,
        processing_ind=processing_ind,
        over_sampling_ind=oversampling_ind,
        keep_column=keep_columns,
        grid_search_ind=grid_search_ind_arg,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators
    )
    X_train, X_test, y_train, y_test, pipeline = trainer.execute()

    if grid_search_ind_arg:
        logger.info('Fin del proceso de Grid search')
        pass

    elif test_arg:
        # Predecir
        y_pred = pipeline.predict(X_test)

        # Evaluar
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n=== Resultados del modelo ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        logger.info('Fin del proceso de testing')

    else:
        joblib.dump(pipeline, f"../models/{model_name_arg}.pkl")
        logger.info(f'El modelo {model_name_arg} ha sido guardado en models/')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train XGBoost pipeline")

    # Add parameters
    parser.add_argument(
        "--model-name-arg",
        type=str,
        default="xgb",
        help="Model name"
    )
    parser.add_argument(
        "--processing-ind",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable Grid Search for hyperparameter tuning"
    )
    parser.add_argument(
        "--oversampling-ind",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable Grid Search for hyperparameter tuning"
    )
    parser.add_argument(
        "--grid-search-ind-arg",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable Grid Search for hyperparameter tuning"
    )
    parser.add_argument(
        "--test-arg",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable test mode"
    )
    parser.add_argument(
        "--prod-arg",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable prod mode"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost (float)")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max depth of XGBoost trees (int)")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds (int)"
    )

    args = parser.parse_args()

    logger.info(args)

    main(
        model_name_arg=args.model_name_arg,
        processing_ind=args.processing_ind,
        oversampling_ind=args.oversampling_ind,
        grid_search_ind_arg=args.grid_search_ind_arg,
        test_arg=args.test_arg,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators
    )