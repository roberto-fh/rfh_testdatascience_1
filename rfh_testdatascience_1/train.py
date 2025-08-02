from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from rfh_testdatascience_1.pipeline import PipelineModel
from rfh_testdatascience_1.config import param_grid


class TrainerModel:

    def __init__(self,
                 df,
                 target_column,
                 processing_ind,
                 over_sampling_ind,
                 keep_column,
                 grid_search_ind,
                 learning_rate,
                 max_depth,
                 n_estimators):
        self.df = df
        self.target_column = target_column
        self.processing_ind = processing_ind
        self.over_sampling_ind = over_sampling_ind
        self.keep_column = keep_column
        self.grid_search_ind = grid_search_ind
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def split_target(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        return X, y

    def categorical_columns(self):
        return self.df.select_dtypes(include=["object", "category"]).columns.tolist()

    def execute(self):
        X, y = self.split_target()
        categorical_features = self.categorical_columns()

        logger.info('Aplicando transformaciones a los datos')
        transformer = PipelineModel(
            categorical_features,
            keep_columns=self.keep_column,
            processing_ind=self.processing_ind,
            over_sampling_ind=self.over_sampling_ind,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
        )

        pipeline = transformer.execute()

        logger.info('Separación test y train')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.grid_search_ind:
            logger.info('Inicio grid search')
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='f1',
                cv=5,
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X_train, y_train)
            print("Best params:", grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            return X_train, X_test, y_train, y_test, grid_search
        else:
            logger.info('Ajustando pipeline')
            pipeline.fit(X_train, y_train)
            return X_train, X_test, y_train, y_test, pipeline