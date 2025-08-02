from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from loguru import logger
from sklearn.preprocessing import FunctionTransformer

from rfh_testdatascience_1.config import continent_map


def fill_cat_missing_func(df, unknown_label):
    return df.fillna(unknown_label)


def add_feature_columns(df: pd.DataFrame, continent_map: dict) -> pd.DataFrame:
    """
    Adds derived columns to the DataFrame:
      1. 'continent' based on 'native-country' and a mapping dictionary
      2. 'education_group' grouped by 'education-num'
      3. 'work_category' categorized by 'hours-per-week'
    """
    df = df.copy()

    # Continent column
    df['continent'] = (
        df['native-country']
        .replace(' United-States', 'United-States')
        .str.strip()
        .replace(continent_map)
    )

    # Education group column
    def group_education(num):
        if num <= 4:
            return 'Primary'
        elif num <= 9:
            return 'Secondary'
        elif num <= 12:
            return 'University'
        else:
            return 'Postgraduate'

    df['education_group'] = df['education-num'].apply(group_education)

    # Work category column
    def categorize_hours(h):
        if h < 35:
            return 'Part-time'
        elif h <= 50:
            return 'Full-time'
        else:
            return 'Extended'

    df['work_category'] = df['hours-per-week'].apply(categorize_hours)

    return df


def select_columns_func(df, keep_columns):
    return df[keep_columns]


class PipelineModel:
    def __init__(self,
                 categorical_features,
                 keep_columns,
                 processing_ind=False,
                 over_sampling_ind=False,
                 learning_rate=0.1,
                 max_depth=5,
                 n_estimators=200):
        self.steps = []
        self.categorical_features = categorical_features
        self.keep_columns = keep_columns
        self.processing_ind = processing_ind
        self.over_sampling_ind = over_sampling_ind
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    @staticmethod
    def fill_cat_variable():
        transformer = FunctionTransformer(
            fill_cat_missing_func,
            validate=False,
            kw_args={"unknown_label": "unknown"}
        )
        return 'fill_cat_missing_transformer', transformer

    @staticmethod
    def add_features():
        transformer = FunctionTransformer(
            add_feature_columns,
            validate=False,
            kw_args={"continent_map": continent_map}
        )
        return 'add_features_transformer', transformer

    def select_columns_important(self):
        transformer = FunctionTransformer(
            select_columns_func,
            validate=False,
            kw_args={"keep_columns": self.keep_columns}
        )
        return 'select_columns_transformer', transformer

    def one_hot_encoding(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        return 'one_hot_enconding', preprocessor

    def over_sampling(self):
        return 'oversample', SMOTE(random_state=42)

    def execute(self):
        if self.processing_ind:
            self.steps.append(self.fill_cat_variable())
            self.steps.append(self.add_features())
            self.steps.append(self.select_columns_important())
            self.categorical_features = ['workclass', 'education_group', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'continent', 'work_category']
        self.steps.append(self.one_hot_encoding())
        if self.over_sampling_ind:
            self.steps.append(self.over_sampling())

        # XGBoost Model
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            eval_metric='logloss'
        )

        self.steps.append(('model', model))

        return ImbPipeline(steps=self.steps)