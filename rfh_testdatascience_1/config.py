from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Paths a los archivos
train_path = RAW_DATA_DIR / "adult.data"
test_path = RAW_DATA_DIR / "adult.test"

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

na_value = '?'

dict_map_train = {
    '<=50K': 0,
    '>50K': 1
}

dict_map_test = {
    '<=50K.': 0,
    '>50K.': 1
}

target_column = 'income'

continent_map = {
    # América
    'Canada': 'America',
    'Cuba': 'America',
    'Jamaica': 'America',
    'Mexico': 'America',
    'Puerto-Rico': 'America',
    'Honduras': 'America',
    'Columbia': 'America',
    'Haiti': 'America',
    'Dominican-Republic': 'America',
    'El-Salvador': 'America',
    'Guatemala': 'America',
    'Trinadad&Tobago': 'America',
    'Nicaragua': 'America',
    'Outlying-US(Guam-USVI-etc)': 'America',
    'Ecuador': 'America',
    'Peru': 'America',
    'South': 'America',

    # Europa
    'England': 'Europe',
    'Germany': 'Europe',
    'Italy': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'France': 'Europe',
    'Yugoslavia': 'Europe',
    'Scotland': 'Europe',
    'Greece': 'Europe',
    'Ireland': 'Europe',
    'Hungary': 'Europe',
    'Holand-Netherlands': 'Europe',

    # Asia
    'India': 'Asia',
    'Iran': 'Asia',
    'Philippines': 'Asia',
    'Cambodia': 'Asia',
    'Thailand': 'Asia',
    'Laos': 'Asia',
    'Taiwan': 'Asia',
    'China': 'Asia',
    'Japan': 'Asia',
    'Vietnam': 'Asia',
    'Hong': 'Asia',  # Hong Kong
}

keep_columns = [
    'age',
    'workclass',
    'fnlwgt',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'continent',
    'education_group',
    'work_category'
]

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1]
}