# rfh_testdatascience_1

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prueba de nivel posición Capgemini. Modelo de clasificación.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── prod           <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         rfh_testdatascience_1 and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── rfh_testdatascience_1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes rfh_testdatascience_1 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── main.py                 <- Code to create features for modeling
    │
    ├── main_prod.py            <- Code to create features for modeling
    │
    ├── pipeline.py             <- Code to create features for modeling
    │
    └── train.py                <- Code to create visualizations
```

--------

## Python Version
- Python 3.12.10

---

## Environment Setup
Install all dependencies using:

```bash
pip install -r requirements.txt
```
### Train a Model (`main.py`)

The training script `main.py` accepts **7 arguments**:

1. **`model_name_arg` (str)**  
   Model name to be used or saved.

2. **`processing_ind` (bool)**  
   Whether to process the input dataset:  
   - Handle null values  
   - Perform feature engineering  
   - Perform feature selection

3. **`oversampling_ind` (bool)**  
   Whether to apply oversampling to handle class imbalance.

4. **`grid_search_ind_arg` (bool)**  
   Whether to perform hyperparameter tuning via GridSearch.

5. **`test_arg` (bool)**  
   Whether to evaluate the metrics of a specific model.

6. **`learning_rate` (float [0-1])**  
   Learning rate for the model.

7. **`max_depth` (int)**  
   Maximum depth of the model.

8. **`n_estimators` (int)**  
   Number of estimators (trees) for the model.


### Model Saving

If **`grid_search_ind_arg = False`** and **`test_arg = False`**,  
the trained model is **saved** in the `models/` folder using the provided **`model_name_arg`**.

### Production (`main_prod.py`)

The production script `main_prod.py` accepts **1 argument**:

1. **Model name** to use for predictions.

#### Requirements for production
- Place the data to analyze in `data/prod/`.
- The script will return **Accuracy** and **F1-score** for the given model.

### 🖥️ Example Commands

#### Train a model with preprocessing and save it
```bash
python main.py --model_name_arg xgb_model --processing_ind True --oversampling_ind True --grid_search_ind_arg False --test_arg False --learning_rate 0.1 --max_depth 6 --n_estimators 200
```

#### Evaluate an existing model in production
```bash
python main_prod.py --model_name_arg xgb_model
```