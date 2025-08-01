from loguru import logger
import pandas as pd


class PreprocessingData:

    def __init__(self, csv_path, column_names, na_value, dict_to_map, target_column):
        self.csv_path = csv_path
        self.column_names = column_names
        self.na_value = na_value
        self.dict_to_map = dict_to_map
        self.target_colum = target_column

    @staticmethod
    def load_dataset(csv_path: str, column_names=None, na_value: str = " ?"):
        """
        Load a CSV dataset into a pandas DataFrame.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        column_names : list, optional
            List of column names. If None, the CSV header is used.
        na_value : str, optional
            Value to be interpreted as NaN (default: " ?").

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        logger.info('Leyendo data')
        df = pd.read_csv(
            csv_path,
            header=None if column_names else "infer",  # If column names provided, no header in CSV
            names=column_names,
            na_values=na_value,
            skipinitialspace=True  # Strips leading spaces in values
        )
        logger.info('Data raw leido')

        return df

    @staticmethod
    def map_income_dataset(dict_map: dict, df: pd.DataFrame, target_col: str = "income") -> pd.DataFrame:
        """
        Map the income target column ['<=50K', '>50K'] to numeric 0/1 directly in the dataset.

        Parameters
        ----------
        dict_map: dic
            Dictionary to map old values to new values in the target column.
        df : pd.DataFrame
            Original dataset with the target column.
        target_col : str, optional
            Name of the target column to transform (default 'income').

        Returns
        -------
        pd.DataFrame
            Dataset with target column mapped to 0 and 1.
        """
        df[target_col] = df[target_col].map(dict_map)
        return df

    def execute(self):
        data_raw = self.load_dataset(
            self.csv_path,
            self.column_names,
            self.na_value
        )

        data_process = self.map_income_dataset(
            self.dict_to_map,
            data_raw,
            self.target_colum
        )

        return data_process
