from loguru import logger
import typer

app = typer.Typer()

import pandas as pd

class PreprocessingData:

    def __init__(self, csv_path, column_names, na_value):
        self.csv_path = csv_path
        self.column_names = column_names
        self.na_value = na_value

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

    def execute(self):
        data_raw = self.load_dataset(
            self.csv_path,
            self.column_names,
            self.na_value
        )

        return data_raw
