import pandas as pd

# filter.py
class FilterTask:
    def __init__(self, column_name, threshold, filter_in=True):
        self.column_name = column_name
        self.threshold = threshold
        self.filter_in = filter_in

    def task(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' does not exist in the DataFrame.")
        
        if self.filter_in:
            return df[df[self.column_name] < self.threshold]
        else:
            return df[df[self.column_name] >= self.threshold]