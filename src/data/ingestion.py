import pandas as pd
import dvc.api

class DataIngestor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        
    def load_data(self) -> pd.DataFrame:
        with dvc.api.open(self.data_path, repo='.') as fd:
            self.raw_data = pd.read_csv(fd)
        return self.raw_data
    
    def validate_data(self) -> bool:
        required_columns = {'text', 'sentiment'}
        if not required_columns.issubset(self.raw_data.columns):
            missing = required_columns - set(self.raw_data.columns)
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def save_raw_data(self, save_path: str):
        self.raw_data.to_parquet(save_path)