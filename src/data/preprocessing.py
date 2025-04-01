from sklearn.model_selection import train_test_split
from pycaret.nlp import *

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.processed_data = None
        
    def clean_text(self):
        self.data['cleaned_text'] = self.data['text'].str.lower()
        return self
    
    def split_data(self, test_size=0.2):
        return train_test_split(
            self.data, 
            test_size=test_size,
            stratify=self.data['sentiment']
        )
    
    def setup_pycaret(self):
        return setup(
            data=self.data,
            target='sentiment',
            session_id=42,
            log_experiment=True
        )