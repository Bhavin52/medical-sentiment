import mlflow
from pycaret.classification import *

class ModelTrainer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.experiment = None
        self.best_model = None
        
    def setup_experiment(self):
        self.experiment = setup(
            data=self.data,
            target='sentiment',
            log_experiment=True,
            experiment_name='medical_sentiment',
            verbose=False
        )
        return self
    
    def train(self):
        self.best_model = compare_models()
        return self
    
    def save_model(self, model_name: str):
        final_model = finalize_model(self.best_model)
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=final_model,
            registered_model_name=model_name
        )
        return final_model