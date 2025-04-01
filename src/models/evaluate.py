import mlflow
from pycaret.classification import predict_model, pull

class ModelEvaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.metrics = None
        
    def evaluate(self):
        predictions = predict_model(self.model, data=self.test_data)
        self.metrics = pull()
        return self.metrics
    
    def log_results(self):
        with mlflow.start_run():
            mlflow.log_metrics(self.metrics)
            mlflow.log_dict(self.metrics, "evaluation_metrics.json")