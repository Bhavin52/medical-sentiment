from pathlib import Path
from src.data.ingestion import DataIngestor
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

class MLPipeline:
    def __init__(self, config: dict):
        self.config = config
        Path(self.config['data_path']).mkdir(parents=True, exist_ok=True)
        
    def run(self):
        # Data Ingestion
        ingestor = DataIngestor(self.config['raw_data_path'])
        raw_data = ingestor.load_data()
        ingestor.validate_data()
        ingestor.save_raw_data(self.config['raw_save_path'])
        
        # Data Preprocessing
        processor = DataPreprocessor(raw_data)
        train_data, test_data = processor.split_data()
        
        # Model Training
        trainer = ModelTrainer(train_data)
        model = trainer.setup_experiment().train().save_model('medical_sentiment')
        
        # Model Evaluation
        evaluator = ModelEvaluator(model, test_data)
        metrics = evaluator.evaluate().log_results()
        
        return model, metrics

if __name__ == "__main__":
    config = {
        'raw_data_path': 'data/raw/medical_data.csv',
        'raw_save_path': 'data/processed/base_data.parquet'
    }
    pipeline = MLPipeline(config)
    model, metrics = pipeline.run()
    print(f"Pipeline completed. Metrics: {metrics}")