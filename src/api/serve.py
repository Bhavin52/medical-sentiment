from fastapi import FastAPI
import mlflow.pyfunc

class ModelServer:
    def __init__(self, model_name: str):
        self.app = FastAPI()
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
        
        @self.app.post("/predict")
        async def predict(text: str):
            return {"sentiment": self.model.predict([text])[0]}
    
    def run(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)