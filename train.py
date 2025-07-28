from sklearn.ensemble import IsolationForest
import mlflow
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Sample data generation
def generate_data():
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(1000, 2)
    anomalies = rng.uniform(low=-4, high=4, size=(50, 2))
    return np.vstack([X, anomalies])

# MLflow-tracked training
def train():
    X = generate_data()
    
    with mlflow.start_run(run_name="IsolationForest_Training"):
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        
        # Log metrics
        mlflow.log_metric("train_samples", len(X))
        
        # Save model artifacts
        mlflow.sklearn.log_model(
            sk_model=model,
            name="sklearn_model",
            input_example=X[:5]
        )
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, 2]))],
            target_opset={'': 18, 'ai.onnx.ml': 3}
        )
        onnx_model_path = "model.onnx"
        with open(onnx_model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact(onnx_model_path)

if __name__ == "__main__":
    train()