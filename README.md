# MLOps-Ready AI Analytics Engine

This project demonstrates a complete, but simple, MLOps workflow for an anomaly detection system. It features model training with MLflow tracking, ONNX model conversion, and a user-friendly web interface built with Gradio that includes AI-powered explanations using LangChain and OpenAI.

## Features

- **Anomaly Detection Model**: Uses an `IsolationForest` model from scikit-learn to identify anomalies in 2D data.
- **MLOps with MLflow**: All training runs are tracked with MLflow, logging metrics, and model artifacts. The application dynamically loads the latest model from the MLflow tracking server.
- **ONNX Export**: The trained scikit-learn model is converted to the ONNX format for interoperability and is logged as an artifact in MLflow.
- **Interactive UI**: A web interface built with Gradio allows users to input data points and get instant anomaly predictions.
- **AI-Powered Explanations**: Leverages LangChain and an OpenAI model (GPT-3.5-Turbo) to provide clear, non-technical explanations of the anomaly scores.

## Project Structure

- `train.py`: Script to train the Isolation Forest model and log it to MLflow.
- `app.py`: The Gradio web application for inference and explanation.
- `mlruns/`: Directory created by MLflow to store tracking data (should be in `.gitignore`).
- `.env`: File to store your OpenAI API key (you need to create this).
- `requirements.txt`: A list of all the Python dependencies.

## How to Run

Follow these steps to set up and run the project on your local machine.

### 1. Setup Your Environment

First, it's recommended to create a virtual environment to keep dependencies isolated.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
```

Next, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Set Up Your OpenAI API Key

The application uses an OpenAI model for generating explanations. You'll need an API key for this feature to work.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your OpenAI API key to the file like this: `OPENAI_API_KEY="your-openai-api-key-here"`

> **Note:** If you don't provide an API key, the application will still run, but the AI explanation feature will be disabled.

### 3. Train the Model

Before you can run the application, you need to train the model. This script will create an `mlruns` directory where MLflow will save the model.

```bash
python train.py
```

### 4. Launch the Application

Once the model is trained, launch the Gradio web application.

```bash
python app.py
```

The application will start, and you can access it in your browser at the local URL provided in the terminal (usually `http://127.0.0.1:7860`).
