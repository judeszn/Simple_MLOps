import gradio as gr
import mlflow
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- MLflow Configuration ---

MLFLOW_TRACKING_URI = "mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Model Loading (MLOps Integration) ---
def load_latest_model():
    """
    Loads the latest scikit-learn model from the MLflow tracking server.
    """
    try:
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if len(runs) == 0:
            raise Exception("No MLflow runs found. Please run train.py first.")
        
        latest_run_id = runs.iloc[0].run_id
        print(f"Loading model from run ID: {latest_run_id}")
        
        model_uri = f"runs:/{latest_run_id}/sklearn_model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure mlflow is running and you have trained a model using train.py.")
        return None

# --- LangChain Explanation Engine ---
def create_explanation_chain():
    """
    Creates a LangChain chain to explain anomaly scores.
    """
    prompt_template = """
    You are an AI assistant for an anomaly detection system. Your task is to provide a clear, concise explanation of a data point's anomaly score.

    The model is an Isolation Forest. The anomaly score it produces has the following meaning:
    - Scores close to 1 are highly normal.
    - Scores close to -1 are highly anomalous.
    - Scores around 0 are ambiguous.

    The user's data point received a score of: {score:.4f}

    Based on this score, please provide a brief, one or two-sentence explanation for a non-technical user.
    Start by stating if the point is likely 'Normal' or an 'Anomaly'.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
    chain = prompt | llm | StrOutputParser()
    return chain

# --- Main Application Logic ---
model = load_latest_model()
explanation_chain = create_explanation_chain()

def analyze_data_point(feature1, feature2):
    """
    Analyzes a single data point using the loaded model and LangChain.
    """
    if model is None:
        return "Error", "Error", "Model not loaded. Please check the console for errors."
    if feature1 is None or feature2 is None:
        return "N/A", "N/A", "Please provide values for both features."

    data_point = np.array([[feature1, feature2]])
    
    prediction_val = model.predict(data_point)[0]
    score_val = model.decision_function(data_point)[0]
    prediction_label = "Normal" if prediction_val == 1 else "Anomaly"
    
    explanation = "AI explanation not available. Check that you have a .env file with your GEMINI_API_KEY."
    if os.getenv("GEMINI_API_KEY"):
        explanation = explanation_chain.invoke({"score": score_val})
    return prediction_label, f"{score_val:.4f}", explanation

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Analytics Engine") as demo:
    gr.Markdown("# MLOps-Ready AI Analytics Engine")
    gr.Markdown("Enter a 2D data point to check if it's an anomaly. The system uses an Isolation Forest model tracked with MLflow and a LangChain-powered AI to explain the results.")

    css = """
    .smaller-text {font-size: 14px;}
    """
    gr.HTML(f"<style>{css}</style>")

    with gr.Row():
        with gr.Column():
            feature1_input = gr.Number(label="Feature 1 Value")
            feature2_input = gr.Number(label="Feature 2 Value")
            analyze_btn = gr.Button("Analyze Data Point", variant="primary")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            score_output = gr.Textbox(label="Anomaly Score", interactive=False, elem_classes="smaller-text")
            explanation_output = gr.Textbox(label="AI Explanation", lines=4, interactive=False)

    analyze_btn.click(fn=analyze_data_point, inputs=[feature1_input, feature2_input], outputs=[prediction_output, score_output, explanation_output])
    gr.Examples(examples=[[0.1, 0.2], [-3.5, 3.5], [0, 0], [10, -10]], inputs=[feature1_input, feature2_input])

if __name__ == "__main__":
    demo.launch()