import gradio as gr
from fastai.vision.all import *
# 1. Load the model
learn = load_learner('models/pneumonia_model.pkl')
labels = learn.dls.vocab # ['NORMAL', 'PNEUMONIA']

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    
    # NEW: The "Strictness" Check (Threshold)
    # We only call it Pneumonia if the probability is higher than 85%
    # Otherwise, we classify it as Normal to reduce False Positives
    prob_pneumonia = float(probs[1]) # Index 1 is usually PNEUMONIA
    
    threshold = 0.85 
    if prob_pneumonia > threshold:
        final_pred = "PNEUMONIA"
    else:
        final_pred = "NORMAL"
        
    # Return results for the UI
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# 2. Build the Interface
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection AI (v2.0 - Reduced False Positives)",
    description="This version uses a strict 85% confidence threshold to reduce false alarms."
)

interface.launch()

# Create the Web Interface
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection AI",
    description="Upload a Chest X-ray to detect Pneumonia. Note: This is for educational purposes."
)

interface.launch()