from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotion(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Map ID to emotion label
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    all_probs = {labels[i]: float(probs[i].item()) for i in range(len(labels))}
    predicted_class = torch.argmax(probs).item()
    
    return {
        "emotion": labels[predicted_class],
        "confidence": float(probs[predicted_class].item()),
        "all_probs": all_probs
    }