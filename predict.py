"""Spanish text preprocessing and inference helpers."""
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def load_assets():
    model_path = MODEL_DIR / "sentiment_lstm.keras"
    tokenizer_path = MODEL_DIR / "spanish_tokenizer.json"
    config_path = MODEL_DIR / "config.json"
    missing = [p.name for p in (model_path, tokenizer_path, config_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos del modelo en español: " + ", ".join(missing) + ". Ejecuta `python train.py`."
        )
    model = tf.keras.models.load_model(model_path)
    tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return model, tokenizer, config


def predict_review(text, model=None, tokenizer=None, config=None):
    if model is None or tokenizer is None or config is None:
        model, tokenizer, config = load_assets()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=config["maxlen"], padding="post", truncating="post")
    positive_probability = float(model.predict(padded, verbose=0)[0][0])
    threshold = float(config.get("threshold", 0.5))
    positive = positive_probability >= threshold
    sentiment = "positiva" if positive else "negativa"
    confidence = positive_probability if positive else 1 - positive_probability
    return sentiment, confidence, positive_probability
