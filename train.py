"""Train an LSTM using the Spanish-translated IMDb review dataset.

Place the Kaggle CSV at data/IMDB Dataset SPANISH.csv before running this script.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "IMDB Dataset SPANISH.csv"
MODEL_DIR, OUTPUT_DIR = ROOT / "models", ROOT / "outputs"


def read_dataset(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            "No encuentro el CSV de IMDb en español. Descarga 'IMDB Dataset SPANISH.csv' "
            "desde https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish "
            "y guárdalo en la carpeta data del proyecto."
        )
    df = pd.read_csv(path)
    review_col = next((c for c in ("review_es", "review", "text") if c in df.columns), None)
    label_col = next((c for c in ("sentimiento", "sentiment", "label") if c in df.columns), None)
    if review_col is None or label_col is None:
        raise ValueError(f"CSV columns not recognized. Found: {list(df.columns)}")
    df = df[[review_col, label_col]].dropna().copy()
    df.columns = ["review", "label"]
    df["review"] = df["review"].astype(str).str.strip()
    label_map = {"positive": 1, "positiva": 1, "positivo": 1, "1": 1,
                 "negative": 0, "negativa": 0, "negativo": 0, "0": 0}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["review"].str.len() > 0].drop_duplicates(subset="review").reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain positive and negative labels.")
    return df


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = read_dataset()
    print(f"Rows after cleanup: {len(df):,}")
    print("Class counts (0=negative, 1=positive):\n", df["label"].value_counts().sort_index())

    plt.figure(figsize=(7, 4))
    df["label"].map({0: "Negative", 1: "Positive"}).value_counts().plot(kind="bar", color=["#d97979", "#68a987"])
    plt.title("Spanish IMDb review label counts")
    plt.xlabel("Sentiment")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=150)
    plt.close()

    df["token_count"] = df["review"].str.split().str.len()
    plt.figure(figsize=(8, 4))
    plt.hist(df["token_count"].clip(upper=800), bins=40, color="#5b8db8")
    plt.title("Spanish review lengths (capped at 800 words)")
    plt.xlabel("Words")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "review_lengths.png", dpi=150)
    plt.close()

    texts = df["review"].to_numpy()
    labels = df["label"].to_numpy()
    train_texts, temp_texts, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, y_val, y_test = train_test_split(
        temp_texts, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vocab_size, maxlen = 20000, 400
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=maxlen, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=maxlen, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=maxlen, padding="post", truncating="post")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(maxlen,)),
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=96, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.25)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=8,
        batch_size=64, callbacks=callbacks, verbose=1
    )

    probs = model.predict(x_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))
    print(classification_report(y_test, preds, target_names=["negative", "positive"], zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Training")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Training")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_plot.png", dpi=150)
    plt.close(fig)

    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["negative", "positive"], cmap="Blues")
    plt.title("Spanish IMDb test confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    sample_rows = []
    for i in range(min(10, len(test_texts))):
        sample_rows.append({
            "review": test_texts[i],
            "true_label": "positiva" if y_test[i] else "negativa",
            "predicted_label": "positiva" if preds[i] else "negativa",
            "predicted_class_confidence": float(probs[i] if preds[i] else 1 - probs[i]),
            "positive_probability": float(probs[i]),
        })
    pd.DataFrame(sample_rows).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model.save(MODEL_DIR / "sentiment_lstm.keras")
    (MODEL_DIR / "spanish_tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    config = {"vocab_size": vocab_size, "maxlen": maxlen, "threshold": 0.5,
              "language": "es", "dataset": "Spanish-translated IMDb 50K"}
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    old_word_index = MODEL_DIR / "word_index.json"
    if old_word_index.exists():
        old_word_index.unlink()
    print(f"Saved Spanish model and tokenizer to {MODEL_DIR}")
    print(f"Saved plots, metrics, and test predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
