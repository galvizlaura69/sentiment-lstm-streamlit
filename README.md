# Spanish IMDb Review Sentiment Classifier with LSTM and Streamlit

This project trains a recurrent neural network (Bidirectional LSTM) to classify Spanish-language movie reviews as positive or negative. It uses the Spanish-translated version of the IMDb 50K movie review dataset.

## Project contents

- `notebooks/rnn_experiment.ipynb`: Spanish review dataset exploration and experiment walkthrough.
- `train.py`: data cleaning, stratified train/validation/test split, LSTM training, evaluation, plots, and ten test predictions.
- `predict.py`: same tokenizer and padding steps for new Spanish text.
- `app.py`: Spanish Streamlit interface with a review guide and prediction confidence.
- `models/`: generated trained network, tokenizer, and model configuration.
- `outputs/`: generated metrics, training plots, confusion matrix, and test predictions.

## Dataset

The project uses **IMDb Dataset of 50K Movie Reviews (Spanish)** by Luis Diego Fernandez on Kaggle. It includes the English review, Spanish translation, and sentiment label. The Spanish translations were produced with machine translation and may contain errors. The model trains on the Spanish review text and its positive/negative label.

Download the CSV from [Kaggle](https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish). Place the file named `IMDB Dataset SPANISH.csv` inside the project's `data/` directory:

```text
data/IMDB Dataset SPANISH.csv
```

The dataset is not included in Git because it is large. Keep it locally for training; the notebook, source code, and dataset link are included to make the experiment reproducible.

## Setup on Windows

Python 3.10–3.13 is suitable. From PowerShell opened in the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this in the same window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train and evaluate

After downloading the CSV into `data/`, run:

```powershell
python train.py
```

The script cleans the Spanish review/label columns, creates stratified 80/10/10 train/validation/test splits, fits the tokenizer only on training text, and trains the Bidirectional LSTM. It reports accuracy, precision, recall, F1, and a classification report. It saves the model and tokenizer under `models/`, and saves plots, metrics, and ten held-out test predictions under `outputs/`.

## Run Streamlit

Train the Spanish model first, then:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Enter a Spanish review as one paragraph. A useful review contains an overall opinion and a few details such as story, acting, pacing, or music. Two to five sentences are a good starting format; there is no required template.

The app shows the predicted class and confidence in that predicted class. It also lets you expand the positive and negative class probabilities. The percentage is a model estimate, not a guarantee.

## Google Colab

Upload `rnn_experiment.ipynb`, `train.py`, and the Kaggle CSV into the Colab session. In the notebook, upload the CSV when prompted; ensure `train.py` is in the working directory. The notebook explores the dataset and invokes the same training script so the Colab experiment and application use the same pipeline. Save the completed notebook and generated artifacts back into the Git repository for submission.
