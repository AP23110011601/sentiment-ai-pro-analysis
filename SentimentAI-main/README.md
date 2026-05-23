# SentimentPro-AI (Sentiment Studio Pro)

A premium sentiment analysis application featuring a beautiful Streamlit interface, powered by fine-tuned transformer models (`distilroberta-base`).

## ✨ Features
- **Transformer-based Inference**: Uses Hugging Face's `distilroberta-base` (fine-tuned on `tweet_eval`) for highly accurate sequence classification.
- **Three-Class Sentiment**: Accurately detects **Positive**, **Neutral**, and **Negative** sentiments, complete with confidence scoring and probability distributions.
- **Emotion Detection (NEW)**: Analyzes text beyond basic sentiment to detect 7 specific emotions: Joy, Anger, Sadness, Fear, Surprise, Disgust, and Neutral using a dedicated DistilRoBERTa emotion model.
- **Keyword Explainability (NEW)**: Highlights the specific words (positive, negative, intensifiers, negators) that most influenced the sentiment prediction with color-coded annotations.
- **Multi-language Support (NEW)**: Auto-detects over 40 languages and seamlessly translates reviews to English before analysis using `deep-translator`.
- **Interactive Dashboards (NEW)**: Visualizes batch analysis and session history with Plotly-powered pie charts, bar charts, and confidence histograms.
- **Sentiment Trend Line (NEW)**: Maps the sequential change in sentiment score over time to visualize shifting attitudes.
- **Batch Processing**: Upload CSV/TXT files or paste multiple lines of text to process hundreds of reviews in seconds.
- **PDF Export & Reports (NEW)**: Download beautifully styled, color-coded PDF reports containing summary statistics and embedded visual charts.
- **Persistent Review History (NEW)**: Automatically saves your analysis history locally (`history.json`) so your data persists even after refreshing the app.
- **Modern Web Interface**: A premium Streamlit dashboard featuring a 5-tab layout, glassmorphism, dynamic animations, and a polished dark theme.

## 📂 Project Structure
- `app.py`: Main Streamlit web application.
- `app_styles.py`: Extracted CSS and HTML templates for the premium UI.
- `emotion_detector.py`: Hugging Face pipeline for 7-class emotion detection.
- `keyword_extractor.py`: Logic for highlighting sentiment-bearing words.
- `translator.py`: Language detection and auto-translation logic.
- `pdf_report.py`: PDF generation logic using `fpdf2` and `matplotlib`.
- `predict.py`: CLI script for quick sentiment inference and testing.
- `train.py`: Script to fine-tune the transformer model.
- `data_loader.py`: Loads and samples the `tweet_eval` dataset.
- `preprocessing.py`: Handles text cleaning.
- `evaluate.py`: Model evaluation script.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Running the App
To launch the Sentiment Studio Pro web interface, simply run:
```bash
py -m streamlit run app.py
```
This will open the application in your default web browser (usually at `localhost:8502`).

### Quick CLI Testing
If you want to test the model quickly in your terminal without the UI:
```bash
py predict.py
```

## 🎨 UI Highlights
The application features a fully custom-styled CSS layout overriding default Streamlit elements to provide a "wow" factor, utilizing gradients, rounded corners, drop shadows, and responsive hover states.
