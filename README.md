# 🚀 Sentiment AI Pro Analysis

A premium AI-powered sentiment and emotion analysis platform built with Streamlit and Transformer models.

This application provides advanced sentiment detection, emotion recognition, multilingual support, keyword explainability, interactive dashboards, and PDF reporting — all inside a beautiful modern UI.

---

# ✨ Features

## 🔍 Sentiment Analysis
- Transformer-based sentiment inference using `distilroberta-base`
- Fine-tuned on the `tweet_eval` dataset
- Detects:
  - Positive
  - Neutral
  - Negative
- Displays:
  - Confidence scores
  - Probability distributions

---

## 😊 Emotion Detection
Detects 7 different emotions using a dedicated DistilRoBERTa emotion model:

- Joy
- Anger
- Sadness
- Fear
- Surprise
- Disgust
- Neutral

---

## 🧠 Keyword Explainability
Highlights important sentiment-driving words using color-coded annotations.

Supports:
- Positive keywords
- Negative keywords
- Intensifiers
- Negators

---

## 🌍 Multi-language Support
- Automatically detects 40+ languages
- Translates text into English before analysis
- Powered by `deep-translator`

---

## 📊 Interactive Dashboards
Visualize sentiment insights using Plotly charts:

- Pie Charts
- Bar Graphs
- Confidence Histograms
- Sentiment Trend Lines

---

## 📂 Batch Processing
Analyze multiple reviews instantly by:
- Uploading CSV files
- Uploading TXT files
- Pasting multiple lines of text

---

## 📄 PDF Report Generation
Generate beautiful downloadable PDF reports including:
- Summary statistics
- Charts and graphs
- Sentiment distributions
- Emotion analysis

---

## 💾 Persistent History
- Automatically stores analysis history locally
- Uses `history.json`
- Data remains available even after refreshing the app

---

## 🎨 Premium UI
Modern Streamlit interface featuring:
- 5-tab dashboard layout
- Glassmorphism design
- Dark theme
- Smooth animations
- Responsive UI components

---

# 📂 Project Structure

```bash
sentiment-ai-pro-analysis/
│
├── app.py                  # Main Streamlit application
├── app_styles.py           # Premium UI styles and templates
├── emotion_detector.py     # Emotion detection pipeline
├── keyword_extractor.py    # Keyword highlighting logic
├── translator.py           # Translation and language detection
├── pdf_report.py           # PDF report generation
├── predict.py              # CLI prediction script
├── train.py                # Model training script
├── data_loader.py          # Dataset loading utilities
├── preprocessing.py        # Text preprocessing
├── evaluate.py             # Model evaluation script
├── requirements.txt        # Project dependencies
└── history.json            # Stored review history
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-ai-pro-analysis.git
cd sentiment-ai-pro-analysis
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit web application:

```bash
py -m streamlit run app.py
```

The application will open in your browser at:

```bash
http://localhost:8501
```

---

# 🖥️ CLI Testing

To quickly test sentiment prediction from the terminal:

```bash
py predict.py
```

---

# 🧪 Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- DistilRoBERTa
- Plotly
- FPDF2
- Matplotlib
- Deep Translator

---

# 📈 Future Improvements

- Real-time Twitter sentiment analysis
- Voice sentiment detection
- Cloud deployment support
- User authentication system
- Database integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed with ❤️ for advanced AI-powered sentiment analysis.
