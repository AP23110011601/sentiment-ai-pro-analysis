import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from emotion_detector import detect_emotions, get_top_emotion, EMOTION_EMOJI, EMOTION_COLOR
from keyword_extractor import extract_keywords
from translator import detect_and_translate
from app_styles import MAIN_CSS, HEADER_HTML, metric_card_html, SENTIMENT_EMOJI, SENTIMENT_COLOR
from pdf_report import generate_pdf_report

# ================= CONFIG =================
st.set_page_config(page_title="Sentiment Studio Pro", layout="wide")

# ================= MODEL =================
@st.cache_resource
def load_model():
    MODEL_PATH = "sentiment_model"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
num_labels = model.config.num_labels

# Adapt labels to actual model output
if hasattr(model.config, 'id2label'):
    labels = [model.config.id2label[i] for i in range(num_labels)]
else:
    labels = ['Negative', 'Neutral', 'Positive'] if num_labels == 3 else ['Negative', 'Positive']

def extract_probs(probs):
    if num_labels == 3:
        return probs[0].item(), probs[1].item(), probs[2].item()
    return probs[0].item(), 0.0, probs[1].item()

def run_inference(text):
    text_lower = text.lower()
    neutral_keywords = ["okay","fine","average","not bad","so so","nothing special",
        "as expected","normal","decent","fair","moderate","satisfactory","it works","no issues","acceptable"]
    if any(k in text_lower for k in neutral_keywords):
        return "Neutral", 1.0, 0.0, 1.0, 0.0

    strong_pos = ["good","great","excellent","awesome","amazing","perfect","love"]
    strong_neg = ["bad","worst","terrible","awful","poor","hate"]
    words = text_lower.split()
    if len(words) <= 3:
        for w in words:
            if w in strong_neg:
                return ("Negative", 0.90, 0.90, 0.05, 0.05) if num_labels==3 else ("Negative", 0.90, 0.90, 0.0, 0.10)
            if w in strong_pos:
                return ("Positive", 0.90, 0.05, 0.05, 0.90) if num_labels==3 else ("Positive", 0.90, 0.10, 0.0, 0.90)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    neg, neu, pos = extract_probs(probs)
    confidence = max(probs).item()
    score = pos - neg
    if score > 0.2: sentiment = "Positive"
    elif score < -0.2: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, confidence, neg, neu, pos

import json
import os

def load_data(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_data(filename, data):
    if IS_LOCAL:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# Check if running locally (if model folder exists)
IS_LOCAL = os.path.exists("sentiment_model")

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = load_data("history.json") if IS_LOCAL else []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = load_data("batch_results.json") if IS_LOCAL else []

# ================= SIDEBAR =================
with st.sidebar:
    st.subheader("Example Input")
    example = st.selectbox("", [
        "None", "This product is amazing!", "Absolutely love it, best purchase ever!",
        "Worst experience ever", "Completely useless, total waste of money",
        "Good quality but terrible service", "The package arrived on the expected date",
        "Some features work well, others do not", "I have mixed feelings about this product",
    ], key="example_select")
    st.markdown("---")
    st.subheader("About")
    st.write("3-class sentiment + emotion detection using DistilRoBERTa.")
    st.write("• Sentiment: Positive / Neutral / Negative")
    st.write("• Emotions: Joy, Anger, Sadness, Fear, Surprise, Disgust")
    st.write("• Multi-language auto-translation")
    st.markdown("---")
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.batch_results = []
        save_data("history.json", [])
        save_data("batch_results.json", [])
        st.rerun()

# ================= STYLES & HEADER =================
st.markdown(MAIN_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# Top info cards
c1, c2, c3 = st.columns(3)
c1.markdown('<div class="card">MODEL<br><b>DistilRoBERTa</b></div>', unsafe_allow_html=True)
c2.markdown('<div class="card">CLASSES<br><b>Positive / Neutral / Negative</b></div>', unsafe_allow_html=True)
c3.markdown('<div class="card">FEATURES<br><b>8 AI Modules</b></div>', unsafe_allow_html=True)
st.write("")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "📈 Dashboard", "📜 History", "📉 Trend Line"])

# ==================== TAB 1: SINGLE ANALYSIS ====================
with tab1:
    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Analyze Text")
        default_text = example if example != "None" else ""
        review = st.text_area("", height=140, placeholder="Enter text to analyze...", value=default_text)
        cA, cB = st.columns(2)
        analyze = cA.button("Analyze", use_container_width=True)
        clear = cB.button("Clear", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Quick Guide")
        st.write("1. Enter text (any language supported)")
        st.write("2. Click Analyze for sentiment + emotions")
        st.write("3. View confidence, keywords & charts")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze and review.strip():
        # --- Multi-language translation ---
        translated_text, src_lang, was_translated = detect_and_translate(review)

        if was_translated:
            st.markdown(f'<div class="lang-badge">🌐 Detected: <b>{src_lang}</b> → Translated to English</div>', unsafe_allow_html=True)
            with st.expander("View translation"):
                st.write(f"**Original:** {review}")
                st.write(f"**Translated:** {translated_text}")
            analysis_text = translated_text
        else:
            st.markdown(f'<div class="lang-badge">🌐 Language: <b>English</b></div>', unsafe_allow_html=True)
            analysis_text = review

        # --- Sentiment ---
        sentiment, confidence, neg, neu, pos = run_inference(analysis_text)

        # --- Emotions ---
        emotions = detect_emotions(analysis_text)
        top_emotion, top_emotion_score = get_top_emotion(emotions)

        # --- Keywords ---
        kw = extract_keywords(analysis_text)

        # Store result
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": review, "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "positive_prob": round(pos, 4), "neutral_prob": round(neu, 4), "negative_prob": round(neg, 4),
            "top_emotion": top_emotion, "emotion_score": round(top_emotion_score, 4),
            "language": src_lang, "sentiment_score": round(pos - neg, 4),
            "word_count": len(review.split()), "char_count": len(review)
        }
        st.session_state.history.append(result)
        save_data("history.json", st.session_state.history)

        # --- Display Results ---
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Analysis Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card_html("SENTIMENT", f"{SENTIMENT_EMOJI[sentiment]} {sentiment}"), unsafe_allow_html=True)
        m2.markdown(metric_card_html("CONFIDENCE", f"{confidence*100:.1f}%"), unsafe_allow_html=True)
        m3.markdown(metric_card_html("TOP EMOTION", f"{EMOTION_EMOJI.get(top_emotion,'❓')} {top_emotion.title()}"), unsafe_allow_html=True)
        m4.markdown(metric_card_html("LANGUAGE", f"🌐 {src_lang}"), unsafe_allow_html=True)

        st.markdown("")

        # Confidence meter
        st.write("**Confidence Meter**")
        st.progress(min(confidence, 1.0))

        st.markdown("")
        col_chart, col_details = st.columns([1.2, 1])

        with col_chart:
            # Sentiment pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Negative','Neutral','Positive'], values=[neg*100, neu*100, pos*100],
                marker=dict(colors=['#ef4444','#f59e0b','#22c55e']),
                textinfo='label+percent', textfont=dict(size=13, color='white'),
                hole=0.4
            )])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), showlegend=False, height=350, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_details:
            st.markdown("**PROBABILITY BREAKDOWN**")
            st.write(f"🟢 Positive: **{pos*100:.2f}%**"); st.progress(pos)
            st.write(f"🟡 Neutral: **{neu*100:.2f}%**"); st.progress(neu)
            st.write(f"🔴 Negative: **{neg*100:.2f}%**"); st.progress(neg)

        st.markdown('</div>', unsafe_allow_html=True)

        # --- Emotion Detection Panel ---
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🎭 Emotion Analysis")
        emo_df = pd.DataFrame(emotions)
        emo_df['emoji'] = emo_df['label'].map(EMOTION_EMOJI)
        emo_df['pct'] = (emo_df['score'] * 100).round(1)

        fig_emo = px.bar(emo_df, x='score', y='label', orientation='h',
            color='label', color_discrete_map=EMOTION_COLOR,
            text=emo_df['pct'].astype(str) + '%')
        fig_emo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), showlegend=False, height=280, yaxis_title="", xaxis_title="Score",
            margin=dict(t=10,b=30,l=10,r=10))
        fig_emo.update_traces(textposition='outside')
        st.plotly_chart(fig_emo, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Keyword Highlighting ---
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔍 Keyword Highlighting")
        st.markdown(f'<div class="keyword-box">{kw["annotated_html"]}</div>', unsafe_allow_html=True)
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("🟢 Positive", len(kw["positive_words"]))
        kc2.metric("🔴 Negative", len(kw["negative_words"]))
        kc3.metric("🔵 Intensifiers", len(kw["intensifiers"]))
        kc4.metric("🟠 Negators", len(kw["negators"]))
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: BATCH ANALYSIS ====================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Batch Analysis Input")
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"], horizontal=True)
    df_input = None

    if input_method == "Paste Text":
        batch_text = st.text_area("Paste reviews (one per line)", height=200,
            placeholder="Enter multiple reviews here...\nEach line is a separate review.")
        if batch_text:
            texts = [l.strip() for l in batch_text.split('\n') if l.strip()]
            if texts:
                df_input = pd.DataFrame({"text": texts})
                st.info(f"Loaded {len(df_input)} reviews")
    else:
        uploaded = st.file_uploader("Choose a CSV or TXT file", type=["csv","txt"], key="batch_upload")
        if uploaded:
            try:
                if uploaded.type == "text/plain":
                    content = uploaded.read().decode("utf-8")
                    texts = [l.strip() for l in content.split('\n') if l.strip()]
                    df_input = pd.DataFrame({"text": texts})
                else:
                    df_input = pd.read_csv(uploaded)
                if 'text' not in df_input.columns:
                    st.error("File must contain a 'text' column"); df_input = None
                else:
                    st.info(f"📋 Loaded {len(df_input)} rows")
            except Exception as e:
                st.error(f"Error: {e}"); df_input = None

    if df_input is not None and not df_input.empty:
        if st.button("🚀 Analyze Batch", use_container_width=True):
            progress = st.progress(0)
            results_batch = []
            for idx, row in df_input.iterrows():
                text = str(row['text']).strip()
                if text:
                    tr_text, lang, _ = detect_and_translate(text)
                    sentiment, conf, neg, neu, pos = run_inference(tr_text)
                    emos = detect_emotions(tr_text)
                    top_emo, _ = get_top_emotion(emos)
                    results_batch.append({
                        "text": text, "sentiment": sentiment, "confidence": round(conf, 4),
                        "top_emotion": top_emo, "language": lang,
                        "positive_prob": round(pos,4), "neutral_prob": round(neu,4), "negative_prob": round(neg,4),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                progress.progress((idx + 1) / len(df_input))
            st.session_state.batch_results = results_batch
            save_data("batch_results.json", results_batch)
            st.success(f"✅ Analyzed {len(results_batch)} texts!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Batch results display
    if st.session_state.batch_results:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Batch Results")
        df_res = pd.DataFrame(st.session_state.batch_results)
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("🟢 Positive", len(df_res[df_res['sentiment']=='Positive']))
        bc2.metric("🔴 Negative", len(df_res[df_res['sentiment']=='Negative']))
        bc3.metric("🟡 Neutral", len(df_res[df_res['sentiment']=='Neutral']))
        bc4.metric("📊 Avg Confidence", f"{df_res['confidence'].mean()*100:.1f}%")
        st.dataframe(df_res, use_container_width=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇ Download CSV", df_res.to_csv(index=False),
                "batch_results.csv", "text/csv", use_container_width=True)
        with dl2:
            pdf_bytes = generate_pdf_report(df_res, title="Batch Analysis Report")
            st.download_button("📄 Download PDF Report", pdf_bytes,
                "batch_report.pdf", "application/pdf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 3: DASHBOARD ====================
with tab3:
    if not st.session_state.history:
        st.info("📊 Analyze some reviews first to see the dashboard.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Sentiment Distribution")
        dc1, dc2 = st.columns(2)
        with dc1:
            sent_counts = df_hist['sentiment'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=sent_counts.index, values=sent_counts.values,
                marker=dict(colors=[SENTIMENT_COLOR.get(s,'#888') for s in sent_counts.index]),
                hole=0.45, textinfo='label+percent', textfont=dict(size=13, color='white')
            )])
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=350, showlegend=False, margin=dict(t=20,b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        with dc2:
            fig_bar = px.bar(sent_counts, x=sent_counts.index, y=sent_counts.values,
                color=sent_counts.index, color_discrete_map=SENTIMENT_COLOR,
                labels={'x':'Sentiment','y':'Count'})
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=350, showlegend=False, margin=dict(t=20,b=40))
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Emotion distribution
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🎭 Emotion Distribution")
        emo_counts = df_hist['top_emotion'].value_counts()
        fig_emo_d = px.bar(emo_counts, x=emo_counts.index, y=emo_counts.values,
            color=emo_counts.index, color_discrete_map=EMOTION_COLOR,
            labels={'x':'Emotion','y':'Count'})
        fig_emo_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=300, showlegend=False, margin=dict(t=20,b=40))
        st.plotly_chart(fig_emo_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Confidence histogram
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📏 Confidence Distribution")
        fig_conf = px.histogram(df_hist, x='confidence', nbins=20, color_discrete_sequence=['#8b5cf6'],
            labels={'confidence':'Confidence Score'})
        fig_conf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=300, margin=dict(t=20,b=40))
        st.plotly_chart(fig_conf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 4: HISTORY ====================
with tab4:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📜 Review History")
    if not st.session_state.history:
        st.info("No predictions yet. Start analyzing text to build history.")
    else:
        df_h = pd.DataFrame(st.session_state.history)
        # Filter
        filter_sent = st.multiselect("Filter by sentiment:", ["Positive","Neutral","Negative"],
            default=["Positive","Neutral","Negative"])
        df_filtered = df_h[df_h['sentiment'].isin(filter_sent)]
        st.write(f"Showing **{len(df_filtered)}** of {len(df_h)} reviews")

        for _, row in df_filtered.iterrows():
            emoji = SENTIMENT_EMOJI.get(row['sentiment'], '⚪')
            emo_emoji = EMOTION_EMOJI.get(row.get('top_emotion',''), '❓')
            st.markdown(f'''<div class="history-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span>{emoji} <b>{row["sentiment"]}</b> — {emo_emoji} {row.get("top_emotion","").title()}</span>
                    <span style="opacity:0.5;font-size:12px;">{row["timestamp"]}</span>
                </div>
                <div style="margin-top:8px;opacity:0.8;font-size:14px;">{row["text"][:200]}{"..." if len(str(row["text"]))>200 else ""}</div>
                <div style="margin-top:6px;font-size:12px;opacity:0.5;">
                    Confidence: {row["confidence"]*100:.1f}% | 🌐 {row.get("language","English")} | {row.get("word_count","")} words
                </div>
            </div>''', unsafe_allow_html=True)

        st.markdown("")
        st.dataframe(df_filtered, use_container_width=True)
        hl1, hl2 = st.columns(2)
        with hl1:
            st.download_button("⬇ Download CSV", df_filtered.to_csv(index=False),
                "history.csv", "text/csv", use_container_width=True)
        with hl2:
            pdf_bytes = generate_pdf_report(df_filtered, title="Review History Report")
            st.download_button("📄 Download PDF Report", pdf_bytes,
                "history_report.pdf", "application/pdf", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 5: TREND LINE ====================
with tab5:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📉 Sentiment Trend Line")
    if len(st.session_state.history) < 2:
        st.info("Analyze 2+ reviews to see the sentiment trend line.")
    else:
        df_t = pd.DataFrame(st.session_state.history)
        df_t['review_num'] = range(1, len(df_t) + 1)
        df_t['score'] = df_t['sentiment_score']

        fig_trend = go.Figure()
        # Colored background regions
        fig_trend.add_hrect(y0=0.2, y1=1, fillcolor="rgba(34,197,94,0.08)", line_width=0)
        fig_trend.add_hrect(y0=-0.2, y1=0.2, fillcolor="rgba(245,158,11,0.08)", line_width=0)
        fig_trend.add_hrect(y0=-1, y1=-0.2, fillcolor="rgba(239,68,68,0.08)", line_width=0)

        # Line
        colors = [SENTIMENT_COLOR.get(s, '#888') for s in df_t['sentiment']]
        fig_trend.add_trace(go.Scatter(
            x=df_t['review_num'], y=df_t['score'], mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
            text=df_t['sentiment'], hovertemplate='Review %{x}<br>Score: %{y:.2f}<br>%{text}'
        ))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=400,
            xaxis_title="Review Sequence", yaxis_title="Sentiment Score (Pos − Neg)",
            yaxis=dict(range=[-1.1, 1.1]), margin=dict(t=20,b=40),
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.caption("🟢 Green zone = Positive | 🟡 Yellow zone = Neutral | 🔴 Red zone = Negative")
    st.markdown('</div>', unsafe_allow_html=True)