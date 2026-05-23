"""
app_styles.py — CSS styles for Sentiment Studio Pro
"""

MAIN_CSS = """
<style>
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] {
    background: #0f172a !important;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b, #0f172a) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.stApp { background: radial-gradient(circle at top, #0f172a, #020617); color: white; }
.glass {
    background: rgba(255,255,255,0.05);
    padding: 25px; border-radius: 18px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 15px;
}
.header {
    background: linear-gradient(135deg, #1e293b, #0f4a46);
    padding: 35px; border-radius: 20px; color: white; margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px; border-radius: 15px;
    text-align: center; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.05);
}
.stButton>button {
    width: 100%; border-radius: 10px; padding: 12px;
    font-weight: 600;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white; border: none;
}
textarea {
    background: #020617 !important; color: white !important;
    border-radius: 12px !important; border: 1px solid #334155 !important;
}
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.metric-card {
    text-align: center; padding: 20px; border-radius: 12px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
}
.keyword-box {
    padding: 20px; border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 16px; line-height: 2;
}
.lang-badge {
    display: inline-block; padding: 6px 14px; border-radius: 8px;
    background: rgba(59,130,246,0.2); border: 1px solid rgba(59,130,246,0.3);
    font-size: 13px; margin-bottom: 10px;
}
.history-card {
    padding: 15px; border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
</style>
"""

HEADER_HTML = """
<div class="header">
    <h1>Sentiment Studio Pro</h1>
    <p>Premium sentiment analysis with emotion detection, keyword highlighting, multi-language support, and interactive dashboards.</p>
</div>
"""

def metric_card_html(title, value):
    return f'''<div class="metric-card">
        <div style="font-size:14px;opacity:0.7;margin-bottom:8px;">{title}</div>
        <div style="font-size:28px;font-weight:bold;">{value}</div>
    </div>'''

SENTIMENT_EMOJI = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
SENTIMENT_COLOR = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"}
