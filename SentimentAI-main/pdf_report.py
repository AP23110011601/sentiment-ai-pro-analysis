"""
pdf_report.py
─────────────
Generates styled PDF reports with charts for batch analysis and review history.
Uses fpdf2 for PDF generation and matplotlib for chart images.
"""

from fpdf import FPDF
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import io
import tempfile
import os


class SentimentReport(FPDF):
    """Custom PDF class with branded header/footer."""

    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 30, 'F')
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 10, 'Sentiment Studio Pro', align='C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')


def _add_summary_section(pdf, df):
    """Add summary statistics section."""
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, 'Summary Statistics', ln=True)
    pdf.ln(2)

    total = len(df)
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    neu_count = len(df[df['sentiment'] == 'Neutral'])
    avg_conf = df['confidence'].mean() * 100 if 'confidence' in df.columns else 0

    pdf.set_font('Helvetica', '', 10)
    col_w = 47.5

    stats = [
        ('Total Reviews', str(total)),
        ('Positive', f'{pos_count} ({pos_count/total*100:.1f}%)' if total else '0'),
        ('Negative', f'{neg_count} ({neg_count/total*100:.1f}%)' if total else '0'),
        ('Neutral', f'{neu_count} ({neu_count/total*100:.1f}%)' if total else '0'),
        ('Avg Confidence', f'{avg_conf:.1f}%'),
    ]

    if 'top_emotion' in df.columns and not df['top_emotion'].empty:
        top_emo = df['top_emotion'].mode()[0]
        stats.append(('Top Emotion', top_emo.title()))

    for label, value in stats:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(col_w, 8, f'  {label}', border=1, fill=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(col_w, 8, f'  {value}', border=1, ln=True)

    pdf.ln(8)


def _create_sentiment_pie_chart(df):
    """Create sentiment distribution pie chart, return temp file path."""
    counts = df['sentiment'].value_counts()
    colors_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
    colors = [colors_map.get(s, '#888') for s in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_weight('bold')
    ax.set_title('Sentiment Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _create_emotion_bar_chart(df):
    """Create emotion distribution bar chart, return temp file path."""
    if 'top_emotion' not in df.columns:
        return None

    counts = df['top_emotion'].value_counts()
    colors_map = {
        'joy': '#22c55e', 'anger': '#ef4444', 'sadness': '#3b82f6',
        'fear': '#a855f7', 'surprise': '#f59e0b', 'disgust': '#6b7280', 'neutral': '#94a3b8'
    }
    colors = [colors_map.get(e, '#888') for e in counts.index]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Emotion Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Count', fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10, fontweight='bold')

    ax.invert_yaxis()
    plt.tight_layout()

    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _create_confidence_histogram(df):
    """Create confidence distribution histogram, return temp file path."""
    if 'confidence' not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df['confidence'] * 100, bins=15, color='#8b5cf6', edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_title('Confidence Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Confidence (%)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.axvline(df['confidence'].mean() * 100, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {df["confidence"].mean()*100:.1f}%')
    ax.legend(fontsize=9)
    plt.tight_layout()

    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _add_charts_section(pdf, df):
    """Generate charts and embed them in the PDF."""
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, 'Visual Insights', ln=True)
    pdf.ln(2)

    chart_files = []

    # Sentiment pie chart
    try:
        pie_path = _create_sentiment_pie_chart(df)
        chart_files.append(pie_path)
        pdf.image(pie_path, x=10, w=90)
    except Exception as e:
        print("Error adding pie chart:", e)

    # Emotion bar chart (side by side with pie)
    try:
        emo_path = _create_emotion_bar_chart(df)
        if emo_path:
            chart_files.append(emo_path)
            pdf.image(emo_path, x=105, y=pdf.get_y() - 75, w=95)
    except Exception as e:
        print("Error adding emotion chart:", e)

    pdf.ln(5)

    # Confidence histogram (new row)
    try:
        conf_path = _create_confidence_histogram(df)
        if conf_path:
            chart_files.append(conf_path)
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.image(conf_path, x=30, w=150)
            pdf.ln(5)
    except Exception as e:
        print("Error adding confidence chart:", e)

    # Clean up temp files
    for f in chart_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except Exception as e:
            print("Error deleting temp file:", e)

    pdf.ln(5)


def _add_results_table(pdf, df):
    """Add the detailed results table."""
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, 'Detailed Results', ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)

    cols = [('Review Text', 70), ('Sentiment', 25), ('Confidence', 22)]
    if 'top_emotion' in df.columns:
        cols.append(('Emotion', 22))
    if 'language' in df.columns:
        cols.append(('Language', 22))

    total_w = sum(w for _, w in cols)
    remaining = 190 - total_w
    if remaining > 0:
        cols[0] = (cols[0][0], cols[0][1] + remaining)

    for name, w in cols:
        pdf.cell(w, 7, f'  {name}', border=1, fill=True)
    pdf.ln()

    # Table rows
    pdf.set_text_color(30, 30, 30)
    for idx, row in df.iterrows():
        if pdf.get_y() > 260:
            pdf.add_page()

        sent = row.get('sentiment', '')
        if sent == 'Positive':
            pdf.set_fill_color(220, 252, 231)
        elif sent == 'Negative':
            pdf.set_fill_color(254, 226, 226)
        else:
            pdf.set_fill_color(254, 249, 195)

        pdf.set_font('Helvetica', '', 7)
        raw_text = str(row.get('text', ''))[:80].replace('\n', ' ').replace('\r', '')
        text_preview = raw_text.encode('latin-1', 'replace').decode('latin-1')
        conf = row.get('confidence', 0)

        pdf.cell(cols[0][1], 6, f'  {text_preview}', border=1, fill=True)
        pdf.cell(cols[1][1], 6, f'  {sent}', border=1, fill=True)
        pdf.cell(cols[2][1], 6, f'  {conf*100:.1f}%', border=1, fill=True)

        col_idx = 3
        if 'top_emotion' in df.columns and col_idx < len(cols):
            emo = str(row.get('top_emotion', '')).title()
            pdf.cell(cols[col_idx][1], 6, f'  {emo}', border=1, fill=True)
            col_idx += 1
        if 'language' in df.columns and col_idx < len(cols):
            lang = str(row.get('language', ''))
            pdf.cell(cols[col_idx][1], 6, f'  {lang}', border=1, fill=True)

        pdf.ln()


def generate_pdf_report(df, title="Sentiment Analysis Report"):
    """
    Generate a PDF report with summary, charts, and detailed table.
    Returns bytes for st.download_button.
    """
    pdf = SentimentReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(10)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', ln=True, align='C')
    pdf.ln(10)

    # Summary stats
    _add_summary_section(pdf, df)

    # Charts with visual insights
    if len(df) >= 2:
        _add_charts_section(pdf, df)

    # Detailed table (new page for clarity)
    pdf.add_page()
    _add_results_table(pdf, df)

    return bytes(pdf.output())
