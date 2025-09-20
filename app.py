import streamlit as st
from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer
import base64, os, docx, PyPDF2
import torch
import os, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------- Load Model Once --------------------
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

summarizer, tokenizer = load_summarizer()

# -------------------- Background Image --------------------
def set_background():
    local_path = "assets/background.png"
    fallback_url = "https://images.unsplash.com/photo-1522202176988-66273c2fd55f"

    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        background = f"url('data:image/png;base64,{bin_str}')"
    else:
        background = f"url('{fallback_url}')"

    page_bg = f"""
    <style>
    .stApp {{
        background-image: {background};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 12px;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_background()

# -------------------- Helper Functions --------------------
def summarize_text(text, max_len=400, min_len=100):  # default longer summary
    tokens = tokenizer.encode(text, truncation=True, max_length=1024)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    if len(tokens) >= 1024:
        summaries = []
        words = truncated_text.split()
        for i in range(0, len(words), 900):
            chunk = " ".join(words[i:i + 900])
            summary_chunk = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary_chunk)
        return " ".join(summaries)
    else:
        summary = summarizer(
            truncated_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summary[0]['summary_text']

def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    results = {}
    for metric, score in scores.items():
        results[metric] = {
            "Precision": round(score.precision, 4),
            "Recall": round(score.recall, 4),
            "F1": round(score.fmeasure, 4)
        }
    return results

def read_file(uploaded_file):
    """Read txt, pdf, or docx files and return extracted text."""
    text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    return text

# -------------------- Page Config --------------------
st.set_page_config(page_title="AI Document Summarizer", page_icon="📄", layout="wide")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("⚙️ Settings")
    # Higher sliders for longer summaries
    max_len = st.slider("Max Summary Length", 100, 800, 400)
    min_len = st.slider("Min Summary Length", 50, 200, 100)
    st.markdown("---")
    st.info("Upload a TXT, PDF, or DOCX file!")

# -------------------- Header --------------------
st.markdown(
    "<h1 style='text-align: center; color: #2F4F4F;'>📄 AI Document Summarization Tool</h1>"
    "<p style='text-align: center; color: gray;'>Summarize long documents into detailed, clear summaries.</p>",
    unsafe_allow_html=True
)

# -------------------- Upload Section --------------------
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
reference_summary = st.text_area("Optional: Paste a reference summary here (for ROUGE evaluation)", "")

# -------------------- Layout: Two Columns --------------------
col1, col2 = st.columns(2)

if uploaded_file is not None:
    text = read_file(uploaded_file)

    if text.strip():
        with col1:
            st.subheader("📜 Original Text")
            st.text_area("Full Document", text[:3000] + "..." if len(text) > 3000 else text, height=300)

        with col2:
            st.subheader("📝 Generated Summary")
            if st.button("Summarize"):
                with st.spinner("Summarizing... Please wait!"):
                    summary = summarize_text(text, max_len=max_len, min_len=min_len)
                st.success("✅ Summary generated successfully!")
                st.text_area("Summary", summary, height=300)
                st.download_button("📥 Download Summary", data=summary, file_name="summary.txt")

                if reference_summary.strip():
                    st.subheader("📊 ROUGE Evaluation")
                    scores = compute_rouge(reference_summary, summary)
                    st.dataframe(scores)
    else:
        st.error("❌ Could not extract text from the uploaded file. Please check the file format.")

# -------------------- Footer --------------------
st.markdown("<hr><p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit, BART & ROUGE</p>", unsafe_allow_html=True)
