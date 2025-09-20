import streamlit as st
from transformers import pipeline, AutoTokenizer

# -------------------- Load Model --------------------
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------- Helper Functions --------------------
def chunk_text(text, max_tokens=900):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

def summarize_text(text, max_len=200, min_len=50):
    tokens = tokenizer.encode(text, truncation=True, max_length=1024)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    if len(tokens) >= 1024:
        summaries = []
        for chunk in chunk_text(truncated_text):
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

# -------------------- Page Config --------------------
st.set_page_config(page_title="AI Document Summarizer", page_icon="📄", layout="wide")

# -------------------- Add Background Image --------------------
background_image_url = "https://lh3.googleusercontent.com/gg-dl/AJfQ9KRpv8UGyeTSPPz6L4nlyYmIDAPsrwv6SJUEtaUpF3En2mA8hwZZDT2mAuP8oNJrqQ3iHYrzzNJ2nboiRA-xTm8CyjMLTiEwVnWEdV8QkGXUrI1gD33RXmDwppj4ciF3L_9ZE6ERljvtqg7OwnAZvriyah6JbgKoYEjGrpdD6pID5-YdNw=s1024"  # You can replace with any link

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
.block-container {{
    background: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 12px;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("⚙️ Settings")
    max_len = st.slider("Max Summary Length", 50, 500, 200)
    min_len = st.slider("Min Summary Length", 10, 100, 50)
    st.markdown("---")
    st.info("Upload your text file to summarize!")

# -------------------- Header --------------------
st.markdown(
    "<h1 style='text-align: center; color: #2F4F4F;'>📄 AI Document Summarization Tool</h1>"
    "<p style='text-align: center; color: gray;'>Summarize long documents into concise, clear summaries.</p>",
    unsafe_allow_html=True
)

# -------------------- Upload Section --------------------
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# -------------------- Layout: Two Columns --------------------
col1, col2 = st.columns(2)

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    with col1:
        st.subheader("📜 Original Text")
        st.text_area("Full Document", text[:3000] + "..." if len(text) > 3000 else text, height=300)

    with col2:
        st.subheader("📝 Generated Summary")
        if st.button("Summarize"):
            with st.spinner("Summarizing... Please wait!"):
                summary = summarize_text(text, max_len=max_len, min_len=min_len)
            st.success("Summary generated successfully!")
            st.text_area("Summary", summary, height=300)
            st.download_button("📥 Download Summary", data=summary, file_name="summary.txt")

# -------------------- Footer --------------------
st.markdown("<hr><p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & BART</p>", unsafe_allow_html=True)
