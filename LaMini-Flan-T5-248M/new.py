import nltk
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize
import PyPDF2

nltk.download("punkt")

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_document(file_path):
    if file_path.endswith(".txt"):
        return read_text_file(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type. Only .txt and .pdf are supported.")

def chunk_text(text, max_tokens=1024):
    sentences = sent_tokenize(text)
    current_chunk = []
    current_len = 0
    chunks = []

    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_len + len(tokens) <= max_tokens:
            current_chunk.append(sentence)
            current_len += len(tokens)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(tokens)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def summarize_chunk(chunk):
    inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, min_length=30, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_document(text):
    chunks = chunk_text(text)
    print(f"🔍 Total Chunks: {len(chunks)}\n")
    
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    
    combined_summary = " ".join(summaries)
    if len(tokenizer.encode(combined_summary)) > 1024:
        print("\n⚠️ Second-pass summarization due to length...")
        return summarize_chunk(combined_summary)
    
    return combined_summary

# === Example Usage ===
file_path = "your_legal_document.pdf"  # or "your_legal_document.txt"
try:
    legal_text = load_document(file_path)
    final_summary = summarize_document(legal_text)
    print("\n📄 Final Summary:\n")
    print(final_summary)
except Exception as e:
    print(f"Error: {e}")