import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load the PDF document
def load_pdf(file):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(file)  # Using PdfReader
    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num]
        pdf_text += page_obj.extract_text()
    return pdf_text

# Load the pre-trained model and tokenizer (LLaMA)
def load_model():
    model_name = "Meta AI/llama-base"  # Use LLaMA-based model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Process the user's question
def process_question(question, context, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=1024)
    return inputs

# Find the answer in the PDF text
def find_answer(model, inputs, tokenizer):
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = inputs["input_ids"][0][answer_start:answer_end]
    return tokenizer.decode(answer, skip_special_tokens=True)

# Split the PDF text into manageable chunks
def chunk_text(text, chunk_size=1024):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Main application
def main():
    st.title("PDF Chatbot with Question Answering")
    
    # File upload
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if file:
        # Load model
        model, tokenizer = load_model()

        # Extract text from the uploaded PDF
        pdf_text = load_pdf(file)
        
        # Process questions
        question = st.text_input("Ask a question")
        if question:
            # Handle PDF chunking for long PDFs
            text_chunks = chunk_text(pdf_text)
            
            # Iterate over chunks and try to find an answer
            answer = None
            for chunk in text_chunks:
                inputs = process_question(question, chunk, tokenizer)
                answer = find_answer(model, inputs, tokenizer)
                if answer:
                    break

            if answer:
                st.write("Answer:", answer)
            else:
                st.write("Sorry, I couldn't find an answer in the document.")

# Run the application
if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    main()