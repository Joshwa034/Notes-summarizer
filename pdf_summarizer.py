import PyPDF2
from transformers import pipeline


def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):

            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def summarize_text(text):
    # Specify the model name and revision you want to use for summarization
    model_name = "t5-small"  # Replace with your preferred model name
    summarizer = pipeline("summarization", model=model_name)

    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def main(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    summary = summarize_text(text)
    print("Original Text:")
    print(text)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    pdf_file = "t.pdf"  # Replace with the path to your PDF file
    main(pdf_file)
