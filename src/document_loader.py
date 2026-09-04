import fitz


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.
    """

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return pages


def get_document_statistics(pages):
    """
    Calculate basic statistics about extracted document text.
    """

    total_pages = len(pages)

    total_characters = sum(
        len(page["text"]) for page in pages
    )

    total_words = sum(
        len(page["text"].split())
        for page in pages
    )

    return {
        "pages": total_pages,
        "characters": total_characters,
        "words": total_words
    }