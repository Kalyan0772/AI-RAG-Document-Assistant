from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(pages, document_name):
    """
    Convert extracted PDF pages into LangChain Documents
    with useful metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    documents = []

    for page in pages:

        page_text = page["text"].strip()

        if not page_text:
            continue

        page_documents = splitter.create_documents(
            [page_text]
        )

        for chunk in page_documents:

            chunk.metadata = {
                "source": document_name,
                "page": page["page_number"]
            }

            documents.append(chunk)

    # Add unique chunk IDs
    for index, document in enumerate(documents, start=1):

        document.metadata["chunk_id"] = index

    return documents