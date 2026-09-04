from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings


# =========================================
# Vector Store Configuration
# =========================================

VECTORSTORE_DIR = Path("data/vectorstore")


# =========================================
# Create Vector Store
# =========================================

def create_vector_store(documents):
    """
    Create a FAISS vector store from document chunks.

    Embeddings are normalized so that similarity
    can be calculated consistently.
    """

    if documents is None:
        raise ValueError(
            "No documents/chunks were provided."
        )

    if not documents:
        raise ValueError(
            "No documents/chunks were provided."
        )

    # -----------------------------------------
    # Get embedding model
    # -----------------------------------------

    embeddings = get_embeddings()

    if embeddings is None:
        raise ValueError(
            "Embedding model could not be initialized."
        )

    # -----------------------------------------
    # Create normalized FAISS vector store
    # -----------------------------------------

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
        normalize_L2=True
    )

    # -----------------------------------------
    # Save vector store
    # -----------------------------------------

    save_vector_store(
        vector_store
    )

    return vector_store


# =========================================
# Save Vector Store
# =========================================

def save_vector_store(vector_store):
    """
    Save FAISS vector store to disk.
    """

    if vector_store is None:
        raise ValueError(
            "Cannot save an empty vector store."
        )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTORSTORE_DIR)
    )


# =========================================
# Check Vector Store
# =========================================

def vector_store_exists():
    """
    Check whether a valid FAISS vector store exists.
    """

    if not VECTORSTORE_DIR.exists():
        return False

    index_file = VECTORSTORE_DIR / "index.faiss"
    metadata_file = VECTORSTORE_DIR / "index.pkl"

    if not index_file.exists():
        return False

    if not metadata_file.exists():
        return False

    return True


# =========================================
# Load Vector Store
# =========================================

def load_vector_store():
    """
    Load the locally saved FAISS vector store.
    """

    if not vector_store_exists():
        return None

    # -----------------------------------------
    # Get current embedding model
    # -----------------------------------------

    embeddings = get_embeddings()

    if embeddings is None:
        raise ValueError(
            "Embedding model could not be initialized."
        )

    # -----------------------------------------
    # Load FAISS
    # -----------------------------------------

    try:

        vector_store = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:

        raise RuntimeError(
            "Failed to load the FAISS vector store. "
            f"Details: {e}"
        ) from e

    # -----------------------------------------
    # Validate vector store
    # -----------------------------------------

    if vector_store is None:
        raise ValueError(
            "FAISS vector store loaded as None."
        )

    if not hasattr(vector_store, "index"):
        raise ValueError(
            "Loaded vector store does not contain "
            "a valid FAISS index."
        )

    if vector_store.index is None:
        raise ValueError(
            "FAISS index is None."
        )

    # -----------------------------------------
    # Check number of vectors
    # -----------------------------------------

    total_vectors = int(
        vector_store.index.ntotal
    )

    if total_vectors <= 0:
        raise ValueError(
            "The FAISS vector store contains no vectors. "
            "Please process the documents again."
        )

    return vector_store