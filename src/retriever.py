from typing import Optional

from src.vector_store import load_vector_store


# =========================================
# Retrieval Configuration
# =========================================

DEFAULT_TOP_K = 5


# =========================================
# Validate top_k
# =========================================

def validate_top_k(
    top_k: Optional[int]
) -> int:
    """
    Make sure top_k is always a valid
    positive integer.
    """

    if top_k is None:
        return DEFAULT_TOP_K

    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K

    if value <= 0:
        return DEFAULT_TOP_K

    return value


# =========================================
# Get Retriever
# =========================================

def get_retriever(
    top_k: Optional[int] = DEFAULT_TOP_K
):
    """
    Create a LangChain FAISS retriever.
    """

    top_k = validate_top_k(top_k)

    vector_store = load_vector_store()

    if vector_store is None:
        raise ValueError(
            "Vector store could not be loaded. "
            "Please process the documents first."
        )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k
        }
    )


# =========================================
# Create Contextual Retrieval Query
# =========================================

def create_contextual_query(
    question: str,
    chat_history=None
) -> str:
    """
    Create a retrieval query using the current
    question and the most recent conversation.

    This helps FAISS understand follow-up questions
    such as:

        User: What is RAG?
        User: What are its main components?

    The second question can be searched together
    with the previous question.
    """

    # -----------------------------------------
    # Validate current question
    # -----------------------------------------

    if question is None:
        return ""

    if not isinstance(question, str):
        question = str(question)

    question = question.strip()

    if not question:
        return ""

    # -----------------------------------------
    # No history
    # -----------------------------------------

    if not chat_history:
        return question

    # -----------------------------------------
    # Find previous user messages
    # -----------------------------------------

    previous_user_messages = []

    for message in chat_history:

        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role != "user":
            continue

        content = message.get("content")

        if not content:
            continue

        content = str(content).strip()

        if not content:
            continue

        previous_user_messages.append(
            content
        )

    # -----------------------------------------
    # No previous user questions
    # -----------------------------------------

    if not previous_user_messages:
        return question

    # -----------------------------------------
    # Use the most recent previous question
    # -----------------------------------------

    previous_question = (
        previous_user_messages[-1]
    )

    # -----------------------------------------
    # Build contextual retrieval query
    # -----------------------------------------

    contextual_query = (
        f"Previous question: {previous_question}\n"
        f"Current question: {question}"
    )

    return contextual_query


# =========================================
# Retrieve Documents
# =========================================

def retrieve_documents(
    question: str,
    top_k: Optional[int] = DEFAULT_TOP_K,
    chat_history=None
):
    """
    Retrieve documents from FAISS.

    Supports conversation-aware retrieval.

    FAISS returns squared L2 distances.

    Because the vector store uses normalized
    embeddings:

        cosine_similarity = 1 - distance / 2

    This produces a similarity score where:

        1.0 = very similar
        0.0 = unrelated
        negative = very dissimilar

    Returns:

        [
            (Document, similarity_score),
            ...
        ]
    """

    # =========================================
    # Validate Question
    # =========================================

    if question is None:
        return []

    if not isinstance(question, str):
        question = str(question)

    question = question.strip()

    if not question:
        return []

    # =========================================
    # Validate top_k
    # =========================================

    top_k = validate_top_k(
        top_k
    )

    # =========================================
    # Create Contextual Query
    # =========================================

    retrieval_query = create_contextual_query(
        question=question,
        chat_history=chat_history
    )

    if not retrieval_query:
        return []

    # =========================================
    # Load Vector Store
    # =========================================

    vector_store = load_vector_store()

    if vector_store is None:
        raise ValueError(
            "Vector store could not be loaded. "
            "Please process the documents first."
        )

    # =========================================
    # Validate FAISS Index
    # =========================================

    if not hasattr(
        vector_store,
        "index"
    ):
        raise ValueError(
            "FAISS vector store does not contain "
            "an index."
        )

    if vector_store.index is None:
        raise ValueError(
            "FAISS index is None."
        )

    # =========================================
    # Get Number of Vectors
    # =========================================

    total_vectors = int(
        vector_store.index.ntotal
    )

    if total_vectors <= 0:
        return []

    # =========================================
    # Calculate Search K
    # =========================================

    search_k = min(
        top_k,
        total_vectors
    )

    if search_k <= 0:
        return []

    # =========================================
    # Debug Contextual Query
    # =========================================

    print()
    print("========================================")
    print("CONTEXTUAL RETRIEVAL")
    print("========================================")
    print("Original question:")
    print(question)
    print()
    print("Retrieval query:")
    print(retrieval_query)
    print()
    print("Requested top_k:", top_k)
    print("Actual search_k:", search_k)
    print("Total vectors:", total_vectors)
    print("========================================")
    print()

    # =========================================
    # Perform FAISS Similarity Search
    # =========================================

    results = (
        vector_store
        .similarity_search_with_score(
            query=retrieval_query,
            k=search_k
        )
    )

    if results is None:
        return []

    # =========================================
    # Debug Information
    # =========================================

    print()
    print("========================================")
    print("RAG RETRIEVAL DEBUG")
    print("========================================")
    print("Question:", question)
    print("Retrieval query:", retrieval_query)
    print("Requested top_k:", top_k)
    print("Actual search_k:", search_k)
    print("Total vectors:", total_vectors)
    print("Number of results:", len(results))
    print()

    # =========================================
    # Process Results
    # =========================================

    cleaned_results = []

    for index, result in enumerate(results):

        if result is None:
            continue

        if not isinstance(
            result,
            tuple
        ):
            continue

        if len(result) != 2:
            continue

        document, distance = result

        if document is None:
            continue

        if distance is None:
            continue

        try:
            distance = float(
                distance
            )
        except (
            TypeError,
            ValueError
        ):
            continue

        # -----------------------------------------
        # Convert normalized L2 distance
        # to cosine similarity
        #
        # distance = 2 - 2*cosine
        #
        # cosine = 1 - distance/2
        # -----------------------------------------

        similarity = (
            1.0 - (distance / 2.0)
        )

        # -----------------------------------------
        # Keep similarity in safe range
        # -----------------------------------------

        similarity = max(
            -1.0,
            min(
                1.0,
                similarity
            )
        )

        # -----------------------------------------
        # Get Metadata
        # -----------------------------------------

        metadata = getattr(
            document,
            "metadata",
            {}
        )

        if metadata is None:
            metadata = {}

        # -----------------------------------------
        # Debug output
        # -----------------------------------------

        print(
            f"Result {index + 1}"
        )

        print(
            f"Raw distance: {distance:.6f}"
        )

        print(
            f"Similarity score: {similarity:.6f}"
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown"
            )
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "Unknown"
            )
        )

        print(
            "----------------------------------------"
        )

        # -----------------------------------------
        # Store document and similarity
        # -----------------------------------------

        cleaned_results.append(
            (
                document,
                similarity
            )
        )

    print(
        "========================================"
    )
    print()

    return cleaned_results