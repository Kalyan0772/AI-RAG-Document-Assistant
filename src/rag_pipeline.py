# =========================================
# DOCUMIND AI - RAG Pipeline
# =========================================

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from src.retriever import retrieve_documents
from src.llm import get_llm
from src.prompt import SYSTEM_PROMPT
from src.citations import create_citations

from config.config import RELEVANCE_THRESHOLD


DEFAULT_TOP_K = 5


def validate_top_k(top_k):
    """
    Validate the number of documents to retrieve.
    """

    if top_k is None:
        return DEFAULT_TOP_K

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K

    if top_k <= 0:
        return DEFAULT_TOP_K

    return top_k


def create_guardrail_message():
    """
    Standard response when the uploaded documents
    do not contain enough relevant information.
    """

    return (
        "I couldn't find enough relevant information "
        "in the uploaded documents to answer this question."
    )


def format_chat_history(chat_history):
    """
    Convert Streamlit chat history into a compact text
    representation for the RAG prompt.

    Only previous user and assistant messages are included.
    """

    if not chat_history:
        return "No previous conversation."

    history_parts = []

    for message in chat_history:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if not role or not content:
            continue

        content = str(content).strip()

        if not content:
            continue

        if role == "user":
            history_parts.append(
                f"USER: {content}"
            )

        elif role == "assistant":
            history_parts.append(
                f"ASSISTANT: {content}"
            )

    if not history_parts:
        return "No previous conversation."

    return "\n\n".join(history_parts)
    
def create_retrieval_query(question, chat_history):
    """
    Create a retrieval-friendly query by combining the
    current question with relevant previous conversation.

    This helps resolve follow-up questions containing
    references such as:
    - it
    - its
    - they
    - this
    - that
    - the above
    """

    if not chat_history:
        return question

    previous_user_questions = []

    for message in chat_history:

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = message.get("content")

        if not content:
            continue

        content = str(content).strip()

        if not content:
            continue

        previous_user_questions.append(content)

    if not previous_user_questions:
        return question

    # Use the most recent previous user question.
    previous_question = previous_user_questions[-1]

    retrieval_query = (
        f"Previous question: {previous_question}\n"
        f"Current question: {question}"
    )

    return retrieval_query

def filter_relevant_documents(results):
    """
    Filter retrieved documents using the configured
    relevance threshold.
    """

    relevant_documents = []

    if not results:
        return relevant_documents

    print()
    print("========================================")
    print("RELEVANCE FILTER")
    print("========================================")
    print(
        "Relevance threshold:",
        RELEVANCE_THRESHOLD
    )
    print()

    for index, item in enumerate(results):

        if item is None:
            continue

        if not isinstance(item, tuple):
            continue

        if len(item) != 2:
            continue

        document, score = item

        if document is None:
            continue

        if score is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        print(
            f"Document {index + 1} "
            f"score: {score:.6f}"
        )

        if score >= RELEVANCE_THRESHOLD:

            relevant_documents.append(document)

            print("Status: ACCEPTED")

        else:

            print("Status: REJECTED")

    print()
    print(
        "Accepted documents:",
        len(relevant_documents)
    )

    print("========================================")
    print()

    return relevant_documents


def build_context(documents):
    """
    Build the document context that will be sent
    to the LLM.
    """

    if not documents:
        return ""

    context_parts = []

    for document in documents:

        metadata = getattr(
            document,
            "metadata",
            {}
        ) or {}

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        text = getattr(
            document,
            "page_content",
            ""
        ) or ""

        if not text.strip():
            continue

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{text}
"""
        )

    return "\n\n".join(context_parts)


def answer_question(
    question,
    top_k=DEFAULT_TOP_K,
    chat_history=None
):
    """
    Complete RAG pipeline with conversation memory:

    1. Validate question
    2. Read previous conversation
    3. Retrieve relevant documents
    4. Filter weak results
    5. Build document context
    6. Build conversation context
    7. Apply grounding guardrails
    8. Build RAG prompt
    9. Send context + history + question to LLM
    10. Generate grounded answer
    11. Create citations
    12. Return final result
    """

    # -----------------------------------------
    # STEP 1: Validate question
    # -----------------------------------------

    if question is None:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved_documents": []
        }

    if not isinstance(question, str):

        question = str(question)

    question = question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved_documents": []
        }

    # -----------------------------------------
    # STEP 2: Validate top_k
    # -----------------------------------------

    top_k = validate_top_k(top_k)

    # -----------------------------------------
    # STEP 3: Format conversation history
    # -----------------------------------------

    chat_history_text = format_chat_history(
        chat_history
    )

    print()
    print("========================================")
    print("CONVERSATION MEMORY")
    print("========================================")
    print(
        "Previous messages:",
        len(chat_history) if chat_history else 0
    )
    print("========================================")
    print()

    # -----------------------------------------
    # STEP 4: Retrieve documents
    # -----------------------------------------

    try:

        results = retrieve_documents(
            question=question,
            top_k=top_k
        )

    except Exception as e:

        print("RETRIEVAL ERROR:", e)

        return {
            "answer": (
                "An error occurred while searching "
                "the uploaded documents."
            ),
            "sources": [],
            "retrieved_documents": []
        }

    # -----------------------------------------
    # STEP 5: Check retrieval results
    # -----------------------------------------

    if not results:

        print("No documents were retrieved.")

        return {
            "answer": create_guardrail_message(),
            "sources": [],
            "retrieved_documents": []
        }

    # -----------------------------------------
    # DEBUG INFORMATION
    # -----------------------------------------

    print()
    print("========================================")
    print("DOCUMIND AI - RAG PIPELINE")
    print("========================================")
    print("Question:", question)
    print("Retrieved documents:", len(results))
    print(
        "Relevance threshold:",
        RELEVANCE_THRESHOLD
    )
    print()

    # -----------------------------------------
    # STEP 6: Filter relevant documents
    # -----------------------------------------

    relevant_documents = filter_relevant_documents(
        results
    )

    # -----------------------------------------
    # STEP 7: Guardrail
    # -----------------------------------------

    if not relevant_documents:

        print("GUARDRAIL ACTIVATED")
        print(
            "No retrieved document passed "
            "the relevance threshold."
        )
        print()

        return {
            "answer": create_guardrail_message(),
            "sources": [],
            "retrieved_documents": results
        }

    # -----------------------------------------
    # STEP 8: Build document context
    # -----------------------------------------

    context = build_context(
        relevant_documents
    )

    # -----------------------------------------
    # STEP 9: Check context
    # -----------------------------------------

    if not context.strip():

        print("GUARDRAIL ACTIVATED")

        print(
            "Relevant documents were found, "
            "but no readable text was available."
        )

        return {
            "answer": (
                "Relevant documents were found, "
                "but no readable text was available."
            ),
            "sources": [],
            "retrieved_documents": results
        }

    # -----------------------------------------
    # DEBUG CONTEXT
    # -----------------------------------------

    print("========================================")
    print("CONTEXT SENT TO LLM")
    print("========================================")
    print(context[:5000])
    print("========================================")
    print()

    # -----------------------------------------
    # STEP 10: Create RAG prompt
    # -----------------------------------------

    prompt = SYSTEM_PROMPT.format(
        context=context,
        chat_history=chat_history_text,
        question=question
    )

    # -----------------------------------------
    # STEP 11: Load LLM
    # -----------------------------------------

    try:

        llm = get_llm()

    except Exception as e:

        print(
            "LLM INITIALIZATION ERROR:",
            e
        )

        return {
            "answer": (
                "The AI model could not be initialized. "
                f"Details: {e}"
            ),
            "sources": [],
            "retrieved_documents": results
        }

    # -----------------------------------------
    # STEP 12: Generate grounded answer
    # -----------------------------------------

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content=prompt
                ),
                HumanMessage(
                    content=question
                )
            ]
        )

    except Exception as e:

        print("LLM ERROR:", e)

        return {
            "answer": (
                "The AI model encountered an error "
                "while generating the answer. "
                f"Details: {e}"
            ),
            "sources": [],
            "retrieved_documents": results
        }

    # -----------------------------------------
    # STEP 13: Extract answer
    # -----------------------------------------

    answer = getattr(
        response,
        "content",
        None
    )

    if answer is None:

        answer = str(response)

    answer = str(answer).strip()

    if not answer:

        answer = (
            "The AI model did not return an answer."
        )

    # -----------------------------------------
    # STEP 14: Create citations
    # -----------------------------------------

    try:

        sources = create_citations(
            relevant_documents
        )

    except Exception as e:

        print(
            "CITATION ERROR:",
            e
        )

        sources = []

    # -----------------------------------------
    # STEP 15: Final result
    # -----------------------------------------

    print("========================================")
    print("RAG RESPONSE GENERATED")
    print("========================================")
    print(
        "Conversation memory:",
        "USED" if chat_history else "NOT USED"
    )
    print("Citations:", len(sources))
    print("========================================")
    print()

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_documents": results
    }