import streamlit as st

from pathlib import Path
import sys
import os

# --------------------------------------------------
# Project setup
# --------------------------------------------------

ROOT_DIR = Path(__file__).parent

sys.path.append(str(ROOT_DIR))


# --------------------------------------------------
# Imports
# --------------------------------------------------

from src.document_loader import (
    extract_text_from_pdf,
    get_document_statistics
)

from src.text_cleaner import clean_text

from src.chunker import create_chunks

from src.vector_store import create_vector_store

from src.rag_pipeline import answer_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Load CSS
# --------------------------------------------------

css_path = ROOT_DIR / "ui" / "styles.css"

with open(
    css_path,
    "r",
    encoding="utf-8"
) as css_file:

    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "documents" not in st.session_state:

    st.session_state.documents = []


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "indexed" not in st.session_state:

    st.session_state.indexed = False


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "total_questions" not in st.session_state:

    st.session_state.total_questions = 0


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">

        <div style="font-size:55px;">
        </div>

        <h1>DOCUMIND AI</h1>

        <p style="color:#94a3b8;">
        Intelligent Document Assistant
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Documents",
            "AI Chat",
            "Analytics",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Status

    if st.session_state.indexed:

        st.success(
            "Knowledge Base Ready"
        )

    else:

        st.info(
            "Knowledge Base Not Indexed"
        )


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">DOCUMIND AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your intelligent assistant for understanding documents'
        '</div>',
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    total_documents = len(
        st.session_state.documents
    )

    total_pages = sum(
        document["statistics"]["pages"]
        for document in st.session_state.documents
    )

    total_words = sum(
        document["statistics"]["words"]
        for document in st.session_state.documents
    )

    total_chunks = len(
        st.session_state.chunks
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-number">
            {total_documents}
            </div>

            <div class="stat-label">
            Documents
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-number">
            {total_pages}
            </div>

            <div class="stat-label">
            Pages
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-number">
            {total_chunks}
            </div>

            <div class="stat-label">
            Chunks
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-number">
            {st.session_state.total_questions}
            </div>

            <div class="stat-label">
            Questions
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Upload
    # ----------------------------------------------

    st.markdown(
        """
        <div class="glass-card">

        <h2> Upload Your Documents</h2>

        <p style="color:#94a3b8;">
        Upload one or multiple PDF documents to create
        your private AI knowledge base.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.markdown(
            "### Selected Documents"
        )

        for uploaded_file in uploaded_files:

            upload_path = (
                ROOT_DIR
                / "data"
                / "uploads"
                / uploaded_file.name
            )

            # Save file

            with open(
                upload_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            # Avoid duplicate documents

            existing_names = [
                document["name"]
                for document in
                st.session_state.documents
            ]

            if uploaded_file.name in existing_names:

                continue

            # --------------------------------------
            # Extract PDF
            # --------------------------------------

            try:

                pages = extract_text_from_pdf(
                    upload_path
                )

                # Clean text

                for page_data in pages:

                    page_data["text"] = clean_text(
                        page_data["text"]
                    )

                statistics = get_document_statistics(
                    pages
                )

                st.session_state.documents.append(
                    {
                        "name": uploaded_file.name,
                        "path": str(upload_path),
                        "pages": pages,
                        "statistics": statistics
                    }
                )

                st.success(
                    f"Processed {uploaded_file.name}"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Pages",
                    statistics["pages"]
                )

                c2.metric(
                    "Words",
                    f"{statistics['words']:,}"
                )

                c3.metric(
                    "Characters",
                    f"{statistics['characters']:,}"
                )

            except Exception as error:

                st.error(
                    f"Error processing "
                    f"{uploaded_file.name}: {error}"
                )

    # ----------------------------------------------
    # Build knowledge base
    # ----------------------------------------------

    if st.session_state.documents:

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.markdown(
            "### Build AI Knowledge Base"
        )

        if st.button(
            "Process Documents & Build Knowledge Base",
            use_container_width=True
        ):

            with st.status(
                "Building your AI knowledge base...",
                expanded=True
            ):

                st.write(
                    "Preparing documents..."
                )

                all_chunks = []

                for document in st.session_state.documents:

                    st.write(
                        f"Chunking {document['name']}..."
                    )

                    chunks = create_chunks(
                        document["pages"],
                        document["name"]
                    )

                    all_chunks.extend(
                        chunks
                    )

                st.write(
                    f"Created {len(all_chunks)} chunks."
                )

                st.write(
                    "Generating embeddings..."
                )

                vector_store = create_vector_store(
                    all_chunks
                )

                st.session_state.chunks = (
                    all_chunks
                )

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.indexed = True

                st.write(
                    "Building semantic search index..."
                )

                st.write(
                    "Knowledge base ready!"
                )

            st.success(
                f"Successfully indexed "
                f"{len(all_chunks)} chunks."
            )


# ==================================================
# DOCUMENTS
# ==================================================

elif page == "Documents":

    st.markdown(
        '<div class="main-title">'
        'DOCUMENT LIBRARY'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your indexed document knowledge base'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.documents:

        st.info(
            "No documents uploaded yet."
        )

    else:

        for document in st.session_state.documents:

            statistics = document[
                "statistics"
            ]

            st.markdown(
                f"""
                <div class="document-card">

                <h3>
                {document["name"]}
                </h3>

                <p style="color:#94a3b8;">
                {statistics["pages"]} pages
                • {statistics["words"]:,} words
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(
                "View Extracted Text"
            ):

                for page_data in document["pages"]:

                    st.markdown(
                        f"#### Page {page_data['page_number']}"
                    )

                    text = page_data["text"]

                    if text:

                        st.write(
                            text[:3000]
                        )

                    else:

                        st.warning(
                            "No text found on this page."
                        )

            st.divider()


# ==================================================
# AI CHAT
# ==================================================

elif page == "AI Chat":

    st.markdown(
        '<div class="main-title">'
        'AI DOCUMENT ASSISTANT'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions and get grounded answers from your documents'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.indexed:

        st.warning(
            "Please upload documents and build the "
            "AI Knowledge Base first."
        )

    else:

        # ------------------------------------------
        # Display chat history
        # ------------------------------------------

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    st.markdown(
                        "#### Sources"
                    )

                    for source in message[
                        "sources"
                    ]:

                        st.caption(
                            f"{source['source']} "
                            f"• Page {source['page']}"
                        )

        # ------------------------------------------
        # Chat input
        # ------------------------------------------

        question = st.chat_input(
            "Ask something about your documents..."
        )

        if question:

            # User message

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            # AI response

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching your documents..."
                ):

                    try:

                        result = answer_question(
                            question=question,
                            chat_history=st.session_state.chat_history[:-1]
                        )

                        answer = result[
                            "answer"
                        ]

                        sources = result[
                            "sources"
                        ]

                        st.markdown(
                            answer
                        )

                        if sources:

                            st.markdown(
                                "#### Sources"
                            )

                            for source in sources:

                                st.caption(
                                    f"{source['source']} "
                                    f"• Page "
                                    f"{source['page']}"
                                )

                        # Save AI response

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )

                        st.session_state.total_questions += 1

                    except Exception as error:

                        st.error(
                            f"Error while answering: "
                            f"{error}"
                        )


# ==================================================
# ANALYTICS
# ==================================================

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">'
        'DOCUMENT ANALYTICS'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Knowledge base statistics and system performance'
        '</div>',
        unsafe_allow_html=True
    )

    documents = len(
        st.session_state.documents
    )

    pages = sum(
        document["statistics"]["pages"]
        for document in
        st.session_state.documents
    )

    words = sum(
        document["statistics"]["words"]
        for document in
        st.session_state.documents
    )

    chunks = len(
        st.session_state.chunks
    )

    questions = st.session_state.total_questions

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Documents",
            documents
        )

        st.metric(
            "Pages",
            pages
        )

        st.metric(
            "Words",
            f"{words:,}"
        )

    with c2:

        st.metric(
            "Chunks",
            chunks
        )

        st.metric(
            "Questions",
            questions
        )

        status = (
            "Ready"
            if st.session_state.indexed
            else "Not Ready"
        )

        st.metric(
            "Knowledge Base",
            status
        )

    st.markdown(
        "### Document Breakdown"
    )

    for document in st.session_state.documents:

        statistics = document[
            "statistics"
        ]

        st.write(
            f" **{document['name']}** — "
            f"{statistics['pages']} pages, "
            f"{statistics['words']:,} words"
        )


# ==================================================
# SETTINGS
# ==================================================

elif page == "Settings":

    st.markdown(
        '<div class="main-title">'
        'RAG SETTINGS'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Configure how your document retrieval system works'
        '</div>',
        unsafe_allow_html=True
    )

    st.slider(
        "Chunk Size",
        200,
        2000,
        1000,
        100
    )

    st.slider(
        "Chunk Overlap",
        0,
        500,
        200,
        50
    )

    st.slider(
        "Top-K Retrieved Chunks",
        1,
        10,
        5
    )

    st.slider(
        "LLM Temperature",
        0.0,
        1.0,
        0.2,
        0.1
    )

    st.info(
        "These settings control the document chunking, "
        "retrieval and generation behavior."
    )

    if st.session_state.indexed:

        st.success(
            "Knowledge base is currently indexed."
        )

    else:

        st.warning(
            "Build the knowledge base from the Dashboard."
        )