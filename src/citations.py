# =========================================
# DOCUMIND AI - Citation Generator
# =========================================


def create_citations(documents):
    """
    Create structured citations from retrieved documents.

    Each citation is returned as a dictionary:

    {
        "source": "document.pdf",
        "page": 1
    }

    This format is used by app.py when displaying
    document sources.
    """

    if not documents:
        return []

    citations = []

    for document in documents:

        if document is None:
            continue

        # -----------------------------------------
        # Get document metadata
        # -----------------------------------------

        metadata = getattr(
            document,
            "metadata",
            {}
        ) or {}

        # -----------------------------------------
        # Get source
        # -----------------------------------------

        source = metadata.get(
            "source",
            "Unknown document"
        )

        # -----------------------------------------
        # Get page
        # -----------------------------------------

        page = metadata.get(
            "page",
            None
        )

        # -----------------------------------------
        # Clean source path
        # -----------------------------------------

        if source:

            source = str(source)

            # Convert Windows path to normal path
            source = source.replace("\\", "/")

            # Keep only filename
            if "/" in source:
                source = source.split("/")[-1]

        else:

            source = "Unknown document"

        # -----------------------------------------
        # Convert page to integer when possible
        # -----------------------------------------

        if page is not None:

            try:

                page = int(page)

            except (TypeError, ValueError):

                pass

        # -----------------------------------------
        # Create structured citation
        # -----------------------------------------

        citation = {
            "source": source,
            "page": page
        }

        # -----------------------------------------
        # Avoid duplicate citations
        # -----------------------------------------

        duplicate = False

        for existing in citations:

            if (
                existing["source"] == citation["source"]
                and existing["page"] == citation["page"]
            ):
                duplicate = True
                break

        if not duplicate:

            citations.append(citation)

    return citations