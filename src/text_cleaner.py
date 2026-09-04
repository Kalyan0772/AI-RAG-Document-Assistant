import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful information.
    """

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    text = text.strip()

    return text