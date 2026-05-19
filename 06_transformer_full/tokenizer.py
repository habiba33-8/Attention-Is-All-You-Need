import spacy


# ---------------------------------------------------
# Load spaCy Tokenizers
# ---------------------------------------------------

spacy_en = spacy.load("en_core_web_sm")

spacy_de = spacy.load("de_core_news_sm")


# ---------------------------------------------------
# English Tokenizer
# ---------------------------------------------------

def tokenize_en(text):
    """
    Tokenizes English text into tokens.

    Args:
        text (str): Input English sentence.

    Returns:
        list: List of tokens.
    """

    return [
        token.text.lower()
        for token in spacy_en.tokenizer(text)
    ]


# ---------------------------------------------------
# German Tokenizer
# ---------------------------------------------------

def tokenize_de(text):
    """
    Tokenizes German text into tokens.

    Args:
        text (str): Input German sentence.

    Returns:
        list: List of tokens.
    """

    return [
        token.text.lower()
        for token in spacy_de.tokenizer(text)
    ]


# ---------------------------------------------------
# Quick Test
# ---------------------------------------------------

if __name__ == "__main__":

    en_sentence = "A little girl is smiling."

    de_sentence = "Ein kleines Mädchen lächelt."

    print("English Tokens:")
    print(tokenize_en(en_sentence))

    print("\nGerman Tokens:")
    print(tokenize_de(de_sentence))