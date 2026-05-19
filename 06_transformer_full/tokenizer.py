import spacy


spacy_en = spacy.load("en_core_web_sm")

spacy_de = spacy.load("de_core_news_sm")


def tokenize_en(text):

    return [
        token.text.lower()
        for token in spacy_en.tokenizer(text)
    ]


def tokenize_de(text):
    
    return [
        token.text.lower()
        for token in spacy_de.tokenizer(text)
    ]

if __name__ == "__main__":

    en_sentence = "A little girl is smiling."

    de_sentence = "Ein kleines Mädchen lächelt."

    print("English Tokens:")
    print(tokenize_en(en_sentence))

    print("\nGerman Tokens:")
    print(tokenize_de(de_sentence))
