import os
import string
import urllib.request
import re


def tokenize_using_re(text):
    # Splits on whitespace. This method includes whitespace. PROBLEM: Punctuation is part of words.
    # result = re.split(r'(\s)', text)

    # Splits on whitespace and punctuation. This method includes whitespace and punctuation.
    result = re.split(r'([{}]|\s)'.format(re.escape(string.punctuation)), text)

    # Removes whitespace from result.
    result = [token.strip() for token in result if token.strip()]

    return result

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: word for word, i in vocab.items()}

    def encode(self, text):
        tokens = tokenize_using_re(text)
        ids = [self.str_to_int[token] for token in tokens]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove spaces before punctuation. Does not handle punctuation like '(' which needs no space after it not before it.
        pattern = r'\s+([{}])'.format(re.escape(string.punctuation))
        text = re.sub(pattern, r'\1', text)

        return text


def create_vocab_from_tokens(tokens):
    all_words = sorted(set(tokens))
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab


def main():

    # Download the text file from the internet if it doesn't exist
    file_path = "pride_and_prejudice.txt"
    if not os.path.exists(file_path):
        url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
        urllib.request.urlretrieve(url, file_path)

    # utf-8-sig is used to handle BOM (Byte Order Mark) in the file
    # BOM is a Unicode character used to signal the endianness of a text file
    with open(file_path, "r", encoding="utf-8-sig") as file:
        raw_text = file.read()
    # print(len(raw_text))
    # print(raw_text[:100])

    tokens = tokenize_using_re(raw_text)
    # print(len(tokens))
    # print(tokens[:200])

    vocab = create_vocab_from_tokens(tokens)
    # print(len(vocab))
    # print(list(vocab.keys())[:25])

    tokenizer = Tokenizer(vocab)
    text = """To me this humour seems to possess a
greater affinity, on the whole, to that of Addison than to any other of
the numerous species of this great British genus"""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
