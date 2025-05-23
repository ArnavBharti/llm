import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken

class GPT2Dataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPT2Dataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main():

    file_path = "pride_and_prejudice.txt"
    if not os.path.exists(file_path):
        url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
        urllib.request.urlretrieve(url, file_path)

    # utf-8-sig is used to handle BOM (Byte Order Mark) in the file
    # BOM is a Unicode character used to signal the endianness of a text file
    with open(file_path, "r", encoding="utf-8-sig") as file:
        raw_text = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(raw_text)

    max_length = 4
    batch_size = 8
    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    for batch in dataloader:
        input_ids, target_ids = batch
        token_embeddings = token_embedding_layer(input_ids)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))
        input_embeddings = token_embeddings + pos_embeddings
        break


if __name__ == "__main__":
    main()
