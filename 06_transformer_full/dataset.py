import torch

from collections import Counter

from datasets import load_from_disk
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import tokenize_en, tokenize_de


SPECIALS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:

    def __init__(self):

        self.word2idx = dict(SPECIALS)

        self.idx2word = {
            v: k for k, v in SPECIALS.items()
        }

    def build_vocab(self, tokenized_sentences):

        counter = Counter()

        for tokens in tokenized_sentences:

            counter.update(tokens)

        idx = len(self.word2idx)

        for word in counter.keys():

            if word not in self.word2idx:

                self.word2idx[word] = idx

                self.idx2word[idx] = word

                idx += 1

    def numericalize(self, tokens):

        return [

            self.word2idx.get(
                token,
                UNK_IDX
            )

            for token in tokens
        ]

    def __len__(self):

        return len(self.word2idx)

    def get_itos(self):

        return self.idx2word


class Multi30kDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset = load_from_disk(dataset_path)

        self.src_vocab = Vocabulary()

        self.tgt_vocab = Vocabulary()

        # -----------------------------------------
        # Build vocabularies
        # -----------------------------------------

        src_tokens = []
        tgt_tokens = []

        for item in self.dataset:

            src_tokens.append(
                tokenize_en(
                    item["en"]
                )
            )

            tgt_tokens.append(
                tokenize_de(
                    item["de"]
                )
            )

        self.src_vocab.build_vocab(src_tokens)

        self.tgt_vocab.build_vocab(tgt_tokens)

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        src_text = item["en"]

        tgt_text = item["de"]

        src_tokens = tokenize_en(src_text)

        tgt_tokens = tokenize_de(tgt_text)

        src_ids = [BOS_IDX]

        src_ids += self.src_vocab.numericalize(
            src_tokens
        )

        src_ids += [EOS_IDX]

        tgt_ids = [BOS_IDX]

        tgt_ids += self.tgt_vocab.numericalize(
            tgt_tokens
        )

        tgt_ids += [EOS_IDX]

        return (
            torch.tensor(src_ids),
            torch.tensor(tgt_ids)
        )


def collate_fn(batch):

    src_batch = [item[0] for item in batch]

    tgt_batch = [item[1] for item in batch]

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return src_batch, tgt_batch