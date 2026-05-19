import os
import torch

from transformer import Transformer

from tokenizer import tokenize_en

from masking import (
    create_padding_mask,
    create_look_ahead_mask
)

from dataset import (
    Multi30kDataset,
    BOS_IDX,
    EOS_IDX,
    PAD_IDX
)

from checkpoint_io import load_state_dict


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/multi30k_local_data/train"
)

dataset = Multi30kDataset(dataset_path)


SRC_VOCAB_SIZE = len(dataset.src_vocab)

TGT_VOCAB_SIZE = len(dataset.tgt_vocab)


model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    max_len=5000,
    dropout=0.1
).to(DEVICE)

checkpoint_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "transformer.pt"
)

model.load_state_dict(
    load_state_dict(checkpoint_path, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully.")


# ---------------------------------------------------
# Translation Function
# ---------------------------------------------------

def translate(sentence, max_len=50):

    # -----------------------------------------
    # Tokenize source sentence
    # -----------------------------------------

    src_tokens = tokenize_en(sentence)

    # -----------------------------------------
    # Convert tokens to IDs
    # -----------------------------------------

    src_ids = [BOS_IDX]
    src_ids += dataset.src_vocab.numericalize(src_tokens)
    src_ids += [EOS_IDX]

    # -----------------------------------------
    # Create source tensor
    # -----------------------------------------

    src_tensor = torch.tensor(
        src_ids
    ).unsqueeze(0).to(DEVICE)

    # -----------------------------------------
    # Source mask
    # -----------------------------------------

    src_mask = create_padding_mask(
        src_tensor,
        PAD_IDX
    ).to(DEVICE)

    # -----------------------------------------
    # Start decoder with <BOS>
    # -----------------------------------------

    generated_tokens = [BOS_IDX]

    # -----------------------------------------
    # Autoregressive decoding
    # -----------------------------------------

    for _ in range(max_len):

        tgt_tensor = torch.tensor(
            generated_tokens
        ).unsqueeze(0).to(DEVICE)

        # -------------------------------------
        # Target mask
        # -------------------------------------

        tgt_mask = create_look_ahead_mask(
            tgt_tensor.size(1)
        ).to(DEVICE)

        # -------------------------------------
        # Forward pass
        # -------------------------------------

        with torch.no_grad():

            output = model(
                src_tensor,
                tgt_tensor,
                src_mask,
                tgt_mask
            )

        # -------------------------------------
        # Get next token
        # -------------------------------------

        next_token = output[
            :, -1
        ].argmax(dim=-1).item()

        generated_tokens.append(next_token)

        # -------------------------------------
        # Stop if EOS generated
        # -------------------------------------

        if next_token == EOS_IDX:
            break

    # ---------------------------------------------------
    # Convert IDs back to words
    # ---------------------------------------------------

    predicted_tokens = []

    vocab_itos = dataset.tgt_vocab.get_itos()

    for idx in generated_tokens:

        token = vocab_itos[idx]

        if token not in ["<BOS>", "<EOS>", "<PAD>"]:

            predicted_tokens.append(token)

    # ---------------------------------------------------
    # Final sentence
    # ---------------------------------------------------

    translated_sentence = " ".join(
        predicted_tokens
    )

    return translated_sentence


# ---------------------------------------------------
# Interactive Inference
# ---------------------------------------------------

if __name__ == "__main__":

    while True:

        sentence = input(
            "\nEnter English sentence: "
        )

        if sentence.lower() == "exit":
            break

        translation = translate(sentence)

        print("\nGerman Translation:")
        print(translation)