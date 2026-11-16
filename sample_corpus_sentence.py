import os
import re
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np

# ---------------------------------------------------------
# Causal Self-Attention & Transformer Architecture
# ---------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=3, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, d_model),
            'wpe': nn.Embedding(block_size, d_model),
            'h': nn.ModuleList([Block(d_model, n_head, block_size) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(d_model)
        })
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# ---------------------------------------------------------
# Sentence Selection & Completion Generation
# ---------------------------------------------------------

def select_valid_sentence_and_beginning(corpus_text, orig2internal, enc):
    # Split text into candidate sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', corpus_text)
    
    # Filter for clean, short sentences (between 5 and 20 words)
    candidates = []
    for s in raw_sentences:
        clean = s.strip()
        words = clean.split()
        if 5 <= len(words) <= 20 and clean[0].isupper() and clean.endswith(('.', '!', '?')):
            candidates.append(clean)

    # Sort candidates putting common openings first
    common_openings = ("If ", "Why ", "The ", "You ", "We ", "It ", "A ", "In ", "For ")
    candidates.sort(key=lambda s: 0 if s.startswith(common_openings) else 1)

    for sentence in candidates:
        words = sentence.split()
        # Take first 3-5 words as sentence beginning
        num_prefix_words = min(4, len(words) - 1)
        beginning = " ".join(words[:num_prefix_words])

        # Tokenize sentence beginning using o200k_base
        toks = enc.encode(beginning)

        # Check that ALL tokens exist in the saved model vocabulary
        if all(tok in orig2internal for tok in toks):
            return sentence, beginning

    raise RuntimeError("No candidate sentence found where all prompt tokens exist in the model vocabulary.")


def generate_completion(model, prompt_str, orig2internal, internal2orig, enc, device, max_new_tokens=30, block_size=64):
    model.eval()
    
    prompt_tokens_orig = enc.encode(prompt_str)
    prompt_internal = [orig2internal[tok] for tok in prompt_tokens_orig]

    idx = torch.tensor([prompt_internal], dtype=torch.long, device=device)
    generated_str = prompt_str

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            next_internal_id = idx_next.item()
            
            orig_id = internal2orig[next_internal_id]
            try:
                token_bytes = enc.decode_bytes([orig_id])
                token_text = token_bytes.decode("utf-8", errors="replace")
            except Exception:
                token_text = enc.decode([orig_id])

            generated_str += token_text

    return generated_str


def main():
    checkpoint_path = "model_checkpoint.pt"
    corpus_path = "fineweb_corpus.txt"
    output_path = "corpus_completions.txt"
    num_samples = 100

    if not os.path.exists(checkpoint_path):
        print(f"Error: '{checkpoint_path}' not found. Please train the model first.")
        sys.exit(1)

    if not os.path.exists(corpus_path):
        print(f"Error: '{corpus_path}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    vocab_size = checkpoint["vocab_size"]
    orig2internal = {int(k): v for k, v in checkpoint["orig2internal"].items()}
    internal2orig = {int(k): v for k, v in checkpoint["internal2orig"].items()}

    device = torch.device("cpu")
    model = TinyCausalTransformer(vocab_size=vocab_size, d_model=128, n_head=4, n_layer=3, block_size=64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    enc = tiktoken.get_encoding("o200k_base")

    print(f"Reading '{corpus_path}' to select candidate sentence...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    chosen_sentence, sentence_beginning = select_valid_sentence_and_beginning(corpus_text, orig2internal, enc)

    print(f"Chosen Sentence: \"{chosen_sentence}\"")
    print(f"Sentence Beginning: \"{sentence_beginning}\"")

    print(f"Generating {num_samples} completions...")
    completions = []
    for i in range(1, num_samples + 1):
        completion = generate_completion(
            model,
            sentence_beginning,
            orig2internal,
            internal2orig,
            enc,
            device,
            max_new_tokens=30,
            block_size=64
        )
        clean_completion = completion.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        completions.append(clean_completion)

    print(f"Saving outputs to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Chosen Sentence: {chosen_sentence}\n")
        f.write(f"Sentence Beginning: {sentence_beginning}\n\n")
        f.write("Completions:\n")
        for i, comp in enumerate(completions, 1):
            f.write(f"{i:3d}. {comp}\n")

    print(f"✓ Done! Results saved to {output_path}")


if __name__ == "__main__":
    main()