import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt

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
# Helper Functions
# ---------------------------------------------------------

def get_batch(data, batch_size, block_size, device):
    ix = np.random.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + block_size + 1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def stream_generation(model, prompt_str, orig2internal, internal2orig, enc, device, max_new_tokens=30, seed=42, block_size=64):
    # Set seed for exact reproducible generation
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    
    prompt_tokens_orig = enc.encode(prompt_str)
    prompt_internal = [orig2internal[tok] for tok in prompt_tokens_orig if tok in orig2internal]
    if not prompt_internal:
        prompt_internal = [0]

    idx = torch.tensor([prompt_internal], dtype=torch.long, device=device)

    header = f"=== STREAMING GENERATION (Prompt: '{prompt_str}') ==="
    print(f"\n{header}")
    sys.stdout.write(prompt_str)
    sys.stdout.flush()

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

            sys.stdout.write(token_text)
            sys.stdout.flush()
            generated_str += token_text

    print("\n" + "=" * len(header) + "\n")
    model.train()
    return generated_str

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------

def main():
    if not os.path.exists("mapped_tokens.npy") or not os.path.exists("vocab_mapping.json"):
        print("Preprocessed files not found. Please run prepare_data.py first!")
        sys.exit(1)

    print("Loading preprocessed dataset and vocabulary mapping...")
    mapped_tokens = np.load("mapped_tokens.npy")

    with open("vocab_mapping.json", "r", encoding="utf-8") as f:
        vocab_info = json.load(f)

    vocab_size = vocab_info["vocab_size"]
    orig2internal = {int(k): v for k, v in vocab_info["orig2internal"].items()}
    internal2orig = {int(k): v for k, v in vocab_info["internal2orig"].items()}

    enc = tiktoken.get_encoding("o200k_base")

    # Hyperparameters
    device = torch.device("cpu")
    batch_size = 16
    block_size = 64
    d_model = 128
    n_head = 4
    n_layer = 3
    learning_rate = 1e-3
    target_train_seconds = 60.0

    print(f"Initializing Tiny Causal Transformer (Vocab Size: {vocab_size}, Params: d_model={d_model}, n_head={n_head}, n_layer={n_layer})...")
    model = TinyCausalTransformer(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    prompt_str = "|Viewing Single Post From: Spoilers for the Week of February 11th|"

    training_logs = []
    generated_samples = []

    print("\nStarting CPU Training for ~60 seconds...\n")
    print(f"{'Step':>6} | {'Loss':>8} | {'Elapsed Time':>12} | {'Tokens/Sec':>12}")
    print("-" * 48)

    start_time = time.time()
    step = 0
    total_tokens_processed = 0

    # Initial text generation before training (Step 0)
    sample_text = stream_generation(model, prompt_str, orig2internal, internal2orig, enc, device, seed=42, block_size=block_size)
    generated_samples.append(f"--- STEP {step} (BEFORE TRAINING) ---\n{sample_text}\n")

    while True:
        step_start_t = time.time()
        
        xb, yb = get_batch(mapped_tokens, batch_size, block_size, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start_time
        batch_tokens = batch_size * block_size
        total_tokens_processed += batch_tokens
        tps = total_tokens_processed / elapsed if elapsed > 0 else 0.0

        log_line = f"{step:6d} | {loss.item():8.4f} | {elapsed:11.2f}s | {tps:12.1f}"
        print(log_line)
        training_logs.append((step, loss.item(), elapsed, tps))

        # Generate every 20 updates
        if step > 0 and step % 20 == 0:
            sample_text = stream_generation(model, prompt_str, orig2internal, internal2orig, enc, device, seed=42, block_size=block_size)
            generated_samples.append(f"--- STEP {step} ---\n{sample_text}\n")

        step += 1
        if elapsed >= target_train_seconds:
            break

    # Final text generation at the end
    print("\nFinal evaluation sample:")
    sample_text = stream_generation(model, prompt_str, orig2internal, internal2orig, enc, device, seed=42, block_size=block_size)
    generated_samples.append(f"--- FINAL STEP {step} ---\n{sample_text}\n")

    print("\nSaving outputs...")

    # Save training log
    with open("training_log.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Step':>6} | {'Loss':>8} | {'Elapsed Time':>12} | {'Tokens/Sec':>12}\n")
        f.write("-" * 48 + "\n")
        for s, l, e, t in training_logs:
            f.write(f"{s:6d} | {l:8.4f} | {e:11.2f}s | {t:12.1f}\n")

    # Save generated samples
    with open("generated_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(generated_samples))

    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'orig2internal': vocab_info['orig2internal'],
        'internal2orig': vocab_info['internal2orig'],
        'step': step
    }
    torch.save(checkpoint, "model_checkpoint.pt")

    # Save loss plot
    steps_arr = [entry[0] for entry in training_logs]
    losses_arr = [entry[1] for entry in training_logs]

    plt.figure(figsize=(10, 5))
    plt.plot(steps_arr, losses_arr, label="Training Loss", color="#1f77b4", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Tiny Causal Transformer - Loss Curve (~60s CPU Run)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.close()

    print("✓ Successfully saved:")
    print("  - training_log.txt")
    print("  - generated_samples.txt")
    print("  - model_checkpoint.pt")
    print("  - loss_plot.png")

if __name__ == "__main__":
    main()
