import os
import json
import numpy as np
import tiktoken

def main():
    corpus_path = "fineweb_corpus.txt"
    tokens_path = "fineweb_tokens.txt"

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Missing required corpus file: {corpus_path}")

    print(f"Loading tokenizer 'o200k_base'...")
    enc = tiktoken.get_encoding("o200k_base")

    print(f"Reading {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Generate reference token sequence from corpus
    ref_tokens = enc.encode(corpus_text)
    print(f"Corpus token count using o200k_base: {len(ref_tokens)}")

    # Check fineweb_tokens.txt if present
    valid = False
    if os.path.exists(tokens_path):
        print(f"Validating {tokens_path}...")
        try:
            with open(tokens_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if "," in content:
                existing_tokens = [int(t.strip()) for t in content.split(",") if t.strip()]
            else:
                existing_tokens = [int(t.strip()) for t in content.split() if t.strip()]
            
            if existing_tokens == ref_tokens:
                print("✓ fineweb_tokens.txt matches corpus tokens perfectly!")
                valid = True
                tokens = existing_tokens
            else:
                print("⚠ fineweb_tokens.txt does NOT match corpus tokens. Regenerating...")
        except Exception as e:
            print(f"⚠ Error reading fineweb_tokens.txt: {e}. Regenerating...")

    if not valid:
        tokens = ref_tokens
        with open(tokens_path, "w", encoding="utf-8") as f:
            f.write(", ".join(map(str, tokens)))
        print(f"✓ Saved regenerated token IDs to {tokens_path}")

    # Build compact internal vocabulary mapping
    unique_tokens = sorted(list(set(tokens)))
    vocab_size = len(unique_tokens)
    print(f"Unique tokens in corpus: {vocab_size}")

    orig2internal = {tok: i for i, tok in enumerate(unique_tokens)}
    internal2orig = {i: tok for i, tok in enumerate(unique_tokens)}

    mapped_tokens = np.array([orig2internal[tok] for tok in tokens], dtype=np.int64)

    # Save array and mapping
    np.save("mapped_tokens.npy", mapped_tokens)

    mapping_data = {
        "vocab_size": vocab_size,
        "orig2internal": {str(k): v for k, v in orig2internal.items()},
        "internal2orig": {str(k): v for k, v in internal2orig.items()}
    }

    with open("vocab_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    print("✓ Data preparation complete!")
    print("Saved: mapped_tokens.npy, vocab_mapping.json")

if __name__ == "__main__":
    main()
