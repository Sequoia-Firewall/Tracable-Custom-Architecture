"""
Character-level tokenizer -- the simplest REAL tokenizer, chosen over BPE
specifically to avoid building merge-training infrastructure from scratch
for what's meant to be a baseline-test harness, not a tokenizer research
project. Char-level language modeling on tiny-shakespeare is itself a
standard, recognized benchmark mode (e.g. Karpathy's char-rnn/nanoGPT),
not a simplification that invalidates the comparison.
"""


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
