"""Tiny randomly-initialised Qwen2 models for CPU unit tests (no downloads)."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def tiny_qwen(seed=0, layers=2, hidden=128, inter=256, vocab=512, tie=True, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=tie, attn_implementation="sdpa", use_sliding_window=False)
    m = Qwen2ForCausalLM(cfg).to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m
