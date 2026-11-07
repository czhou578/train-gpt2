"""Smoke tests for the GPT model in model.py.

The tests build a tiny model on the CPU. They check the output shapes, the loss,
the gradient flow, the optimizer groups, text generation, and the block-size crop.
"""

import math

import pytest
import torch

from model import GPT, GPTConfig


@pytest.fixture(scope="module")
def tiny_config():
    return GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)


@pytest.fixture(scope="module")
def model(tiny_config):
    torch.manual_seed(0)
    return GPT(tiny_config)


def test_forward_with_targets_returns_logits_and_finite_loss(model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (3, tiny_config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, tiny_config.block_size, tiny_config.vocab_size)
    assert loss is not None
    assert math.isfinite(loss.item())
    # A fresh model predicts close to a uniform distribution over the vocabulary.
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 1.0


def test_forward_without_targets_keeps_only_the_last_position(model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (2, 5))
    logits, loss = model(x)
    assert logits.shape == (2, 1, tiny_config.vocab_size)
    assert loss is None


def test_backward_produces_a_gradient_for_every_parameter(model, tiny_config):
    model.zero_grad(set_to_none=True)
    x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size))
    _, loss = model(x, x)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_configure_optimizers_splits_the_decay_groups(model):
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_generate_appends_the_requested_number_of_tokens(model, tiny_config):
    model.eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=10)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)
    assert int(out.max()) < tiny_config.vocab_size


def test_crop_block_size_shrinks_the_position_embedding(tiny_config):
    small = GPT(tiny_config)
    small.crop_block_size(8)
    assert small.config.block_size == 8
    assert small.transformer.wpe.weight.shape[0] == 8
    x = torch.randint(0, tiny_config.vocab_size, (1, 8))
    logits, _ = small(x)
    assert logits.shape == (1, 1, tiny_config.vocab_size)


def test_estimate_mfu_is_positive(model):
    assert model.estimate_mfu(fwdbwd_per_iter=1, dt=1.0) > 0
