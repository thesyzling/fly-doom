from types import SimpleNamespace

import numpy as np
import pytest


def test_balanced_updates_contain_every_class_before_clipping():
    from flydoom.laya_teacher import balanced_batches
    labels = np.repeat(np.arange(4), [90, 6, 3, 1])
    batches = list(balanced_batches(labels, np.random.default_rng(4), 16))
    assert len(batches) == 7
    for batch in batches:
        assert np.bincount(labels[batch], minlength=4).tolist() == [4, 4, 4, 4]
        assert np.all(batch < len(labels))
    with pytest.raises(ValueError, match="every action"):
        list(balanced_batches(np.zeros(10, dtype=int), np.random.default_rng(4)))


def test_probe_uses_unique_balanced_training_examples():
    pytest.importorskip("torch")
    from flydoom.learning_diagnosis import unique_balanced_indices
    states = [{"example": i} for i in range(20)]
    labels = np.arange(20) % 4
    chosen = unique_balanced_indices(states, labels)
    assert np.bincount(labels[chosen], minlength=4).tolist() == [4, 4, 4, 4]
    assert len(set(chosen)) == 16


def test_cached_logits_and_accumulated_gradients_match_original_laya():
    torch = pytest.importorskip("torch")
    pytest.importorskip("laya")
    from laya.common import DecisionModel
    from flydoom.laya_teacher import batch_logits
    from flydoom.laya_features import cache_features, cached_logits
    torch.manual_seed(3)
    torch.set_num_threads(1)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = torch.nn.Embedding(20, 8)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    model = DecisionModel(Encoder(), head_layers=1, dropout=0)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.eval()
    agent = SimpleNamespace(model=model, device="cpu", tok=SimpleNamespace(pad_token_id=0))
    items = [{"ids": list(range(1, 8 + i)), "markers": [1, 2, 3, 4], "qtype": 0} for i in range(4)]
    features = cache_features(agent, items)
    assert all(not f["hidden"].requires_grad for f in features)
    live = batch_logits(agent, items)
    cached = cached_logits(agent, features)
    torch.testing.assert_close(live, cached, rtol=1e-5, atol=1e-6)
    labels = torch.arange(4)
    torch.nn.functional.cross_entropy(live, labels).backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for start in (0, 2):
        loss = torch.nn.functional.cross_entropy(cached_logits(agent, features[start:start + 2]),
            labels[start:start + 2], reduction="sum") / 4
        loss.backward()
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-6)
    assert all(p.grad is None for p in model.encoder.parameters())
