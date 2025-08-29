from ccjoker.dataset import TriplecargoDataset, collate_fn
from ccjoker.model import ModelConfig, PolicyValueNet


def test_model_forward_shapes():
    ds = TriplecargoDataset("tests/fixtures/sample.jsonl")
    num_cards = ds.max_card_id + 2
    cfg = ModelConfig(num_cards=num_cards, card_embed_dim=16, hidden_dim=64, dropout=0.0)
    model = PolicyValueNet(cfg)

    # Make a tiny batch
    batch = [ds[0], ds[1]]
    x_b, y_policy_b, y_value_b = collate_fn(batch)

    policy_logits, value_logits = model(x_b)
    assert policy_logits.shape == (2, 45)
    assert value_logits.shape == (2, 3)

    # Backprop sanity check
    loss = policy_logits.mean() + value_logits.mean()
    loss.backward()
    # Ensure some gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad