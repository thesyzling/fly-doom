"""Cache a frozen Laya encoder while retaining the original trainable head."""

import torch
from torch.nn.utils.rnn import pad_sequence


def cache_features(agent, items, batch_size=2):
    from laya.common import collate_items
    agent.model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            selected = items[start:start + batch_size]
            batch = collate_items([[item] for item in selected], agent.tok.pad_token_id)
            hidden = agent.model.encoder(input_ids=batch["input_ids"].to(agent.device),
                attention_mask=batch["attention_mask"].to(agent.device)).last_hidden_state
            for i, item in enumerate(selected):
                length = int(batch["attention_mask"][i].sum())
                features.append({"hidden": hidden[i, :length].detach().cpu().clone(),
                                 "markers": item["markers"], "qtype": item["qtype"]})
            if (start // batch_size + 1) % 50 == 0:
                print(f"Frozen encoder cache: {start + len(selected)}/{len(items)} observations", flush=True)
    return features


def cached_logits(agent, features):
    """Match DecisionModel.forward through its choice logits; no act-head loss."""
    hidden = pad_sequence([f["hidden"] for f in features], batch_first=True).to(agent.device)
    lengths = torch.tensor([len(f["hidden"]) for f in features], device=agent.device)
    padding = torch.arange(hidden.shape[1], device=agent.device)[None, :] >= lengths[:, None]
    qtype = torch.tensor([f["qtype"] for f in features], device=agent.device)
    hidden = hidden + agent.model.type_emb(qtype)[:, None, :]
    if agent.model.head is not None:
        for layer in agent.model.head.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
    markers = torch.tensor([f["markers"] for f in features], device=agent.device)
    indices = markers[:, :, None].expand(-1, -1, hidden.shape[-1])
    return agent.model.scorer(torch.gather(hidden, 1, indices)).squeeze(-1).float()


def cached_probabilities(agent, features, batch_size=2):
    agent.model.eval()
    with torch.no_grad():
        return torch.cat([cached_logits(agent, features[i:i + batch_size]).softmax(-1).cpu()
                          for i in range(0, len(features), batch_size)]).numpy()
