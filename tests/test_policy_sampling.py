import numpy as np
import pytest

from flydoom.adaptive_policy import AdaptiveActor
from flydoom.policy_sampling import choose_action, sampling_enabled


def test_sampling_is_reproducible_uses_learned_distribution_and_keeps_weights_frozen():
    actor = AdaptiveActor(np.arange(1), np.array([1], dtype=np.uint64), np.zeros(1), np.ones(1))
    actor.weights[:, -1] = np.log([.01, .04, .1, .8, .01, .01, .01, .02])
    before = actor.weights.copy()
    raw = np.zeros(1)
    def sequence(seed):
        rng = np.random.default_rng(seed)
        return [choose_action(actor, raw, rng, sample=True)[0] for _ in range(200)]
    a = sequence(123)
    assert a == sequence(123) and a != sequence(124)
    assert 130 < a.count(3) < 185
    assert len(set(a)) > 1
    assert choose_action(actor, raw, None, sample=False)[0] == 3
    np.testing.assert_array_equal(actor.weights, before)
    assert not sampling_enabled({}) and sampling_enabled({'decision_sampling': 'trained_softmax_v1'})
    with pytest.raises(ValueError, match='Unknown'):
        sampling_enabled({'decision_sampling': 'invalid'})
    with pytest.raises(ValueError, match='generator'):
        choose_action(actor, raw, None, sample=True)


def test_deployment_metadata_can_inherit_verified_teaching_history(tmp_path):
    import json
    from flydoom.data import digest
    from flydoom.integrated_desk import distillation_evidence
    parent = tmp_path/'parent'
    child = tmp_path/'deployment'
    parent.mkdir()
    child.mkdir()
    history = {'history': [], 'selected_epoch': 10}
    (parent/'distillation.json').write_text(json.dumps(history))
    (parent/'report.json').write_text(json.dumps({'output_sha256': {'distillation.json': digest(parent/'distillation.json','sha256')}}))
    report = {'parent': str(parent), 'parent_report_sha256': digest(parent/'report.json','sha256')}
    assert distillation_evidence(child, report) == (history, str(parent))
    (parent/'distillation.json').write_text('{}')
    with pytest.raises(ValueError, match='history changed'):
        distillation_evidence(child, report)
