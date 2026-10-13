import pytest

from flydoom.precision_training import acceptance, select_candidate


def rows(hits, count=32):
    return [{'seed': i, 'kills': int(i in hits), 'return': 100 if i in hits else -100} for i in range(count)]


def test_more_hits_still_fail_when_vision_is_unnecessary_or_old_successes_lost():
    parent = rows(set(range(13)))
    candidate = rows(set(range(3, 20)))
    blind = rows(set(range(14)))
    result = acceptance({'legacy_parent': parent, 'corrected_connected': candidate, 'corrected_disconnected': blind})
    assert not result['passed']
    assert len(result['reasons']) == 2
    assert result['paired_parent']['net_additional_hits'] == 4
    assert result['paired_parent']['lost_parent_successes'] == 3


def test_development_ties_prefer_unchanged_decoder_and_reject_unpaired_starts():
    data = {'legacy_parent': rows({0}), 'stable_frozen': rows({0, 1}), 'full': rows({0, 1})}
    assert select_candidate(data) == 'stable_frozen'
    data['full'][0]['seed'] = 100
    with pytest.raises(ValueError, match='ordered starts'):
        select_candidate(data)
