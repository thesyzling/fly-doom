"""Six-action compatibility, causal history and genuine engine movement."""

import struct

import numpy as np
import pytest
import torch

from flydoom import anatomy
from flydoom.movement_core import ACTIONS, MEMORY, Memory, apply_action, expand, make_game
from flydoom.student import SpikingReadout


def test_new_history_retains_forward_and_backward_as_distinct_actions():
    memory = Memory()
    assert len(memory.encode()) == len(MEMORY) == 23
    memory.advance(4); before = memory.encode().copy(); memory.advance(5)
    assert before[MEMORY.index('past_1_MOVE_FORWARD')] == 1
    assert before[MEMORY.index('past_1_MOVE_BACKWARD')] == 0
    assert memory.encode()[MEMORY.index('past_1_MOVE_BACKWARD')] == 1
    assert memory.encode()[MEMORY.index('past_2_MOVE_FORWARD')] == 1
    assert memory.encode()[-1] == 0


def test_parent_logits_preserved_for_corresponding_old_features():
    from flydoom.action_memory import NAMES
    torch.manual_seed(23)
    parent = SpikingReadout(4 + len(NAMES) + 4, 8)
    model = expand(parent, 2)
    old = torch.randn(5, parent.input.in_features)
    names = [f'neural:{i}' for i in range(4)] + list(NAMES) + list(ACTIONS[:4])
    new_names = [f'neural:{i}' for i in range(4)] + list(MEMORY) + list(ACTIONS)
    new = torch.zeros(5, model.input.in_features)
    for i, name in enumerate(names): new[:, new_names.index(name)] = old[:, i]
    torch.testing.assert_close(parent(old), model(new)[:, :4])
    assert model(new).shape == (5, 6)


@pytest.mark.parametrize('action,sign', [(4, 1), (5, -1)])
def test_native_forward_and_backward_change_real_world_position(action, sign):
    game = make_game(74001)
    try:
        game.new_episode(); effect = apply_action(game, action)
        dx = effect['position_after'][0] - effect['position_before'][0]
        assert sign * dx > 1 and effect['game_tics'] == 4
    finally: game.close()


def test_precomputed_geometry_preserves_vertices_and_edges():
    vertices = np.asarray([[1000, 2000, 3000], [2000, 3000, 4000], [3000, 2000, 5000]], dtype='<f4')
    faces = np.asarray([[0, 1, 2]], dtype='<u4')
    points, indices = anatomy.mesh_arrays(struct.pack('<I', 3) + vertices.tobytes() + faces.tobytes())
    np.testing.assert_array_equal(points, vertices); np.testing.assert_array_equal(indices, faces)
    edges = np.asarray([[0, 1], [1, 2]], dtype='<u4')
    points, links = anatomy.skeleton_arrays(struct.pack('<II', 3, 2) + vertices.tobytes() + edges.tobytes())
    np.testing.assert_array_equal(links, edges)
    transformed = anatomy.normalize(points, {'center_um': [1, 2, 3], 'extent_um': 10})
    np.testing.assert_array_equal(transformed[0], [0, 0, 0])


def test_invalid_skeleton_index_is_rejected():
    body = struct.pack('<II', 1, 1) + np.zeros(3, '<f4').tobytes() + np.array([0, 8], '<u4').tobytes()
    with pytest.raises(ValueError): anatomy.skeleton_arrays(body)


def test_idle_observer_can_release_without_loading_weights(monkeypatch):
    from flydoom import runtime_memory
    monkeypatch.setattr(runtime_memory, 'verified_manifest', lambda *_: {'repo': 'test', 'revision': 'test', 'source_commit': 'test'})
    monkeypatch.setattr(runtime_memory, 'VisionClient', lambda: pytest.fail('No inference was requested'))
    observer = runtime_memory.IdleVision()
    assert observer.release()['vision_loaded'] is False
    observer.close()
