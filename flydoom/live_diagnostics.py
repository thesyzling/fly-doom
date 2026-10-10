"""Durable lifecycle/error evidence for the live desk, without image payloads."""

from datetime import datetime, timezone
import json
import logging
import traceback


class LiveDiagnostics:
    def __init__(self, output):
        self.path = output / 'session.log'
        self.logger = logging.Logger(str(self.path.resolve()))
        self.handler = logging.FileHandler(self.path, encoding='utf-8')
        self.logger.addHandler(self.handler)

    def event(self, event, **fields):
        self.logger.info(json.dumps({
            'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields,
        }, allow_nan=False))

    def exception(self, event, **fields):
        self.event(event, traceback=traceback.format_exc(), **fields)

    def close(self):
        self.handler.close()
        self.logger.removeHandler(self.handler)


def episode_end(game, effect, sequence, *, action_start_tick=None):
    """Report engine termination separately from the desk's recording limit."""
    if not effect['episode_finished'] and sequence < 75:
        return None
    # Some terminal transitions reset the engine's reported clock to zero.
    # Count the actual observed tics from the pre-action clock instead.
    tick = (int(game.get_episode_time()) if action_start_tick is None
            else int(action_start_tick) + effect['game_tics'])
    timeout = int(game.get_episode_timeout())
    if effect['episode_finished']:
        if game.is_player_dead():
            reason = 'player_dead'
        elif effect['kill_delta'] > 0:
            reason = 'target_eliminated'
        elif timeout and tick >= timeout:
            reason = 'time_limit'
        else:
            reason = 'engine_finished'
    else:
        reason = 'decision_limit'
    return {'reason': reason, 'game_tick': tick, 'timeout_tics': timeout,
            'decision_limit': 75, 'episode_finished': effect['episode_finished']}
