"""On-policy delayed reward credit with bounded relay and decoder updates.

The relay gradient is a random-direction finite-difference estimate of the
policy objective on complete causal trajectories, not a dopamine simulation.
"""
import numpy as np

from flydoom.laya_brain_fit import active_groups


def discounted_returns(rewards, gamma=.99):
    rewards = np.asarray(rewards, float)
    if rewards.ndim != 1 or not np.isfinite(rewards).all() or not 0 <= gamma <= 1:
        raise ValueError('Invalid reward sequence or discount')
    result = np.empty_like(rewards)
    total = 0.
    for i in range(len(rewards)-1, -1, -1):
        total = rewards[i] + gamma * total
        result[i] = total
    return result


def distribution(actor, raw, weights=None):
    x = np.array([actor.features(row) for row in raw])
    logits = x @ (actor.weights if weights is None else weights).T
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return x, np.exp(logp), logp


def teacher_kl(targets, logp):
    return float(np.mean(np.sum(targets * (np.log(np.maximum(targets, 1e-30))-logp), axis=1)))


class RewardStudent:
    """Preserve the trained parent; update only between rollout batches."""
    def __init__(self, bridge, actor, demo_drive, demo_resets, demo_targets, *, train_relays=True, seed=0):
        self.bridge = bridge
        self.actor = actor
        self.gains = bridge.neutral()
        self.active = active_groups(bridge)
        self.critic = np.zeros(actor.weights.shape[1])
        self.train_relays = train_relays
        self.rng = np.random.default_rng(seed)
        self.demo_drive, self.demo_resets, self.demo_targets = demo_drive, demo_resets, demo_targets
        self.history = []
        raw = bridge.replay(demo_drive, demo_resets, self.gains)
        self.anchor_kl = teacher_kl(demo_targets, distribution(actor, raw)[2])

    def update(self, episodes):
        if not episodes or any(not len(e['actions']) for e in episodes):
            raise ValueError('Require complete nonempty training rollouts')
        raw = np.concatenate([e['raw'] for e in episodes])
        actions = np.concatenate([e['actions'] for e in episodes]).astype(int)
        behavior = np.concatenate([e['probabilities'] for e in episodes])
        x, oldp, oldlog = distribution(self.actor, raw)
        if not np.allclose(oldp, behavior, rtol=1e-6, atol=1e-8):
            raise ValueError('Stale or off-policy rollout; collect with the current frozen actor')
        # Each episode has its own finite horizon: credit cannot cross resets.
        returns = np.concatenate([discounted_returns(e['rewards'])/100 for e in episodes])
        values = x @ self.critic
        advantages = returns-values
        advantages /= max(float(np.std(advantages)), .1)
        advantages = np.clip(advantages, -5, 5)
        indices = np.arange(len(actions))
        demo_raw = self.bridge.replay(self.demo_drive, self.demo_resets, self.gains)
        dx, dp, dl = distribution(self.actor, demo_raw)
        before_kl = teacher_kl(self.demo_targets, dl)
        rehearsal = .1

        def replay(gains):
            return np.concatenate([self.bridge.replay(e['drive'], e['resets'], gains) for e in episodes])

        def score(gains, weights):
            candidate_raw = raw if np.array_equal(gains, self.gains) else replay(gains)
            _, p, logp = distribution(self.actor, candidate_raw, weights)
            ratio = np.exp(np.clip(logp[indices, actions]-oldlog[indices, actions], -20, 20))
            surrogate = float(np.mean(ratio*advantages))
            demonstration = demo_raw if np.array_equal(gains, self.gains) else self.bridge.replay(self.demo_drive, self.demo_resets, gains)
            kl = teacher_kl(self.demo_targets, distribution(self.actor, demonstration, weights)[2])
            drift = float(np.mean(np.sum(oldp*(oldlog-logp), axis=1)))
            return surrogate-rehearsal*kl, kl, max(0., drift)

        gradient = ((np.eye(8)[actions]-oldp)*advantages[:, None]).T @ x / len(x)
        gradient += rehearsal*(self.demo_targets-dp).T @ dx / len(dx)
        weight_step = .03*gradient/max(1., float(np.linalg.norm(gradient)))
        gain_gradient = np.zeros_like(self.gains)
        if self.train_relays:
            radius = .002
            for _ in range(4):
                direction = self.rng.choice([-1., 1.], self.gains.shape)*self.active
                plus = score(self.gains+radius*direction, self.actor.weights)[0]
                minus = score(self.gains-radius*direction, self.actor.weights)[0]
                gain_gradient += (plus-minus)/(2*radius)*direction/4
        gain_step = .01*gain_gradient/max(1., float(np.linalg.norm(gain_gradient)))
        old_weights = self.actor.weights.copy()
        old_gains = self.gains.copy()
        initial = score(self.gains, old_weights)[0]
        accepted = False
        final, after_kl, policy_kl = initial, before_kl, 0.
        for scale in (1., .5, .25, .125, .0625):
            proposed_gains = np.clip(old_gains+scale*gain_step, .9, 1.1)
            proposed_gains[~self.active] = 1.
            proposed_weights = old_weights+scale*weight_step
            objective, kl, drift = score(proposed_gains, proposed_weights)
            if objective > initial+1e-10 and drift <= .01 and kl <= min(before_kl+.03, self.anchor_kl+.1):
                self.gains = proposed_gains
                self.actor.weights = proposed_weights
                accepted = True
                final, after_kl, policy_kl = objective, kl, drift
                break
        # A training-only value baseline reduces variance in the NEXT batch.
        target_critic = np.linalg.solve(x.T@x + .1*np.eye(x.shape[1]), x.T@returns)
        self.critic = .5*self.critic+.5*target_critic
        self.bridge.apply(self.gains)
        result = {'batch': len(self.history)+1, 'episodes': len(episodes), 'decisions': len(x),
                  'accepted': accepted, 'mean_return': float(np.mean([sum(e['rewards']) for e in episodes])),
                  'kills': int(sum(e['kills'] for e in episodes)), 'objective_before': initial, 'objective_after': final,
                  'teacher_train_kl_before': before_kl, 'teacher_train_kl_after': after_kl,
                  'policy_kl': policy_kl, 'advantage_mean': float(advantages.mean()),
                  'value_mse_before': float(np.mean((returns-values)**2)),
                  'decoder_change_l2': float(np.linalg.norm(self.actor.weights-old_weights)),
                  'relay_change_l2': float(np.linalg.norm(self.gains-old_gains))}
        self.history.append(result)
        return result
