"""Reward learning with complete-episode teacher rotation and successful replay.

Only TRAIN trajectories constrain updates. Rehearsal is a retention proxy;
independent game comparisons remain necessary before preferred-model selection.
"""
import numpy as np

from flydoom.balanced_distillation import balanced_weights
from flydoom.laya_brain_fit import targets
from flydoom.reward_student_v2 import RewardStudent, discounted_returns, distribution


def episode_blocks(rows):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Require TRAIN-only complete causal prefixes')
    starts = [i for i, r in enumerate(rows) if r['step'] == 0]
    if not starts or starts[0] != 0:
        raise ValueError('Rehearsal must start at an episode reset')
    blocks = [np.arange(a, b) for a, b in zip(starts, starts[1:]+[len(rows)])]
    for block in blocks:
        part = [rows[i] for i in block]
        if [r['step'] for r in part] != list(range(len(part))) or len({r['seed'] for r in part}) != 1:
            raise ValueError('Rehearsal contains a gap or crosses an episode')
    return blocks


def weighted_kl(target, logp, weight):
    return float(np.sum(weight*np.sum(target*(np.log(np.maximum(target, 1e-30))-logp), axis=1)))


class RetentionStudent(RewardStudent):
    def __init__(self, bridge, actor, drive, resets, rows, *, train_relays=True, seed=0):
        super().__init__(bridge, actor, drive, resets, targets(rows), train_relays=train_relays,
                         seed=seed, relay_directions=4)
        self.blocks = episode_blocks(rows)
        self.teacher_weight = balanced_weights(rows)
        self.full_anchor = weighted_kl(self.demo_targets,
            distribution(actor, bridge.replay(drive, resets, self.gains))[2], self.teacher_weight)
        self.seen = set()
        self.memory = []

    def update(self, episodes):
        if not episodes or any(not len(e['actions']) for e in episodes):
            raise ValueError('Require nonempty TRAIN trajectories')
        if any(e.get('split') != 'reward_train' for e in episodes):
            raise ValueError('Retention accepts reward_train trajectories only')
        raw = np.concatenate([e['raw'] for e in episodes])
        actions = np.concatenate([e['actions'] for e in episodes]).astype(int)
        x, oldp, oldlog = distribution(self.actor, raw)
        if not np.allclose(oldp, np.concatenate([e['probabilities'] for e in episodes]), rtol=1e-6, atol=1e-8):
            raise ValueError('Stale or off-policy rollout')
        # Keep the first four successful TRAIN trajectories; evaluation never enters memory.
        for e in episodes:
            if e['kills'] and len(self.memory) < 4 and not any(m['seed'] == e['seed'] for m in self.memory):
                self.memory.append({k: e[k] for k in ('drive', 'resets', 'probabilities', 'seed')})
        batch = len(self.history)
        selected = [(batch*4+i) % len(self.blocks) for i in range(min(4, len(self.blocks)))]
        self.seen.update(selected)
        indices = np.concatenate([self.blocks[i] for i in selected])
        demo_drive, resets = self.demo_drive[indices], self.demo_resets[indices]
        y = self.demo_targets[indices]
        tw = self.teacher_weight[indices]; tw = tw/tw.sum()
        demo_raw = self.bridge.replay(demo_drive, resets, self.gains)
        dx, dp, dl = distribution(self.actor, demo_raw)
        full_before_raw = self.bridge.replay(self.demo_drive, self.demo_resets, self.gains)
        before_kl = weighted_kl(self.demo_targets, distribution(self.actor, full_before_raw)[2], self.teacher_weight)
        returns = np.concatenate([discounted_returns(e['rewards'])/100 for e in episodes])
        values = x@self.critic
        adv = np.clip((returns-values)/max(float(np.std(returns-values)), .1), -5, 5)
        ix = np.arange(len(actions))
        memory_targets = np.concatenate([m['probabilities'] for m in self.memory]) if self.memory else None

        def memory_raw(gains):
            return np.concatenate([self.bridge.replay(m['drive'], m['resets'], gains) for m in self.memory])

        retention_raw = memory_raw(self.gains) if self.memory else None
        retention_weight = np.concatenate([np.full(len(m['probabilities']), 1/len(m['probabilities']))
                                           for m in self.memory])/len(self.memory) if self.memory else None

        def retention(gains, weights):
            if not self.memory:
                return 0.
            rr = retention_raw if np.array_equal(gains, self.gains) else memory_raw(gains)
            return weighted_kl(memory_targets, distribution(self.actor, rr, weights)[2], retention_weight)

        before_retention = retention(self.gains, self.actor.weights)

        def score(gains, weights):
            unchanged = np.array_equal(gains, self.gains)
            r = raw if unchanged else np.concatenate([self.bridge.replay(e['drive'], e['resets'], gains) for e in episodes])
            _, p, lp = distribution(self.actor, r, weights)
            ratio = np.exp(np.clip(lp[ix, actions]-oldlog[ix, actions], -20, 20))
            dr = demo_raw if unchanged else self.bridge.replay(demo_drive, resets, gains)
            teacher = weighted_kl(y, distribution(self.actor, dr, weights)[2], tw)
            retain = retention(gains, weights)
            objective = float(np.mean(ratio*adv))-.1*teacher-.5*retain
            drift = max(0., float(np.mean(np.sum(oldp*(oldlog-lp), axis=1))))
            return objective, drift, retain

        gradient = ((np.eye(8)[actions]-oldp)*adv[:, None]).T@x/len(x)
        gradient += .1*((y-dp)*tw[:, None]).T@dx
        if self.memory:
            mx, mp, _ = distribution(self.actor, retention_raw)
            gradient += .5*((memory_targets-mp)*retention_weight[:, None]).T@mx
        dw = .03*gradient/max(1., float(np.linalg.norm(gradient)))
        gg = np.zeros_like(self.gains)
        if self.train_relays:
            for _ in range(self.relay_directions):
                d = self.rng.choice([-1., 1.], self.gains.shape)*self.active
                gg += (score(self.gains+.002*d, self.actor.weights)[0]-score(self.gains-.002*d, self.actor.weights)[0])/.004*d/self.relay_directions
        dg = .01*gg/max(1., float(np.linalg.norm(gg)))
        old_gains, old_weights = self.gains.copy(), self.actor.weights.copy()
        initial = score(old_gains, old_weights)[0]
        accepted = False; final = initial; after_kl = before_kl; after_retention = before_retention; drift = 0.
        rejection = []
        for scale in (1., .5, .25, .125, .0625):
            gains = np.clip(old_gains+scale*dg, .9, 1.1); gains[~self.active] = 1.
            weights = old_weights+scale*dw
            objective, policy_kl, retain = score(gains, weights)
            if objective <= initial+1e-10 or policy_kl > .01 or retain > min(.02, before_retention+.005):
                rejection.append('surrogate, on-policy drift or successful-trajectory retention bound')
                continue
            full_raw = full_before_raw if np.array_equal(gains, old_gains) else self.bridge.replay(self.demo_drive, self.demo_resets, gains)
            full_kl = weighted_kl(self.demo_targets, distribution(self.actor, full_raw, weights)[2], self.teacher_weight)
            if full_kl > min(before_kl+.03, self.full_anchor+.1):
                rejection.append('full TRAIN teacher guard')
                continue
            self.gains, self.actor.weights = gains, weights
            accepted = True; final = objective; after_kl = full_kl; after_retention = retain; drift = policy_kl
            break
        target_critic = np.linalg.solve(x.T@x+.1*np.eye(x.shape[1]), x.T@returns)
        self.critic = .5*self.critic+.5*target_critic
        self.bridge.apply(self.gains)
        result = {'batch': batch+1, 'episodes': len(episodes), 'decisions': len(x), 'accepted': accepted,
            'mean_return': float(np.mean([sum(e['rewards']) for e in episodes])), 'kills': int(sum(e['kills'] for e in episodes)),
            'objective_before': initial, 'objective_after': final, 'teacher_train_kl_before': before_kl,
            'teacher_train_kl_after': after_kl, 'policy_kl': drift, 'advantage_mean': float(adv.mean()),
            'value_mse_before': float(np.mean((returns-values)**2)),
            'decoder_change_l2': float(np.linalg.norm(self.actor.weights-old_weights)),
            'relay_change_l2': float(np.linalg.norm(self.gains-old_gains)),
            'relay_gradient_directions': self.relay_directions if self.train_relays else 0,
            'rehearsal_episode_indices': selected, 'rehearsal_episodes_seen': len(self.seen),
            'rehearsal_total_episodes': len(self.blocks), 'full_teacher_guard_frames': len(self.demo_targets),
            'retention_seeds': [m['seed'] for m in self.memory], 'retention_kl_before': before_retention,
            'retention_kl_after': after_retention, 'rejected_line_search_steps': rejection}
        self.history.append(result)
        return result
