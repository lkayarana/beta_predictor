import json
import numpy as np


class BetaHMM:
    def __init__(self):
        self.alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.states = ["N", "B"]

        self.aa_to_idx = {aa: i for i, aa in enumerate(self.alphabet)}
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        self.start_probs = None
        self.trans_probs = None
        self.emit_probs = None

        self.start_log = None
        self.trans_log = None
        self.emit_log = None

    def fit_supervised(self, records, pseudocount=1.0):
        n_states = len(self.states)
        n_obs = len(self.alphabet)

        start_counts = np.full(n_states, pseudocount, dtype=float)
        trans_counts = np.full((n_states, n_states), pseudocount, dtype=float)
        emit_counts = np.full((n_states, n_obs), pseudocount, dtype=float)

        for rec in records:
            seq = rec["sequence"]
            labels = rec["labels"]

            if len(seq) == 0:
                continue
            if len(seq) != len(labels):
                continue

            valid_seq = []
            valid_labels = []

            for aa, lab in zip(seq, labels):
                if aa in self.aa_to_idx and lab in self.state_to_idx:
                    valid_seq.append(aa)
                    valid_labels.append(lab)

            if len(valid_seq) == 0:
                continue

            first_state = self.state_to_idx[valid_labels[0]]
            first_obs = self.aa_to_idx[valid_seq[0]]

            start_counts[first_state] += 1
            emit_counts[first_state, first_obs] += 1

            for i in range(1, len(valid_seq)):
                prev_s = self.state_to_idx[valid_labels[i - 1]]
                curr_s = self.state_to_idx[valid_labels[i]]
                obs = self.aa_to_idx[valid_seq[i]]

                trans_counts[prev_s, curr_s] += 1
                emit_counts[curr_s, obs] += 1

        self.start_probs = start_counts / start_counts.sum()
        self.trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.emit_probs = emit_counts / emit_counts.sum(axis=1, keepdims=True)

        self.start_log = np.log(self.start_probs)
        self.trans_log = np.log(self.trans_probs)
        self.emit_log = np.log(self.emit_probs)

    def predict(self, sequence):
        seq = [aa for aa in sequence if aa in self.aa_to_idx]
        T = len(seq)
        n_states = len(self.states)

        if T == 0:
            return []

        dp = np.full((n_states, T), -np.inf)
        back = np.zeros((n_states, T), dtype=int)

        first_obs = self.aa_to_idx[seq[0]]
        for s in range(n_states):
            dp[s, 0] = self.start_log[s] + self.emit_log[s, first_obs]

        for t in range(1, T):
            obs = self.aa_to_idx[seq[t]]
            for curr_s in range(n_states):
                scores = dp[:, t - 1] + self.trans_log[:, curr_s] + self.emit_log[curr_s, obs]
                back[curr_s, t] = int(np.argmax(scores))
                dp[curr_s, t] = float(np.max(scores))

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(dp[:, -1]))

        for t in range(T - 2, -1, -1):
            path[t] = back[path[t + 1], t + 1]

        return [self.states[i] for i in path]

    def to_dict(self):
        return {
            "alphabet": self.alphabet,
            "states": self.states,
            "start_probs": self.start_probs.tolist(),
            "trans_probs": self.trans_probs.tolist(),
            "emit_probs": self.emit_probs.tolist()
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        model = cls()
        model.alphabet = d["alphabet"]
        model.states = d["states"]
        model.aa_to_idx = {aa: i for i, aa in enumerate(model.alphabet)}
        model.state_to_idx = {s: i for i, s in enumerate(model.states)}

        model.start_probs = np.array(d["start_probs"], dtype=float)
        model.trans_probs = np.array(d["trans_probs"], dtype=float)
        model.emit_probs = np.array(d["emit_probs"], dtype=float)

        model.start_log = np.log(model.start_probs)
        model.trans_log = np.log(model.trans_probs)
        model.emit_log = np.log(model.emit_probs)

        return model