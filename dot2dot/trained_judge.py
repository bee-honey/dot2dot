"""A guess-test judge trained to imitate the OpenAI judge (knowledge distillation).

The LLM judge is accurate but slow and costs money; zero-shot CLIP is free and
instant but agrees with it less often. Here we train a small "head" on top of
frozen CLIP features to predict the LLM's verdict:

    P(the LLM judge would name `subject` when shown `image`)

Features for one (image, subject) pair:
- CLIP image vector * CLIP text vector of the subject (elementwise, 512 numbers):
  which "directions" of agreement matter for recognizability
- cosine similarity, zero-shot probability and rank of the subject among
  the 354 vocabulary labels
- whether the image is an unsolved page or an answer key

The head is a tiny neural network (one hidden layer), trained with PyTorch in
`scripts/train_judge.py`. CLIP itself is not changed ("frozen"): with a
thousand examples, training only a small head avoids overfitting. That's
transfer learning, the standard way to adapt a big pre-trained model on a
small dataset.
"""

from pathlib import Path as FilePath

import numpy as np

from dot2dot.judge import Guess, GuessTest

HEAD_FILE = FilePath(__file__).parent / "data" / "judge_head.pt"
EMBED_DIM = 512


def features(clip, image: np.ndarray, subject: str, is_answer_key: bool) -> np.ndarray:
    """The feature vector for one (image, subject) pair."""
    clip.add_label(subject)
    probs, image_vector = clip.probabilities(image, return_vector=True)
    index = clip.labels.index(subject)
    text_vector = clip.text_vectors[index].float().cpu().numpy()
    cosine = float(image_vector @ text_vector)
    rank = int((probs > probs[index]).sum()) + 1
    extras = [cosine * 10, float(probs[index]), 1.0 / rank, float(rank <= 3), float(is_answer_key)]
    return np.concatenate([image_vector * text_vector * 10, extras]).astype(np.float32)


def make_head(input_dim: int, hidden: int = 64):
    """A small network: features -> 64 hidden units -> 1 probability (as a logit)."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))


class TrainedJudge:
    """CLIP + the trained head. Implements `judge_puzzle` for the guess test."""

    answer_key_numbers = False  # trained on answer keys without printed numbers

    def __init__(self, clip=None, head_file: FilePath = HEAD_FILE):
        import torch

        from dot2dot.clip_judge import default_clip_judge

        self.clip = clip or default_clip_judge()
        saved = torch.load(head_file, map_location="cpu", weights_only=False)
        self.head = make_head(saved["input_dim"], saved["hidden"])
        self.head.load_state_dict(saved["state_dict"])
        self.head.eval()
        self.threshold = saved["threshold"]
        self._torch = torch

    def probability(self, image: np.ndarray, subject: str, is_answer_key: bool) -> float:
        torch = self._torch
        x = torch.from_numpy(features(self.clip, image, subject, is_answer_key)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.head(x))[0, 0])

    # The Judge protocol (used for the truth guess and by tools that call it directly).
    def guess(self, image: np.ndarray) -> Guess:
        return self.clip.guess(image)

    def same_thing(self, truth: str, guesses: list[str]) -> list[bool]:
        return self.clip.same_thing(truth, guesses)

    def judge_puzzle(self, original, page, key, truth: Guess | None = None) -> GuessTest:
        """The guess test, decided by the trained head instead of exact-label matching."""
        truth = truth or self.clip.guess(original)
        p_before = self.probability(page, truth.label, is_answer_key=False)
        p_after = self.probability(key, truth.label, is_answer_key=True)

        def as_guess(image, p: float) -> Guess:
            # Show CLIP's own top label, but report the head's confidence in the truth.
            top = self.clip.guess(image)
            label = truth.label if p >= self.threshold else (top.label if top.label != truth.label else "unsure")
            return Guess(label, p if p >= self.threshold else 1 - p, top.alternatives)

        return GuessTest(
            truth.label,
            as_guess(page, p_before),
            as_guess(key, p_after),
            before_correct=p_before >= self.threshold,
            after_correct=p_after >= self.threshold,
        )
