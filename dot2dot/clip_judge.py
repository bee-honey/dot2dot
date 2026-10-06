"""A free, local guess-test judge built on CLIP.

CLIP (Contrastive Language-Image Pre-training) was trained on 400M+
image/caption pairs to put matching images and texts close together in the
same vector space. That makes it a zero-shot classifier: embed the image,
embed "a drawing of a penguin", "a drawing of an owl", ... and pick the
caption whose vector points the same way (highest cosine similarity).

Unlike the OpenAI judge it can't name anything freely, so it chooses from a
vocabulary: the 345 everyday categories of Google's Quick, Draw! dataset plus
a few popular characters. It implements the same `Judge` protocol, so
`run_guess_test` and `tune_mystery` work with either judge.

Runs locally (Apple GPU via MPS when available); the model (~600 MB)
downloads on first use.
"""

from functools import lru_cache
from pathlib import Path as FilePath

import cv2
import numpy as np

from dot2dot.judge import Guess

VOCABULARY_FILE = FilePath(__file__).parent / "data" / "quickdraw_categories.txt"
EXTRA_LABELS = ["Spider-Man", "Hulk", "superhero", "astronaut", "cartoon character", "princess", "robot", "dinosaur", "unicorn"]
MODEL_NAME, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
# Several phrasings per label, averaged: "prompt ensembling" makes zero-shot
# CLIP noticeably more reliable than a single template.
TEMPLATES = ["a drawing of a {}.", "a line drawing of a {}.", "a sketch of a {}.", "a picture of a {}.", "a cartoon {}."]
# CLIP's similarity scores are scaled by this before softmax (its learned "temperature").
LOGIT_SCALE = 100.0


def vocabulary() -> list[str]:
    labels = [line.strip() for line in VOCABULARY_FILE.read_text().splitlines() if line.strip()]
    return labels + [label for label in EXTRA_LABELS if label not in labels]


class CLIPJudge:
    # Printed numbers confuse CLIP on answer keys (agreement with the LLM judge
    # rose from 46% to 88% without them), so it reads keys without numbers.
    answer_key_numbers = False

    def __init__(self, labels: list[str] | None = None):
        import open_clip
        import torch

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.labels: list[str] = []
        self.text_vectors = None
        self.set_labels(labels or vocabulary())

    def set_labels(self, labels: list[str]) -> None:
        """Embed every label once (averaging the templates); reused for all images."""
        torch = self._torch
        vectors = []
        with torch.no_grad():
            for label in labels:
                tokens = self.tokenizer([t.format(label) for t in TEMPLATES]).to(self.device)
                v = self.model.encode_text(tokens)
                v = v / v.norm(dim=-1, keepdim=True)
                v = v.mean(dim=0)
                vectors.append(v / v.norm())
        self.labels = list(labels)
        self.text_vectors = torch.stack(vectors)

    def add_label(self, label: str) -> None:
        """Make sure a label (e.g. the true subject from a plan) is in the vocabulary."""
        if label not in self.labels:
            self.set_labels(self.labels + [label])

    def probabilities(self, image: np.ndarray, return_vector: bool = False):
        """Probability of each vocabulary label for a BGR image (softmax over labels).

        With `return_vector`, also returns the image's normalized CLIP vector.
        """
        from PIL import Image

        torch = self._torch
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pixels = self.preprocess(pil).unsqueeze(0).to(self.device)
            v = self.model.encode_image(pixels)
            v = v / v.norm(dim=-1, keepdim=True)
            logits = LOGIT_SCALE * v @ self.text_vectors.T
            probs = logits.softmax(dim=-1)[0].float().cpu().numpy()
            return (probs, v[0].float().cpu().numpy()) if return_vector else probs

    def guess(self, image: np.ndarray) -> Guess:
        probs = self.probabilities(image)
        order = np.argsort(probs)[::-1]
        return Guess(self.labels[order[0]], float(probs[order[0]]), [self.labels[i] for i in order[1:4]])

    def same_thing(self, truth: str, guesses: list[str]) -> list[bool]:
        # Guesses come from the same vocabulary, so an exact match is enough.
        return [g.lower() == truth.lower() for g in guesses]

    def truth_rank(self, image: np.ndarray, truth: str) -> tuple[int, float]:
        """Where the true label ranks for this image (1 = top guess), and its probability."""
        self.add_label(truth)
        probs = self.probabilities(image)
        index = self.labels.index(truth)
        return int((probs > probs[index]).sum()) + 1, float(probs[index])


@lru_cache(maxsize=1)
def default_clip_judge() -> CLIPJudge:
    """Load CLIP once per process (a few seconds) and reuse it."""
    return CLIPJudge()
