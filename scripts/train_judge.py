"""Train the distilled guess-test judge: CLIP features -> "would the LLM name it?".

1. Build examples from output/dataset: each puzzle gives two
   (unsolved page, "guessable before") and (answer key, "recognizable after").
2. Compute frozen CLIP features once (cached).
3. 5-fold cross-validation grouped by category: every test fold contains
   only categories the model never saw, so the score measures generalization
   to new subjects, not memorization.
4. Compare with zero-shot CLIP (no training), then train on everything,
   save the head, and test both on the real sample puzzles.

    python scripts/train_judge.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from dot2dot.clip_judge import default_clip_judge
from dot2dot.trained_judge import HEAD_FILE, features, make_head


def load_examples(dataset: Path, clip, cache: Path):
    rows = [json.loads(line) for line in (dataset / "puzzles.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "llm" in r]
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if len(data["y"]) == 2 * len(rows):
            return data["X"], data["y"], data["groups"], data["zero_shot"]
    X, y, groups, zero_shot = [], [], [], []
    for i, row in enumerate(rows):
        for name, is_key, label in (("page", False, row["llm"]["before_correct"]),
                                    ("key_plain", True, row["llm"]["after_correct"])):
            image = cv2.imread(str(dataset / "images" / f"{row['id']}_{name}.png"))
            X.append(features(clip, image, row["category"], is_key))
            y.append(float(label))
            groups.append(row["category"])
            zero_shot.append(clip.guess(image).label == row["category"])  # CLIP's own top-1 answer
        if (i + 1) % 100 == 0:
            print(f"  features: {i + 1}/{len(rows)} puzzles")
    X, y, groups, zero_shot = np.array(X), np.array(y), np.array(groups), np.array(zero_shot)
    np.savez(cache, X=X, y=y, groups=groups, zero_shot=zero_shot)
    return X, y, groups, zero_shot


def train_head(X: np.ndarray, y: np.ndarray, epochs: int = 300, hidden: int = 64, seed: int = 0):
    """A plain PyTorch training loop: forward pass, loss, backward pass, optimizer step."""
    torch.manual_seed(seed)
    model = make_head(X.shape[1], hidden)
    # Positives are rare (~12%), so count each one more in the loss ("class weighting").
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # weight_decay = L2 regularization: keeps weights small to avoid overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    inputs, targets = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()  # compute gradients
        optimizer.step()  # nudge weights downhill
    model.eval()
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X)))[:, 0].numpy()


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The probability cut-off that maximizes accuracy on out-of-fold predictions."""
    candidates = np.linspace(0.05, 0.95, 91)
    return float(max(candidates, key=lambda t: accuracy_score(y, p >= t)))


def real_sample_test(clip, head, threshold: float, judges_file: Path) -> None:
    """Agreement with the LLM on the real sample puzzles, zero-shot vs trained."""
    from dot2dot.pipeline import build
    from dot2dot.preprocess import load_image
    from dot2dot.raster import render

    if not judges_file.exists():
        print("(no output/eval/judges.json; run scripts/compare_judges.py for the real-sample test)")
        return
    rows = json.loads(judges_file.read_text())
    zero, trained, verdict_zero, verdict_trained = [], [], [], []
    for r in rows:
        original = load_image(Path("samples/private") / r["image"])
        result = build(original, 200, None, "auto", mystery_level=r["level"])
        truth = clip.guess(original).label
        page, key = render(result.puzzle), render(result.puzzle, solution=True, numbers=False)
        before = predict(head, features(clip, page, truth, False)[None])[0] >= threshold
        after = predict(head, features(clip, key, truth, True)[None])[0] >= threshold
        llm = r["llm"]
        trained += [before == llm["before_correct"], after == llm["after_correct"]]
        zero += [r["clip"]["before_correct"] == llm["before_correct"], r["clip"]["after_correct"] == llm["after_correct"]]
        verdict = "mystery" if after and not before else "too easy" if after else "unrecognizable"
        verdict_trained.append(verdict == llm["verdict"])
        verdict_zero.append(r["clip"]["verdict"] == llm["verdict"])
    print(f"\nReal sample puzzles ({len(rows)}), agreement with the LLM judge:")
    print(f"  zero-shot CLIP: per-answer {np.mean(zero):.0%}, verdict {np.mean(verdict_zero):.0%}")
    print(f"  trained judge:  per-answer {np.mean(trained):.0%}, verdict {np.mean(verdict_trained):.0%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("output/dataset"))
    parser.add_argument("--out", type=Path, default=HEAD_FILE)
    args = parser.parse_args(argv)

    clip = default_clip_judge()
    X, y, groups, zero_shot = load_examples(args.dataset, clip, args.dataset / "features.npz")
    print(f"{len(y)} examples ({int(y.sum())} positive), {len(set(groups))} categories, {X.shape[1]} features")

    # Cross-validation: out-of-fold predictions for every example.
    oof = np.zeros(len(y))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        model = train_head(X[train], y[train])
        oof[test] = predict(model, X[test])
    threshold = best_threshold(y, oof)
    zero_shot_p = X[:, -4]  # CLIP's zero-shot probability of the true subject

    print("\nHeld-out categories (5-fold, grouped by category), agreement with the LLM judge:")
    print(f"  always 'no':        accuracy {accuracy_score(y, np.zeros_like(y)):.1%}")
    print(f"  zero-shot CLIP:     accuracy {accuracy_score(y, zero_shot):.1%}, ROC AUC {roc_auc_score(y, zero_shot_p):.3f}")
    print(f"  trained judge:      accuracy {accuracy_score(y, oof >= threshold):.1%}, ROC AUC {roc_auc_score(y, oof):.3f}"
          f"  (threshold {threshold:.2f})")

    # Final model on all data, saved for the app.
    head = train_head(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "input_dim": X.shape[1], "hidden": 64, "threshold": threshold,
                "examples": len(y), "categories": len(set(groups))}, args.out)
    print(f"\nSaved {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

    real_sample_test(clip, head, threshold, Path("output/eval/judges.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
