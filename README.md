# dot2dot

Turn any photo into a printable connect-the-dots puzzle.

```
Photo → simplified outline → strategically placed dots → numbered sequence → printable PDF
```

Upload a picture of your kid on a bike, the family dog, or grandma's garden, pick a difficulty, and get a worksheet (plus an answer key) ready to print.

> **Status:** early prototype. Nothing works yet — this README is the plan.

---

## Why

Generating dots is easy. Generating a *good* puzzle is not. Edge detection happily finds every shadow, wrinkle, and background object; a good connect-the-dots keeps only the shapes that make the subject recognizable, and orders the dots so that drawing them is satisfying instead of a tangled mess.

The core is a **deterministic computer-vision pipeline** — no LLM required. AI (segmentation, subject classification, and eventually an orchestrator) gets layered on where it actually improves the output.

---

## Puzzle styles

Six target styles, roughly in the order we plan to build them:

| # | Style | Description | Typical dots |
|---|-------|-------------|--------------|
| 1 | **Classic contour** | Numbered dots along the major outlines (face, hair, glasses, shoulders). One or a few continuous paths. | 40–200 |
| 2 | **Contour + answer key** | Same as classic, rendered twice: the puzzle, and a solution sheet with the connecting lines drawn in. | 40–200 |
| 3 | **Hybrid / partial drawing** | Fine details (eyes, nose, mouth) are pre-drawn as line art; dots only trace the silhouette. Great for young kids. Multiple sequences can use numbers *and* letters. | 25–60 |
| 4 | **Flow lines** | Many short numbered strokes that follow the flow of hair, fur, or fabric. Looks like an engraving when complete. | 300–1000+ |
| 5 | **Stipple** | Dots are distributed by tone rather than edges; the image emerges from dot density. | 500–1500 |
| 6 | **Micro-dot portrait** | Extreme-density stipple with tiny numbers. Expert / poster mode. | 1500+ |

Difficulty presets:

| Preset | Dots | Notes |
|--------|------|-------|
| Kids | 25–40 | Large numbers, hybrid style, simple silhouette |
| Normal | 50–100 | Classic contour |
| Expert | 150–300+ | Classic contour with interior features, or flow lines |

---

## Pipeline

```
           USER PHOTO
               │
               ▼
        Preprocess (resize, denoise, contrast)
               │
               ▼
        Segment subject / remove background
               │
               ▼
        Detect important edges & contours
               │
               ▼
        Simplify contours  (e.g. 4000 px → ~300 pts)
               │
               ▼
        Score & select N points (by difficulty)
               │
               ▼
        Order points  1 → 2 → 3 → … → N
               │
               ▼
        Place numbers (avoid overlaps)
               │
               ▼
        Render SVG  →  puzzle + solution
               │
               ▼
        PDF worksheet
```

### Stages

1. **Preprocess** — resize to a working resolution, grayscale, bilateral filter / CLAHE to suppress noise and lighting.
2. **Segmentation** — isolate the subject so the background contributes no edges (rembg / U²-Net to start; SAM-style models later).
3. **Edge & contour detection** — Canny + `findContours`, with filtering by length and position (silhouette vs. interior features).
4. **Simplification** — Douglas–Peucker (`approxPolyDP`) plus curvature-aware resampling so corners keep dots and straight runs don't waste them.
5. **Point selection** — rank candidate points by importance (curvature, contour length, feature priority) and keep the top N for the chosen difficulty.
6. **Ordering** — follow each contour naturally; join multiple contours with short jumps (nearest-neighbour / TSP-style) so the sequence is easy to follow.
7. **Label placement** — put each number near its dot without colliding with other dots, numbers, or lines.
8. **Rendering** — SVG for the puzzle and solution; ReportLab for the final PDF.

---

## Tech stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ |
| Image processing | OpenCV, NumPy, scikit-image |
| Background removal | rembg (U²-Net) |
| Geometry | Shapely, SciPy |
| Rendering | svgwrite / Pillow |
| PDF | ReportLab |
| Prototype UI | Streamlit |
| API (later) | FastAPI |

---

## Planned project layout

```
dot2dot/
├── dot2dot/
│   ├── preprocess.py     # resize, denoise, normalize
│   ├── segment.py        # background removal / subject mask
│   ├── contours.py       # edge detection, contour extraction
│   ├── simplify.py       # contour simplification & resampling
│   ├── select.py         # importance scoring, pick N dots
│   ├── order.py          # dot sequencing
│   ├── labels.py         # number placement
│   ├── render.py         # SVG output (puzzle + solution)
│   ├── pdf.py            # PDF worksheet generation
│   └── pipeline.py       # end-to-end orchestration
├── app/
│   └── streamlit_app.py  # prototype UI
├── samples/              # test images
├── tests/
├── pyproject.toml
└── README.md
```

---

## Getting started

> Not runnable yet — these are the intended commands.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# CLI
python -m dot2dot samples/dog.jpg --dots 60 --out dog.pdf

# Prototype UI
streamlit run app/streamlit_app.py
```

---

## Roadmap

### Phase 1 — Core pipeline (MVP)
- [ ] Load image, preprocess, extract largest contours
- [ ] Simplify and resample to N dots
- [ ] Order dots along contours
- [ ] Render numbered SVG + solution
- [ ] Export single-page PDF
- [ ] CLI: `python -m dot2dot <image> --dots N`

### Phase 2 — Quality
- [ ] Background removal
- [ ] Curvature-aware point selection
- [ ] Collision-free number placement
- [ ] Multi-contour ordering with minimal jumps
- [ ] Difficulty presets (Kids / Normal / Expert)
- [ ] Streamlit prototype: upload → preview puzzle → preview solution → download PDF

### Phase 3 — More styles
- [ ] Hybrid style (pre-drawn features + dotted silhouette)
- [ ] Flow-line style
- [ ] Stipple / micro-dot styles

### Phase 4 — AI-assisted
- [ ] Subject classification (person / animal / vehicle / object) → per-category strategy
  - Portrait: face outline, hair silhouette, key facial features
  - Animal: silhouette + major features
  - Vehicle: body, wheels, windows
- [ ] Automatic quality evaluation and retry
- [ ] Natural-language requests ("easy puzzle for my 6-year-old, drop the background")

### Phase 5 — Product
- [ ] FastAPI backend
- [ ] Personalized books: upload 15–20 photos → "Tejas's Connect-the-Dots Adventure"
- [ ] AI-generated covers and captions
- [ ] Web / iOS front end

---

## A note on images

The intended use is **your own photos**. Generating and distributing puzzles from celebrity or other copyrighted images can raise copyright and right-of-publicity issues, so the product is designed around personal photos, not celebrity content.

---

## License

TBD
