# dot2dot

Turn any photo into a printable connect-the-dots puzzle.

```
Photo → simplified outline → strategically placed dots → numbered sequence → printable PDF
```

Upload a picture of your kid on a bike, the family dog, or grandma's garden, pick a difficulty, and get a worksheet (plus an answer key) ready to print.

> **Status:** Phase 1 done. The CLI turns a silhouette-style image (subject on a plain background) into a numbered puzzle + answer-key PDF.
>
> New to Python from Java? See [docs/python-for-java-devs.md](docs/python-for-java-devs.md).

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

## Project layout

```
dot2dot/
├── dot2dot/
│   ├── models.py         # Dot, Puzzle data types
│   ├── preprocess.py     # load, resize, grayscale, blur
│   ├── contours.py       # subject mask + outline extraction
│   ├── simplify.py       # outline → exactly N dots, keeping corners
│   ├── order.py          # dot sequencing (clockwise, nearest outline next)
│   ├── labels.py         # number placement (outside the shape)
│   ├── render.py         # SVG output (puzzle + solution)
│   ├── pdf.py            # PDF worksheet (puzzle + answer key)
│   ├── pipeline.py       # end-to-end orchestration
│   ├── cli.py            # command-line interface
│   └── __main__.py       # `python -m dot2dot`
├── samples/              # sample images + generator script
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

Planned for later phases: `segment.py` (background removal), importance-based point selection, `app/streamlit_app.py`.

---

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Make a puzzle (writes output/cat.pdf: page 1 puzzle, page 2 answer key)
dot2dot samples/cat.png --dots 50

# Also write SVG previews, trace up to 3 separate shapes, custom title
dot2dot samples/star.png --dots 30 --outlines 3 --svg --title "Star Power"

# Run the tests
pytest
```

Works best today with a clear subject on a plain, contrasting background.

---

## Roadmap

### Phase 1 — Core pipeline (MVP)
- [x] Load image, preprocess, extract largest contours
- [x] Simplify and resample to N dots
- [x] Order dots along contours
- [x] Render numbered SVG + solution
- [x] Export PDF (puzzle + answer key)
- [x] CLI: `python -m dot2dot <image> --dots N`

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
