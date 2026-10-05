# dot2dot

Turn any photo into a printable connect-the-dots puzzle.

```
Photo → simplified outline → strategically placed dots → numbered sequence → printable PDF
```

Upload a picture of your kid on a bike, the family dog, or grandma's garden, pick a difficulty, and get a worksheet (plus an answer key) ready to print.

> **Status:** Phase 1 done, plus a `lineart` style that traces the lines *inside* inked drawings (comics, cartoons), not just the outer border.
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
│   ├── contours.py       # outline style: subject mask + border
│   ├── lineart.py        # lineart style: ink mask, thick/thin split
│   ├── photo.py          # photo style: background removal, XDoG sketch, faces
│   ├── skeleton.py       # trace 1px skeleton lines into paths
│   ├── simplify.py       # path → exactly N dots, keeping corners
│   ├── order.py          # stroke sequencing (nearest line next)
│   ├── labels.py         # collision-aware number placement
│   ├── quality.py        # coverage/accuracy/part score + check image
│   ├── planner.py        # AI planner (OpenAI vision → structured plan)
│   ├── guidance.py       # applies a plan: ignore, weight parts, repair
│   ├── config.py         # settings from environment / .env
│   ├── render.py         # SVG output (puzzle + solution)
│   ├── pdf.py            # PDF worksheet (puzzle + answer key)
│   ├── pipeline.py       # end-to-end orchestration
│   ├── cli.py            # command-line interface
│   ├── web/
│   │   ├── app.py        # FastAPI app (upload → puzzle JSON, PDF download)
│   │   └── static/index.html  # single-page UI
│   └── __main__.py       # `python -m dot2dot`
├── samples/              # sample images + generator script (private/ is git-ignored)
├── scripts/              # eval_samples.py: visual regression grid
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

# Web app: upload an image, see puzzle + answer key, download the PDF
dot2dot-web                     # then open http://127.0.0.1:8000

# Outline style: trace the subject's border (writes output/cat.pdf)
dot2dot samples/cat.png --dots 50

# Line-art style: follow the drawing's ink lines, including details inside
dot2dot samples/face.png --style lineart --dots 80 --svg

# Custom title, SVG previews, cap the number of separate lines
dot2dot my_cartoon.png --style lineart --dots 250 --max-lines 30 --title "Hero Time" --svg

# Real photo (first run downloads a ~170 MB background-removal model)
dot2dot family_photo.heic --style photo --dots 150

# Let it pick the style (the default), with the AI planner and an instruction
dot2dot penguin.jpg --ai --request "easy puzzle for a 6-year-old"

# Write a coverage check image: gray = traced, magenta = missed
dot2dot my_cartoon.png --style lineart --dots 250 --check

# Run the tests
pytest
```

| Style | Use for | How it works |
|-------|---------|--------------|
| `auto` (default) | Anything | Line art when the background is plain, Photo otherwise |
| `outline` | A clear subject on a plain, contrasting background | Separates subject from background, traces the border |
| `photo` | Real photos: people, pets, objects (JPG, PNG, iPhone HEIC) | Removes the background with a segmentation model (rembg, runs locally), traces the subject's outline, sketches interior features with an XDoG filter, and gives faces extra dots (OpenCV YuNet face detector) |
| `lineart` | Comics, cartoons, coloring-book art with dark ink lines | Traces the full silhouette as one unbroken loop, then adds interior ink lines (thinned to centerlines) and the edges of solid black areas |

Every run prints a quality score, e.g.

```
Quality 92/100 | outline 100% | detail 71% | accuracy 99% | crowded labels 12
```

- **outline**: share of the silhouette traced by the solution lines
- **detail**: share of the interior line work traced
- **accuracy**: share of solution line length that sits on a real line (no shortcuts across empty space)
- **crowded labels**: numbers overlapping another number or dot

In multi-line puzzles, a **ringed dot** marks the start of a new line: lift the pencil and continue from there. Detailed drawings need more dots (200–400) to stay recognizable.

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
- [x] Background removal (`--style photo`)
- [ ] Curvature-aware point selection
- [x] Collision-aware number placement (avoids dots, lines, other numbers)
- [x] Multi-contour ordering with minimal jumps
- [x] Interior lines for inked drawings (`--style lineart`)
- [x] Interior lines for photos (XDoG sketch inside the subject)
- [x] Faces get extra dots (face detection + finer sketch)
- [ ] Merge near-duplicate dots where separate lines meet
- [x] Silhouette always traced as one complete loop in lineart
- [x] Automatic quality score (`--check` for a visual coverage map)
- [x] Importance weighting (`Path.weight`; used for faces in photo style)
- [ ] Difficulty presets (Kids / Normal / Expert)
- [x] Local web app: upload → preview puzzle → preview solution → download PDF (`dot2dot-web`)

### Phase 3 — More styles
- [ ] Hybrid style (pre-drawn features + dotted silhouette)
- [ ] Flow-line style
- [ ] Stipple / micro-dot styles

### Phase 4 — AI-assisted
- [x] AI planner: subject, must-include parts, ignore regions, suggested dots (`--ai`)
- [x] Automatic style selection (`--style auto`)
- [ ] AI reviewer: a vision model scores the finished puzzle and triggers a retry with adjusted settings
- [x] Subject classification via the planner (coloring page / cartoon / photo / render)
  - Portrait: face outline, hair silhouette, key facial features
  - Animal: silhouette + major features
  - Vehicle: body, wheels, windows
- [ ] Automatic quality evaluation and retry
- [x] Natural-language requests ("easy puzzle for my 6-year-old"), via the planner

### Phase 5 — Product
- [ ] FastAPI backend
- [ ] Personalized books: upload 15–20 photos → "Tejas's Connect-the-Dots Adventure"
- [ ] AI-generated covers and captions
- [ ] Web / iOS front end

---

## AI planner (optional)

A vision model looks at the picture before the image processing runs, and
returns a plan: what the subject is, which parts must appear (eyes, beak,
feet...), what to ignore (captions, logos, page frames), and a suggested dot
count. The pipeline then gives those parts extra dots, drops ignored regions,
and repairs parts it missed. The quality score reports coverage per part, so
"feet missing" shows up as a number instead of a silent 100.

Setup (uses OpenAI):

```bash
cp .env.example .env      # then paste your key after OPENAI_API_KEY=
```

Then tick **Use AI planner** in the web app, or pass `--ai` on the command
line. Without a key everything works as before. If the AI call fails, the
puzzle is made without it. The image (downscaled) is sent to OpenAI only when
the planner is used.

## Evaluating changes

```bash
python scripts/eval_samples.py --styles auto              # grid of all samples → output/eval/grid.png
python scripts/eval_samples.py --styles auto --ai         # plus AI-planned versions and plan overlays
```

Runs every image in `samples/private/` and writes one comparison grid, the
regression suite for "does this change make puzzles better?".

## Privacy

Image processing runs locally: background removal and face detection use
models downloaded once to your machine (`~/.rembg`, `~/.cache/dot2dot`).
Photos only leave your machine if you turn on the AI planner. Put personal test photos in `samples/private/`,
which is git-ignored.

## A note on images

The intended use is **your own photos**. Generating and distributing puzzles from celebrity or other copyrighted images can raise copyright and right-of-publicity issues, so the product is designed around personal photos, not celebrity content.

---

## License

TBD
