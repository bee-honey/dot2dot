# dot2dot

Turn any photo into a printable connect-the-dots puzzle.

```
Photo → simplified outline → strategically placed dots → numbered sequence → printable PDF
```

Upload a picture of your kid on a bike, the family dog, or grandma's garden, pick a difficulty, and get a worksheet (plus an answer key) ready to print.

> **Status:** Working CLI and local FastAPI web app with outline, line-art, photo, and sketch extraction; curvature-aware dot placement; SVG/PDF export; and quality reports. Optional AI planning, local/OpenAI judges, and mystery auto-tuning are implemented.
>
> New to Python from Java? See [docs/python-for-java-devs.md](docs/python-for-java-devs.md).

---

## Why

Generating dots is easy. Generating a *good* puzzle is not. Edge detection happily finds every shadow, wrinkle, and background object; a good connect-the-dots keeps only the shapes that make the subject recognizable, and orders the dots so that drawing them is satisfying instead of a tangled mess.

The core is a **deterministic computer-vision pipeline** — no LLM required. Local models handle background removal and face detection; optional AI planning and recognizability judges guide extraction and mystery settings.

---

## Puzzle styles

The implemented extraction styles are `outline`, `lineart`, `photo`, and `sketch`, with `auto` selecting line art or photo processing. Every generated PDF includes a puzzle and answer key. The broader output-style goals are:

| # | Style | Description | Typical dots |
|---|-------|-------------|--------------|
| 1 | **Classic contour** | Numbered dots along the major outlines (face, hair, glasses, shoulders). One or a few continuous paths. | 40–200 |
| 2 | **Contour + answer key** | Same as classic, rendered twice: the puzzle, and a solution sheet with the connecting lines drawn in. | 40–200 |
| 3 | **Hybrid / partial drawing** | Fine details (eyes, nose, mouth) are pre-drawn as line art; dots only trace the silhouette. Great for young kids. Multiple sequences can use numbers *and* letters. | 25–60 |
| 4 | **Flow lines** | Many short numbered strokes that follow the flow of hair, fur, or fabric. Looks like an engraving when complete. | 300–1000+ |
| 5 | **Stipple** | Dots are distributed by tone rather than edges; the image emerges from dot density. | 500–1500 |
| 6 | **Micro-dot portrait** | Extreme-density stipple with tiny numbers. Expert / poster mode. | 1500+ |

Planned difficulty presets (currently, choose the dot count directly):

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

1. **Preprocess** — decode images (including HEIC and transparency), resize, and apply style-specific grayscale and noise filtering.
2. **Segmentation** — isolate the subject using plain-background separation or local rembg / U²-Net background removal.
3. **Path extraction** — trace borders, skeletonized ink, XDoG photo details, or pencil-sketch and color edges, depending on style.
4. **Path selection and dot allocation** — preserve essential silhouettes and divide the dot budget by path length, curvature, and importance, with readability limits.
5. **Simplification** — preserve sharp corners and open endpoints, then sample along paths with extra density at curves. The final dot count may be lower than requested to keep the puzzle readable.
6. **Ordering** — follow each path naturally and choose the nearest next stroke to reduce pencil travel. Mystery mode adds filler strokes and saves revealing features for later.
7. **Label placement** — put each number near its dot without colliding with other dots, numbers, or lines.
8. **Rendering** — SVG for the puzzle and solution; ReportLab for the final PDF.

---

## Tech stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ |
| Image processing | OpenCV, NumPy, scikit-image |
| Background removal | rembg (U²-Net) |
| Geometry / sampling | NumPy, SciPy (via scikit-image) |
| Image decoding | Pillow, pillow-heif |
| Rendering | Custom SVG and OpenCV raster rendering |
| PDF | ReportLab |
| Web UI | Static HTML, CSS, and JavaScript |
| API / server | FastAPI, Uvicorn |
| Optional AI planner / judge | OpenAI SDK |
| Optional local judges | PyTorch, OpenCLIP, bundled trained head |

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
│   ├── sketch.py         # pencil sketch + color edges, weighted by line width
│   ├── skeleton.py       # trace 1px skeleton lines into paths
│   ├── simplify.py       # path selection, curvature-aware dots, readability limits
│   ├── order.py          # stroke sequencing (nearest line next)
│   ├── labels.py         # collision-aware number placement
│   ├── quality.py        # coverage/accuracy/part score + check image
│   ├── planner.py        # AI planner (OpenAI vision → structured plan)
│   ├── judge.py          # guess test + mystery auto-tune (OpenAI judge)
│   ├── clip_judge.py     # local CLIP judge (free, offline)
│   ├── trained_judge.py  # CLIP + trained head (distilled from the OpenAI judge)
│   ├── quickdraw.py      # Quick, Draw! doodles → paths (dataset building)
│   ├── data/             # Quick, Draw! category names, trained judge head
│   ├── mystery.py        # filler strokes, reveal order, dot spread
│   ├── raster.py         # render puzzles as images (for the judge, grids)
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
├── scripts/              # eval_samples.py, compare_judges.py, make_dataset.py, train_judge.py
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

The web app runs locally at `127.0.0.1:8000`. Generated PDFs are kept in an in-memory cache of 20 puzzles; restarting the server clears them. There is no database or persistent puzzle history.

---

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Web app: upload an image, see puzzle + answer key, download the PDF
dot2dot-web                     # then open http://127.0.0.1:8000

# Outline style: trace the subject's border (writes output/cat.pdf)
dot2dot samples/cat.png --style outline --dots 50

# Line-art style: follow the drawing's ink lines, including details inside
dot2dot samples/face.png --style lineart --dots 80 --svg

# Custom title, SVG previews, cap the number of separate lines
dot2dot my_cartoon.png --style lineart --dots 250 --max-lines 30 --title "Hero Time" --svg

# Real photo (first run downloads a ~170 MB background-removal model)
dot2dot family_photo.heic --style photo --dots 150

# Pencil-sketch extraction with color edges and line-width weighting
dot2dot samples/face.png --style sketch --dots 150 --svg

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
| `sketch` | Drawings or photos that benefit from pencil-sketch extraction | Combines pencil-sketch lines with color edges, removes background and frames, and weights bold lines and faces when allocating dots |

Every run prints a quality score, e.g.

```
Quality 92/100 | outline 100% | detail 71% | accuracy 99% | crowded labels 12
```

- **outline**: share of the silhouette traced by the solution lines
- **detail**: share of the interior line work traced
- **accuracy**: share of solution line length that sits on a real line (no shortcuts across empty space)
- **crowded labels**: numbers overlapping another number or dot

For detailed drawings, enable **Keep fine details drawn** in the web app or
pass `--fine-details` on the CLI. Short interior strokes (eyes, fingers,
armor seams) and unselected interior paths are printed on both sheets while
larger paths remain numbered. This helps small features survive limited dot
budgets, but makes the subject easier to guess before solving. It is off by
default and can be combined with mystery mode. Quality coverage includes
these printed lines.

After generation, open **Inspect and correct traced lines** in the web app:

- Compare the original image with extracted lines; hide the original to inspect extraction alone.
- Click lines to include/exclude them or mark important details for preservation.
- Trace missing details by clicking points, then finish an open line or closed shape.
- Apply corrections to regenerate the numbered puzzle, answer key, and downloadable PDF.

Important paths reserve dots before the remaining budget is shared. A very tight budget can be exceeded to retain their minimum geometry; explicit line limits can still exclude paths. Quality warnings identify low interior or planned-part coverage. Coverage is measured against extracted lines, so it does not prove that all source features were found. Use the source overlay to check missing eyes and other identity details. Corrections are kept in the current browser result, not saved as projects.

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
- [x] Curvature-aware point selection
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
- [x] Trace correction: include/exclude paths, protect important lines, manually trace missing details

### Phase 3 — More styles
- [x] Sketch extraction (pencil sketch + color edges, line-width weighting)
- [x] Hybrid option: print fine interior details while larger paths remain numbered
- [ ] Flow-line style
- [ ] Stipple / micro-dot styles

### Phase 4 — AI-assisted
- [x] AI planner: subject, must-include parts, ignore regions, suggested dots (`--ai`)
- [x] Automatic style selection (`--style auto`)
- [x] AI guess test: is the picture hidden before solving and clear after?
- [x] Mystery mode with AI auto-tune (`--mystery auto`)
- [x] Local judge with CLIP (free, offline): 79% verdict agreement with the LLM judge, ~60x faster
- [x] Dataset (Quick, Draw!) + distill the judge into a small trained model (AUC 0.75 → 0.82)
- [ ] More data and categories; learn mystery settings directly; user feedback (thumbs up/down)
- [x] Subject classification via the planner (coloring page / cartoon / photo / render)
  - Portrait: face outline, hair silhouette, key facial features
  - Animal: silhouette + major features
  - Vehicle: body, wheels, windows
- [ ] Automatic quality evaluation and retry
- [x] Natural-language requests ("easy puzzle for my 6-year-old"), via the planner

### Phase 5 — Product
- [x] FastAPI backend
- [ ] Personalized books: upload 15–20 photos → "Tejas's Connect-the-Dots Adventure"
- [ ] AI-generated covers and captions
- [x] Local web front end
- [ ] iOS front end

---

## Mystery mode

A puzzle is more fun when you can't tell what it is until you connect the
dots. Normal dot puzzles give it away: dots crowd along the outline and the
rest of the page is empty. Mystery mode adds light **background filler
lines**, numbered like everything else, so dots cover the page evenly, and
it numbers small give-away features (eyes) last.

| Level | What it does |
|-------|--------------|
| 0 | Off: faithful picture |
| 1 | Filler lines in the background |
| 2 | More fillers, also in empty areas inside the subject; picture geometry stays intact |
| auto | Tries 2, 1 and 0; keeps the most mysterious recognizable result that preserves detail and planned-part coverage relative to level 0 |

The requested dot count is the **picture budget**. Mystery fillers add extra
dots rather than taking them from the subject. Both fixed mystery levels keep
the same picture segments as level 0; numbering and filler strokes change.

**Guess test**: a judge sees the unsolved page, then the answer key, with
no hints, and says what it sees. Three judge options are available; the trained judge instead predicts recognition of the subject using CLIP features:

| Judge | Cost | Speed | Verdict agreement with OpenAI judge (24 real puzzles) |
|-------|------|-------|-----------------------------|
| `trained` (CLI default): CLIP + a small head trained to imitate the OpenAI judge | free, offline | ~50 ms | 88% |
| `clip`: zero-shot CLIP, picks from 354 categories | free, offline | ~50 ms | 79% |
| `openai`: vision LLM, names anything | ~1¢ | ~3 s | n/a |

Install the local judge with `pip install -e ".[clip]"` (PyTorch, ~600 MB model
downloaded on first use). Compare the judges with `python scripts/compare_judges.py`. Verdicts: *mystery* (fooled
before, right after), *too easy* (guessed before solving), *unrecognizable*
(wrong even after). Turn it on in the web app, or:

```bash
dot2dot penguin.jpg --dots 200 --mystery auto           # the trained local judge picks the level
dot2dot penguin.jpg --dots 200 --mystery auto --judge openai
dot2dot penguin.jpg --dots 200 --mystery 2 --guess-test
python scripts/eval_samples.py --styles auto --mystery 0 1 2 --judge --dots 200
```

On the sample set: without mystery mode, 4 of 8 puzzles were guessable from
the dots alone; auto-tune turns most into mysteries, and falls back to level 0
when fillers would make the subject unrecognizable (the Hulk coloring page).

## Training the judge (knowledge distillation)

The `trained` judge was built by teaching a small model to imitate the
OpenAI judge, so the app gets most of its judgment for free:

```bash
python scripts/make_dataset.py --categories 100 --per-category 3 --label   # 600 puzzles, ~$2.80 of labels
python scripts/train_judge.py                                              # trains in seconds
```

1. **Data**: 300 doodles from 100 categories of Google's
   [Quick, Draw!](https://quickdraw.withgoogle.com/data) (CC BY 4.0),
   each made into a plain and a mystery puzzle (600 puzzles).
2. **Labels**: the OpenAI judge makes blind guesses on every unsolved page
   and answer key: 1,200 examples of "would it name the subject?".
3. **Model**: frozen CLIP features → a small neural network (one hidden
   layer, 33K weights, 132 KB file), trained with PyTorch.
4. **Evaluation**: 5-fold cross-validation grouped by category, so test
   folds contain only subjects the model never saw.

| | ROC AUC (unseen categories) | Verdicts on 24 real puzzles |
|---|---|---|
| zero-shot CLIP | 0.752 | 79% |
| trained judge | **0.818** | **88%** |

Accuracy barely moves (87% of labels are "no", so always answering "no" scores 87%);
ROC AUC, which measures ranking and ignores the imbalance, is the honest metric.

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
line. The core pipeline works without a key. The web app falls back to
generation without a plan if planning fails; the CLI reports the error and
exits. The planner sends a downscaled image to OpenAI. Selecting the OpenAI
judge also sends images for recognition tests. The planner's suggested dot
count is advisory; generation uses the requested dot count.

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
Images are sent to OpenAI when you enable the AI planner or select the OpenAI judge. The local CLIP and trained judges run on your machine after their models are downloaded. Put personal test photos in `samples/private/`,
which is git-ignored.

## A note on images

The intended use is **your own photos**. Generating and distributing puzzles from celebrity or other copyrighted images can raise copyright and right-of-publicity issues, so the product is designed around personal photos, not celebrity content.

---

## License

TBD
