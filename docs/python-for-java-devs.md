# dot2dot for Java developers

A tour of this codebase for someone coming from Java + Spring. It covers the
Python tooling first, then walks through each pipeline stage.

---

## 1. Project setup: Maven → pip / venv / pyproject

| Java / Maven | Python | In this repo |
|---|---|---|
| `pom.xml` | `pyproject.toml` | Dependencies, project metadata, tool config |
| `~/.m2/repository` (shared) | `.venv/` (per project) | Isolated install of this project's libraries |
| Maven Central | PyPI (pypi.org) | Where `pip` downloads packages from |
| `mvn install` | `pip install -e ".[dev]"` | Install the project + its deps |
| `<scope>test</scope>` | `[project.optional-dependencies] dev` | pytest is only installed with `[dev]` |
| `mvn test` | `pytest` | Run the tests |
| `java -jar app.jar` | `python -m dot2dot` / `dot2dot` | Run the app |

**Virtual environments** are the biggest difference. Java isolates dependencies per
build via the classpath. Python installs packages *into an interpreter*, so each
project gets its own copy of the interpreter in `.venv/`:

```bash
python3 -m venv .venv          # create (once)
source .venv/bin/activate      # "use this project's Python" for this shell
pip install -e ".[dev]"        # install deps + this project
deactivate                     # back to system Python
```

`-e` means *editable*: rather than copying the code into `.venv`, pip links
to your source folder, so edits take effect without reinstalling. It's a bit like running
from `target/classes` in your IDE, not from a built jar.

---

## 2. Code organization: packages and modules

| Java | Python |
|---|---|
| package `com.example.dot2dot` (a directory) | package `dot2dot/` (a directory with `__init__.py`) |
| class file `Simplify.java` | module `simplify.py` |
| `import com.example.dot2dot.Simplify;` | `from dot2dot.simplify import simplify` |
| `public static void main(String[] args)` | `dot2dot/__main__.py` (runs on `python -m dot2dot`) |
| `public` / `private` | No access control. A leading `_` (like `_fill_gaps`) means "internal, don't call from outside" |

**Not everything needs a class.** In Java you'd write a `SimplifyUtils` class
with static methods. In Python you just write functions in a module; the
module *is* the namespace. That's why `simplify.py`, `order.py`, etc. contain
only functions.

`__init__.py` runs when the package is first imported. Ours re-exports the
public API (`Dot`, `Puzzle`, `generate`) so callers can write
`from dot2dot import generate`. That's similar to a facade.

---

## 3. Language features you'll see here

### Dataclasses ≈ Java records

```python
@dataclass(frozen=True)
class Dot:
    number: int
    x: float
    y: float
    label_dx: float = 0.0     # default value
```

```java
record Dot(int number, double x, double y, double labelDx) {}
```

`@dataclass` generates `__init__` (the constructor), `__eq__`, and `__repr__`
(toString). `frozen=True` makes it immutable, like a record. Decorators
(`@something`) *look* like Java annotations, but they actually run code
that transforms the class or function.

### Type hints ≈ static types, but not enforced

```python
def simplify(outline: np.ndarray, n: int) -> np.ndarray:
```

Python doesn't check these at runtime. They exist for your IDE and for
optional checkers like `mypy` (think of a compile step you opt into). `str | Path`
means "either type", and `list[Dot]` is like `List<Dot>`.

### Tuples and unpacking

```python
height, width = gray.shape            # shape is a tuple (rows, cols)
for (x, y), (dx, dy) in zip(points, normals):
```

`zip` walks two sequences in lockstep. Java has no built-in equivalent; you'd
loop over an index.

### List comprehensions ≈ streams

```python
large = [c for c in contours if cv2.contourArea(c) >= min_area]
```

```java
var large = contours.stream().filter(c -> area(c) >= minArea).toList();
```

### Exceptions

Same idea as Java, but there are no checked exceptions. `raise ValueError(...)`
is roughly `throw new IllegalArgumentException(...)`.

---

## 4. NumPy: the one new concept that matters

OpenCV images and our outlines are **NumPy arrays**: n-dimensional, fixed-type
arrays backed by native memory. Think `double[][]`, but with *vectorized*
operations that run in C:

```python
outline.shape        # (3000, 2): 3000 points, each (x, y)
outline[:, 0]        # all x values (column 0)
outline[-1]          # last point (negative indices count from the end)
outline[indices]     # pick rows by a list of indices
outline - pen        # subtract a point from all 3000 points at once
np.linalg.norm(outline - pen, axis=1)   # distance from pen to every point
```

The last line would be a `for` loop in Java. In NumPy, replacing loops with
whole-array operations is the main way to make code fast ("vectorizing").

An image is a `(height, width, 3)` array of bytes (blue, green, red). After
grayscale conversion it's `(height, width)`. **Note: rows come first, so it's
`[y, x]`, not `[x, y]`.**

---

## 5. The pipeline, stage by stage

`pipeline.py` wires everything together. It plays the role of a Spring `@Service`
calling other services, except these are plain functions:

```
preprocess → make_mask → find_outlines → allocate_dots → simplify
           → order_outlines → outward_normals → Puzzle → save_pdf / save_svg
```

### Stage 1: `preprocess.py`: load, shrink, grayscale
- `cv2.imread` loads the file. **It returns `None` on failure instead of throwing**,
  so we check for that and raise ourselves.
- Shrink to 800px max: phone photos are huge, and fine detail is noise for us.
- Grayscale + Gaussian blur: one brightness channel, smoothed.

### Stage 2: `contours.py`: find the subject's outline
- **Otsu thresholding** turns the image into black/white by automatically
  picking the brightness cutoff that best separates two groups of pixels.
- If the image border came out white, the background was classified as the
  subject, so we invert.
- **Morphological closing** fills small holes and gaps.
- `cv2.findContours` traces the boundary of each white blob as a list of pixels.
  We keep the largest one(s).

Limitation: this only works when the subject contrasts with a plain
background. Real background removal comes in Phase 2.

### Stage 3: `simplify.py`: thousands of pixels → N dots
This is the most algorithmic part.
1. **Corners**: at each point, compare the direction coming in with the
   direction going out (measured a few pixels back and ahead). A turn
   sharper than 45° is a corner, and every corner gets a dot.
2. **Even spacing with a curve bonus**: between corners, dots go at equal
   steps of "effort" = distance + a bonus for turning. Straight runs are
   spaced by length; tight curves (eyes, fingertips) get a few extra dots.
3. **Readable limits**: dots are never closer than about 1/70 of the image
   size (`max_readable_dots`), and where lines meet, a dot on top of an
   earlier line's dot is dropped (`declutter`).

The first version used Douglas–Peucker (`approxPolyDP`), a classic
polyline-simplification algorithm. It's great at corners but picks irregular
points on smooth pixel curves, which looked like extra edges on a circle.
`arc[k]` (cumulative distance along the outline) is still the key data
structure: it turns "how far apart are these two dots along the shape?" into
a subtraction, and `np.searchsorted` finds "the point 120 px along" in O(log n).

### Stage 4: `order.py`: which dot is #1, and which way?
- The **shoelace formula** gives a polygon's signed area. The sign tells us the
  direction (clockwise vs counter-clockwise), and we reverse the points if needed.
- `np.roll` rotates the array so the topmost point is index 0.
- With multiple outlines, a greedy **nearest-neighbor** pass picks the closest
  outline next.

### Stage 5: `labels.py`: where to put each number
For each dot, compute the tangent (from the previous dot to the next), rotate it 90° to get a
normal, and flip the normal if it points into the shape. The number goes a few
pixels along the normal, outside the outline.

### Stage 6: `render.py` / `pdf.py`: output
- SVG is built with f-strings, Python's string interpolation (`f"x={x:.1f}"` ≈
  `String.format("x=%.1f", x)`). It's a StringBuilder approach, no library.
- ReportLab draws on a PDF canvas, similar to Java's `Graphics2D`. **PDF y-axis
  points up and image y-axis points down**, so `to_page()` flips it.
- `to_page` is a **nested function** (a closure). It can read `scale` and
  `offset_x` from the enclosing function, like a lambda capturing effectively-final locals.

### Line-art style: `lineart.py` + `skeleton.py`
Used with `--style lineart` to trace the lines *inside* a drawing.

- **Ink mask**: keep pixels whose brightest color channel (HSV "value") is
  below 60. Plain grayscale would treat deep red as dark; HSV value doesn't.
- **Morphology** is the toolbox here. Think of sliding a small shape (a
  "kernel") over the image:
  - *opening* erases anything the kernel can't fit inside (thin specks, or,
    with a big disk, all lines, leaving only solid black areas),
  - *dilation* grows shapes, *closing* fills small gaps.
- **Skeletonize** (scikit-image) thins every ink line down to a 1-pixel
  centerline, like drawing a pencil line down the middle of a marker stroke.
- **Graph tracing** (`skeleton.py`) is plain algorithms code you'd write the
  same way in Java: pixels are nodes, touching pixels are edges. Walk from
  every end/junction until the next one to get branches, then re-join
  branches that continue smoothly through a junction (pairing ends by the
  dot product of their directions). Python notes:
  - `Pixel = tuple[int, int]` is a *type alias*, like a tiny typedef.
  - `set[Pixel]` / `dict[Pixel, int]` work like `HashSet`/`HashMap`; tuples
    are hashable, so they can be keys (no need for a `Point` class with
    `equals`/`hashCode`).
  - `frozenset((a, b))` is an immutable set used as an *unordered* edge key,
    so `(a, b)` and `(b, a)` are the same edge.
  - `_SkeletonGraph` is a class because it bundles state (pixels, degrees)
    with behavior; a leading `_` marks it module-private.
  - `walk` is defined *inside* `branches`, a closure over `visited`, like a
    local class or lambda capturing a local collection.

### Silhouette: the paint-bucket trick
`lineart.silhouette` finds the subject's outer border, like the paint-bucket
tool in an image editor:
- **Barrier** = ink pixels *or* pixels whose color differs from the
  background (measured in Lab color space, where distance matches what the
  eye sees). The suit's red/blue blocks the paint even where the black
  outline has a gap.
- `cv2.connectedComponents` labels every separate open region; regions
  touching a strip just inside the image edge are background. Everything
  else is the subject. (Seeding from a strip, not the very edge, copes with
  frames drawn around the picture.)
- Interior lines that run along the silhouette are clipped out
  (`clip_paths`) so the edge isn't traced twice.
- `Path.essential = True` marks the silhouette so `select_paths` never
  drops it. `dataclasses.replace(path, points=...)` copies a frozen
  dataclass with one field changed, like a record "wither" in Java.

### Photo style: `photo.py`
- **rembg** runs a U²-Net segmentation model through **onnxruntime**: a
  trained neural network exported to the portable ONNX format, run locally
  like any library call. It returns a mask of the subject.
- **XDoG**: blur the image twice (a little, then more) and subtract.
  Flat areas cancel; edges and thin dark features remain. That turns a photo
  into a sketch the lineart code can trace.
- **YuNet** (`cv2.FaceDetectorYN`) finds faces; lines inside get
  `weight=3`, so `allocate_dots` treats them as three times longer.
- `@lru_cache(maxsize=1)` on `_rembg_session()` loads the model once and
  reuses it, effectively a lazily created singleton bean.
- `from rembg import remove` sits *inside* the function: a deliberate lazy
  import, because loading rembg takes about a second and only photos need it.
- MediaPipe (Google's face-landmark library) was tried first but crashes on
  this Mac with Python 3.14, so it was dropped. Prefer dependencies you can
  verify on your actual platform.

### Image loading
`preprocess.py` reads files with **Pillow** rather than OpenCV:
`ImageOps.exif_transpose` applies the rotation phones store in metadata,
transparent areas are blended onto white, and `pillow_heif` adds iPhone HEIC
support. All three would otherwise be bugs: sideways photos, black
backgrounds, unreadable files.

### Quality score: `quality.py`
Measures the puzzle against the full set of extracted lines (`Result.reference`).
The key tool is a **distance transform**: for every pixel it stores the
distance to the nearest line pixel. "Is this reference pixel within 12 px of
a solution line?" then becomes a single array lookup for every pixel at once:
`(to_solution[reference_pixels] <= tolerance).mean()`.

`Quality.overall` is a `@property`, a method you read like a field (Java
would call it `getOverall()`).

### Number placement: `labels.py`
Each number tries 16 positions around its dot and scores each by its
clearance (distance from the number's box to the nearest other dot, line, or
already-placed number). The highest score wins. All 16 candidates are scored
at once with NumPy broadcasting: arrays shaped `(16, 1, 2)` minus `(1, M, 2)`
give a `(16, M, 2)` grid of differences, with no nested loops.

### AI planner: `planner.py` + `guidance.py`
- `Planner` is a `typing.Protocol`, Python's structural interface: any class
  with a matching `plan(image, request)` method satisfies it, with no
  `implements`. `OpenAIPlanner` is one implementation; tests use a
  `FakePlanner` with the same method.
- **Structured outputs**: the request includes a JSON Schema
  (`PLAN_SCHEMA`) with `strict: True`, so the model must return JSON matching
  it, much like binding a response to a DTO. `parse_plan` then clamps
  anything out of range: never trust model output blindly.
- The planner only adjusts knobs (weights, ignore regions, repairs). The
  geometry stays deterministic, so a bad plan can't break a puzzle.
- `config.py` calls `load_dotenv()`, which reads `.env` into environment
  variables, like `application.properties`, but kept out of git for secrets.
- **Model choice was measured, not guessed**: three models on the same
  samples, comparing box accuracy and latency (see `scripts/eval_samples.py --ai`).

### A dataclass gotcha
`@dataclass` generates `__eq__` comparing every field. With a NumPy array
field, `a == b` returns an array, not a bool, so `path in paths` raises an
error. `Path` uses `@dataclass(frozen=True, eq=False)` to compare by identity,
the same as Java's default `Object.equals`.

### Mocking: `monkeypatch`
`tests/test_planner.py` swaps the real planner for a fake with
`monkeypatch.setattr(web, "get_planner", lambda: fake)`, pytest's equivalent
of Mockito's `when(...).thenReturn(...)`. The patch is undone automatically
after each test.

### CLI: `cli.py`
`argparse` is Python's built-in equivalent of picocli / Apache Commons CLI.
`main()` returns an exit code, and `__main__.py` passes it to `sys.exit`.
`main(argv)` takes the argument list as a parameter so tests can call it directly.

---

## 6. Tests: JUnit → pytest

| JUnit | pytest |
|---|---|
| `@Test void foo()` | any function named `test_*` in a `test_*.py` file |
| `assertEquals(a, b)` | plain `assert a == b` (pytest prints a detailed diff on failure) |
| `@BeforeEach` / test helpers | **fixtures** in `conftest.py` |
| `@TempDir Path dir` | built-in `tmp_path` fixture |
| capturing System.err | built-in `capsys` fixture |

Fixtures are dependency injection: a test asks for `square_image` just by
naming a parameter that way, and pytest finds the fixture with that name and passes its result. It's
the same idea as Spring autowiring by name.

```bash
pytest              # run all
pytest -q           # quiet
pytest -k order     # only tests whose name contains "order"
pytest -x           # stop at first failure
```

---

## 7. The web app: FastAPI ≈ Spring MVC

`dot2dot/web/app.py` is the whole backend.

| Spring | FastAPI (this repo) |
|---|---|
| `@SpringBootApplication` + embedded Tomcat | `app = FastAPI()` + `uvicorn.run(app)` |
| `@PostMapping("/api/puzzles")` | `@app.post("/api/puzzles", response_model=PuzzleOut)` |
| `@RequestParam MultipartFile image` | `image: UploadFile = File(...)` |
| `@RequestParam int dots` (form field) | `dots: int = Form(150)`, converted and validated automatically |
| `@PathVariable String id` | `def download_pdf(puzzle_id: str)` with `"/api/puzzles/{puzzle_id}/pdf"` |
| Response DTO + Jackson | Pydantic `BaseModel` (`PuzzleOut`), serialized to JSON |
| `throw new ResponseStatusException(BAD_REQUEST, ...)` | `raise HTTPException(400, "...")` |
| `src/main/resources/static/` | `StaticFiles(directory=...)` mounted at `/static` |
| `MockMvc` tests | `TestClient(app)` in `tests/test_web.py` |
| Swagger / springdoc | Built in: open http://127.0.0.1:8000/docs |

Things that differ:
- **`def` vs `async def` endpoints.** FastAPI runs `async def` handlers on
  one event loop (like WebFlux) and plain `def` handlers on a thread pool
  (like classic Spring MVC). Image processing is CPU-heavy and blocking, so
  `create_puzzle` is a plain `def`. Making it `async def` would freeze the
  server for everyone while one image processes.
- **No DI container.** Shared objects like the PDF cache are plain
  module-level variables (`pdfs = _PdfCache(...)`), effectively a singleton.
  FastAPI does have `Depends(...)` for injection when you need it.
- **In-memory state.** Recent PDFs live in an LRU cache in memory and
  disappear on restart. A real deployment would use object storage (S3).
- `uvicorn dot2dot.web.app:app --reload` restarts on code changes, like
  Spring DevTools.

The front end (`static/index.html`) is plain HTML/CSS/JS with no build step:
a form posts the image with `fetch` as `multipart/form-data`, and the JSON
response's SVG strings are inserted into the page.

## 8. Spring analogies for later phases

| Coming up | Python tool | Spring equivalent |
|---|---|---|
| Calling the Claude API | `anthropic` SDK | a typed REST client (like `RestClient`) |
| Background jobs | `asyncio` tasks / Celery | `@Async` / Spring Batch |
| Config & secrets | env vars + `pydantic-settings` | `application.yml` + `@ConfigurationProperties` |
