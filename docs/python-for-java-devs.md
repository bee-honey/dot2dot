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
1. **Douglas–Peucker** (`approxPolyDP`): given a tolerance ε, it drops points
   that are within ε of a straight line between their neighbors. Corners
   survive and straight runs collapse. We **binary-search ε** to get ≤ N points.
2. **Merge clumps**: a diagonal edge drawn in pixels is a staircase, so one
   corner can produce 2–3 almost identical points. We keep the middle one of each clump.
3. **Fill gaps**: while we have fewer than N dots, we insert a dot halfway along the
   longest stretch of outline between two dots.

`arc[k]` (cumulative distance along the outline) is the key data structure.
It turns "how far apart are these two dots along the shape?" into a subtraction.

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

## 7. Spring analogies for later phases

| Coming up | Python tool | Spring equivalent |
|---|---|---|
| Web API | FastAPI | Spring MVC / `@RestController` |
| Request/response models | Pydantic | DTOs + Bean Validation |
| Server | uvicorn | embedded Tomcat |
| Prototype UI | Streamlit | (no real equivalent; a whole UI in one script) |
