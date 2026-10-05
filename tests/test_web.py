"""API tests. TestClient calls the app in-process, like Spring's MockMvc."""

import pytest
from fastapi.testclient import TestClient

from dot2dot.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def upload(client, image_path, **form):
    with open(image_path, "rb") as f:
        return client.post("/api/puzzles", files={"image": ("square.png", f, "image/png")}, data=form)


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "dot2dot" in response.text


def test_create_puzzle_returns_both_images_and_quality(client, square_image):
    response = upload(client, square_image, dots="20", style="outline")
    assert response.status_code == 200
    body = response.json()
    assert body["dots"] == 20
    assert body["puzzle_svg"].startswith("<svg") and "<line" not in body["puzzle_svg"]
    assert "<line" in body["solution_svg"]
    assert body["check_png"].startswith("data:image/png;base64,")
    assert body["quality"]["outline_coverage"] == 1.0


def test_pdf_can_be_downloaded_after_generation(client, square_image):
    body = upload(client, square_image, dots="20", style="outline").json()
    response = client.get(body["pdf_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_rejects_non_image_upload(client, tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    response = upload(client, bogus)
    assert response.status_code == 422
    assert "Could not read image" in response.json()["detail"]


def test_rejects_bad_settings(client, square_image):
    assert upload(client, square_image, style="watercolor").status_code == 400
    assert upload(client, square_image, dots="5").status_code == 400


def test_unknown_pdf_is_404(client):
    assert client.get("/api/puzzles/nope/pdf").status_code == 404
