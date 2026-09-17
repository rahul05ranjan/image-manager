import pytest
from PIL import Image

import image_manager


@pytest.mark.parametrize(
    "filename",
    [
        "photo.jpg",
        "photo.jpeg",
        "photo.JPG",
        "photo.JPEG",
        "my photo (1).jpg",
        "photo.dat",
    ],
)
def test_convert_jpg_to_png(tmp_path, filename):
    source = tmp_path / filename
    with Image.new("RGB", (3, 2), "red") as image:
        image.save(source, "JPEG")
    with Image.open(source) as image:
        expected_pixels = image.tobytes()

    image_manager.convert_jpg_to_png(source)

    assert not source.exists()
    output = source.with_suffix(".png")
    assert list(tmp_path.iterdir()) == [output]
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (3, 2)
        assert image.mode == "RGB"
        assert image.tobytes() == expected_pixels


@pytest.mark.parametrize("filename", ["notes.txt", "not_an_image.jpg", "existing.png"])
def test_convert_jpg_to_png_leaves_non_jpeg_untouched(tmp_path, filename):
    source = tmp_path / filename
    if source.suffix == ".png":
        with Image.new("RGB", (2, 2), "blue") as image:
            image.save(source, "PNG")
    else:
        source.write_text("This is not a JPEG.")
    original_bytes = source.read_bytes()

    image_manager.convert_jpg_to_png(source)

    assert source.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [source]


def test_convert_jpg_to_png_does_not_overwrite_existing_png(tmp_path):
    """Known bug: conversion silently replaces an existing same-stem PNG."""
    source = tmp_path / "photo.jpg"
    output = tmp_path / "photo.png"
    with Image.new("RGB", (2, 2), "red") as image:
        image.save(source, "JPEG")
    with Image.new("RGB", (2, 2), "blue") as image:
        image.save(output, "PNG")
    original_bytes = output.read_bytes()

    # Catch expected FileExistsError
    try:
        image_manager.convert_jpg_to_png(source)
    except FileExistsError:
        pass

    assert output.read_bytes() == original_bytes
    assert source.exists()


def test_convert_jpgs_to_pngs_mixed_directory_recursively(tmp_path):
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    sources = [
        tmp_path / "photo.jpg",
        tmp_path / "another.jpeg",
        nested / "my photo (1).JPG",
    ]
    for source in sources:
        with Image.new("RGB", (2, 3), "green") as image:
            image.save(source, "JPEG")
    text = nested / "notes.txt"
    text.write_text("Keep these notes.")
    png = tmp_path / "existing.png"
    with Image.new("RGB", (1, 1), "blue") as image:
        image.save(png, "PNG")
    originals = {path: path.read_bytes() for path in (text, png)}

    image_manager.convert_jpgs_to_pngs(tmp_path)

    for source in sources:
        assert not source.exists()
        with Image.open(source.with_suffix(".png")) as image:
            assert image.format == "PNG"
            assert image.size == (2, 3)
    for path, original_bytes in originals.items():
        assert path.read_bytes() == original_bytes
    expected_files = {source.with_suffix(".png") for source in sources}
    expected_files.update([text, png])
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == expected_files


def test_convert_jpgs_to_pngs_empty_directory(tmp_path):
    image_manager.convert_jpgs_to_pngs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_convert_jpgs_to_pngs_nonexistent_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        image_manager.convert_jpgs_to_pngs(tmp_path / "missing")


def test_convert_jpgs_to_pngs_file_path_is_rejected(tmp_path):
    """Validation accepts files, but directory iteration rejects them."""
    source = tmp_path / "photo.jpg"
    with Image.new("RGB", (2, 2), "red") as image:
        image.save(source, "JPEG")
    original_bytes = source.read_bytes()

    with pytest.raises(NotADirectoryError):
        image_manager.convert_jpgs_to_pngs(source)

    assert source.read_bytes() == original_bytes
    assert not source.with_suffix(".png").exists()


def test_is_jpg_checks_contents_instead_of_extension(tmp_path):
    jpeg = tmp_path / "photo.txt"
    text = tmp_path / "notes.jpg"
    with Image.new("RGB", (1, 1), "red") as image:
        image.save(jpeg, "JPEG")
    text.write_text("Not an image.")
    assert image_manager.is_jpg(jpeg) is True
    assert image_manager.is_jpg(text) is False


def test_is_jpg_empty_file_raises_index_error(tmp_path):
    """Current detection cannot handle an empty file."""
    source = tmp_path / "empty.txt"
    source.touch()
    with pytest.raises(IndexError):
        image_manager.is_jpg(source)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("photo", "spr_photo"),
        ("spr_photo", "spr_photo"),
        ("1photo", "spr_photo"),
        ("my photo (1)", "spr_my photo _1"),
    ],
)
def test_sanitize_name(name, expected):
    assert image_manager.sanitize_name(name) == expected


@pytest.mark.parametrize(
    "text, prefix, expected",
    [
        ("photo", "spr", "spr_photo"),
        ("_photo", "spr", "spr_photo"),
        ("spr_photo", "spr", "spr_photo"),
        ("sprout", "spr", "sprout"),
        ("", "img", "img_"),
        ("photo", "img", "img_photo"),
    ],
)
def test_check_prefix(text, prefix, expected):
    assert image_manager.check_prefix(text, prefix) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("photo", "photo"),
        ("_photo", "_photo"),
        ("1photo", "_photo"),
        (" photo", "_photo"),
    ],
)
def test_check_starting_character(text, expected):
    assert image_manager.check_starting_character(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("photo", "photo"),
        ("photo (1)", "photo _1"),
        ("(photo)", "_photo"),
        ("photo__", "photo_"),
        ("photo !-", "photo !-"),
        ("", ""),
    ],
)
def test_replace_special_characters(text, expected):
    assert image_manager.replace_special_characters(text) == expected


@pytest.mark.parametrize(
    "function", [image_manager.check_starting_character, image_manager.sanitize_name]
)
def test_name_helpers_reject_empty_names(function):
    with pytest.raises(IndexError):
        function("")
