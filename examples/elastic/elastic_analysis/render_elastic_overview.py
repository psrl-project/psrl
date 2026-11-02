"""Export the resource elasticity overview PDF as a vector SVG."""

from pathlib import Path

import pymupdf


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    figures = root / "docs" / "_static" / "img"
    source = figures / "resource_elasticity_overview.pdf"
    target = source.with_suffix(".svg")
    with pymupdf.open(source) as document:
        if len(document) != 1:
            raise ValueError("The overview figure must contain exactly one page.")
        svg = document[0].get_svg_image(text_as_path=True)
    target.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
