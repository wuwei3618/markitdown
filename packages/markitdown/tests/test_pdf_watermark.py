#!/usr/bin/env python3 -m pytest
"""Tests for the unified, evidence-fusion watermark remover in the PDF converter.

The single ``pdf_remove_watermarks`` flag fuses several signals: rotation and
cross-page repetition act as anchors, and a run is removed only when an anchor
is corroborated by a second independent family of evidence (rotation,
repetition, light color, oversized font, or margin band).

Covered here:
- Unit behavior of the building blocks (rotation, signature, lightness, fusion).
- End-to-end: a diagonal light/oversized stamp is removed; repeated margin
  header/footer is removed; unique body text survives.
- Precision: a lone rotated run (single signal) and a lone repeated in-body
  run (single signal) are NOT removed.
"""

import io

import pytest

from markitdown import MarkItDown, StreamInfo
from markitdown.converters._pdf_converter import (
    _color_lightness,
    _is_rotated_char,
    _is_watermark_run,
    _word_signature,
)


def _has_fpdf2() -> bool:
    try:
        import fpdf  # noqa: F401

        return True
    except ImportError:
        return False


BODY_TEXT = "Quarterly revenue report for fiscal year 2024."
WATERMARK_TEXT = "CONFIDENTIAL"

# Distinct, full-sentence body text per page (different words AND positions),
# so the boilerplate heuristic does not mistake body content for a repeat.
BODY_SENTENCES = [
    "Chapter zero introduces the migration roadmap and the rollout schedule details.",
    "Section one analyzes throughput regressions across the production cluster fleet.",
    "Appendix two compares pricing tiers for several downstream analytics workloads.",
    "Exhibit three summarizes interviews with field reliability and platform engineers.",
]
# A unique phrase from each sentence above, used to assert survival per page.
UNIQUE_PHRASES = [
    "migration roadmap",
    "throughput regressions",
    "pricing tiers",
    "field reliability",
]


def _make_diagonal_watermark_pdf() -> bytes:
    """Single page: horizontal body text + a 45-degree light, oversized stamp."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", size=14)
    pdf.set_xy(10, 30)
    pdf.multi_cell(0, 8, BODY_TEXT)

    pdf.set_font("Helvetica", size=50)
    pdf.set_text_color(200, 200, 200)  # light gray
    with pdf.rotation(45, x=105, y=150):
        pdf.text(40, 150, WATERMARK_TEXT)

    return bytes(pdf.output())


def _make_multipage_pdf(num_pages: int = 4) -> bytes:
    """Multipage PDF: repeated header/footer + light oversized horizontal stamp.

    Each page also has distinct body text so we can confirm real content
    survives boilerplate removal.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for i in range(num_pages):
        pdf.add_page()

        # Repeated header in the top margin band (same text + position).
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", size=10)
        pdf.set_xy(10, 8)
        pdf.cell(0, 6, "ACME CORP - INTERNAL USE ONLY")

        # Repeated horizontal watermark: light gray + oversized, mid-page.
        pdf.set_text_color(200, 200, 200)
        pdf.set_font("Helvetica", size=40)
        pdf.set_xy(10, 150)
        pdf.cell(0, 6, "DRAFT COPY")

        # Unique body text per page (normal black, body size).
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", size=14)
        pdf.set_xy(10, 40)
        pdf.multi_cell(0, 8, BODY_SENTENCES[i % len(BODY_SENTENCES)])

        # Repeated footer in the bottom margin band.
        pdf.set_font("Helvetica", size=10)
        pdf.set_xy(10, 285)
        pdf.cell(0, 6, "Copyright ACME Corp")

    return bytes(pdf.output())


def _make_repeated_table_header_pdf(num_pages: int = 3) -> bytes:
    """Multipage PDF where a dark, body-size table header repeats in-body.

    This is the false-positive trap for repetition-only removal: the header row
    recurs at the same position on every page but is legitimate content, so the
    fused rule (repetition is a single family) must keep it.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for p in range(num_pages):
        pdf.add_page()
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", size=12)
        # Repeated header row mid-page (not in margin band).
        pdf.set_xy(10, 100)
        pdf.cell(40, 8, "Product")
        pdf.cell(40, 8, "Quantity")
        pdf.cell(40, 8, "Price")
        # Unique data row per page.
        pdf.set_xy(10, 110)
        pdf.cell(40, 8, f"Widget{p}")
        pdf.cell(40, 8, f"{p + 1}0")
        pdf.cell(40, 8, f"${p + 1}.99")
    return bytes(pdf.output())


def _convert(pdf_bytes: bytes, **kwargs):
    md = MarkItDown()
    return md.convert_stream(
        io.BytesIO(pdf_bytes),
        stream_info=StreamInfo(extension=".pdf", mimetype="application/pdf"),
        **kwargs,
    )


class TestUnitHelpers:
    """Unit tests for the detection building blocks."""

    def test_axis_aligned_char_is_not_rotated(self):
        assert _is_rotated_char({"matrix": (1, 0, 0, 1, 100, 200)}) is False

    def test_rotated_char_is_detected(self):
        assert _is_rotated_char({"matrix": (0.7071, 0.7071, -0.7071, 0.7071, 0, 0)})

    def test_missing_matrix_treated_as_not_rotated(self):
        assert _is_rotated_char({}) is False
        assert _is_rotated_char({"matrix": None}) is False

    def test_signature_same_text_same_position_matches(self):
        a = {"text": "Confidential", "x0": 100.0, "top": 50.0}
        b = {"text": "confidential ", "x0": 101.0, "top": 51.0}
        assert _word_signature(a) == _word_signature(b)

    def test_signature_different_position_differs(self):
        a = {"text": "Total", "x0": 100.0, "top": 50.0}
        b = {"text": "Total", "x0": 400.0, "top": 700.0}
        assert _word_signature(a) != _word_signature(b)

    def test_color_lightness(self):
        assert _color_lightness(None) == 0.0
        assert _color_lightness((0, 0, 0)) == 0.0
        assert _color_lightness((1, 1, 1)) == 1.0
        assert _color_lightness(0.8) == pytest.approx(0.8)
        assert _color_lightness((200, 200, 200)) == pytest.approx(200 / 255)


class TestFusionLogic:
    """Unit tests for _is_watermark_run evidence fusion, using fake runs/page."""

    class _FakePage:
        width = 600.0
        height = 800.0

    def _run(self, **over):
        run = {
            "text": "X",
            "x0": 250.0,
            "x1": 350.0,
            "top": 400.0,
            "bottom": 440.0,
            "rotated": False,
            "lightness": 0.0,
            "size": 12.0,
        }
        run.update(over)
        return run

    def _analysis(self, **over):
        a = {"num_pages": 4, "repeat_counts": {}, "body_median_size": 12.0}
        a.update(over)
        return a

    def test_lone_rotation_is_kept(self):
        # Rotated only -> single family -> not a watermark.
        run = self._run(rotated=True)
        assert _is_watermark_run(run, self._FakePage(), self._analysis()) is False

    def test_lone_repetition_is_kept(self):
        # Repeated in-body, dark, body-size -> single family -> kept.
        run = self._run()
        analysis = self._analysis(repeat_counts={_word_signature(run): 4})
        assert _is_watermark_run(run, self._FakePage(), analysis) is False

    def test_rotation_plus_light_is_removed(self):
        # Rotation anchor + light color family -> removed.
        run = self._run(rotated=True, lightness=0.8)
        assert _is_watermark_run(run, self._FakePage(), self._analysis()) is True

    def test_repetition_plus_margin_is_removed(self):
        # Repetition anchor + margin band family -> removed (header/footer).
        run = self._run(top=10.0, bottom=24.0)  # top margin band
        analysis = self._analysis(repeat_counts={_word_signature(run): 4})
        assert _is_watermark_run(run, self._FakePage(), analysis) is True

    def test_cosmetic_anchor_single_page(self):
        # Very light + oversized + centered -> cosmetic anchor (no rep needed).
        run = self._run(lightness=0.85, size=40.0)
        analysis = self._analysis(num_pages=1)
        assert _is_watermark_run(run, self._FakePage(), analysis) is True

    def test_big_dark_centered_title_is_kept(self):
        # Oversized + centered but DARK and not repeated/rotated -> kept.
        run = self._run(size=40.0, lightness=0.0)
        analysis = self._analysis(num_pages=1)
        assert _is_watermark_run(run, self._FakePage(), analysis) is False


@pytest.mark.skipif(not _has_fpdf2(), reason="fpdf2 not installed")
class TestEndToEnd:
    """End-to-end conversion behavior with generated PDFs."""

    def test_diagonal_watermark_present_by_default(self):
        result = _convert(_make_diagonal_watermark_pdf())
        text = result.text_content
        assert "revenue report" in text
        collapsed = text.replace("\n", "").replace(" ", "")
        assert WATERMARK_TEXT in collapsed

    def test_diagonal_watermark_removed(self):
        result = _convert(_make_diagonal_watermark_pdf(), pdf_remove_watermarks=True)
        text = result.text_content
        collapsed = text.replace("\n", "").replace(" ", "")
        assert "revenue report" in text, "Body text must survive removal"
        assert WATERMARK_TEXT not in collapsed, f"Watermark not removed: {text!r}"

    def test_boilerplate_present_by_default(self):
        result = _convert(_make_multipage_pdf())
        text = result.text_content
        assert "INTERNAL USE ONLY" in text
        assert "DRAFT COPY" in text
        assert "Copyright ACME Corp" in text
        assert UNIQUE_PHRASES[0] in text

    def test_boilerplate_removed_when_enabled(self):
        result = _convert(_make_multipage_pdf(), pdf_remove_watermarks=True)
        text = result.text_content
        assert "INTERNAL USE ONLY" not in text, f"Header not removed: {text!r}"
        assert "DRAFT COPY" not in text, f"Watermark not removed: {text!r}"
        assert "Copyright ACME Corp" not in text, f"Footer not removed: {text!r}"
        for i, phrase in enumerate(UNIQUE_PHRASES):
            assert phrase in text, f"Body page {i} ({phrase!r}) lost: {text!r}"

    def test_repeated_table_header_is_preserved(self):
        """A dark, body-size header row repeating in-body must NOT be removed."""
        result = _convert(
            _make_repeated_table_header_pdf(), pdf_remove_watermarks=True
        )
        text = result.text_content
        # Repetition alone (single family) is not enough to remove it.
        assert "Product" in text, f"Legit repeated table header lost: {text!r}"
        assert "Quantity" in text
        assert "Price" in text
        # Unique data also survives.
        assert "Widget0" in text

    def test_single_page_keeps_non_watermark_text(self):
        result = _convert(
            _make_multipage_pdf(num_pages=1), pdf_remove_watermarks=True
        )
        text = result.text_content
        # Header is not repeated (1 page) and is dark/body-size -> kept.
        assert "INTERNAL USE ONLY" in text
        assert UNIQUE_PHRASES[0] in text
        # The light oversized horizontal "DRAFT COPY" is not centered enough to
        # be a cosmetic anchor on a single page, but body content is intact.
