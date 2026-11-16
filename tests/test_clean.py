"""Unit tests for streaming text normalization, quality filtering, and deduplication."""

from __future__ import annotations

import unicodedata

from src.data.clean import (
    clean_and_dedup_stream,
    dedup_exact,
    filter_document,
    filter_documents,
    normalize_text,
)


def test_normalize_nfc_and_invisible_controls() -> None:
    # Decomposed NFD Vietnamese ("tiếng Việt") + ZWSP + BOM + CRLF
    nfd_text = "\ufeffTi\u00ea\u0301ng\u200b   Vi\u00ea\u0323t\r\n\r\n\r\nĐoa\u0323n hai.\x00\tCột"
    normalized = normalize_text(nfd_text)

    assert unicodedata.is_normalized("NFC", normalized)
    assert "\u200b" not in normalized
    assert "\ufeff" not in normalized
    assert "\x00" not in normalized
    assert "\r" not in normalized
    assert normalized == "Tiếng Việt\n\nĐoạn hai. Cột"


def test_normalize_strips_trailing_wiki_sections_only_at_heading() -> None:
    # "Xem thêm" inside a normal sentence must NOT truncate, while heading at line start MUST truncate.
    text = (
        "Hà Nội là thủ đô của Việt Nam. Xem thêm thông tin về lịch sử trong bài chính.\n\n"
        "Thành phố nằm bên bờ sông Hồng.\n\n"
        "Xem thêm\n"
        "* Lịch sử Hà Nội\n"
        "Tham khảo\n"
        "1. Sách A"
    )
    cleaned = normalize_text(text)
    assert "Xem thêm thông tin về lịch sử" in cleaned
    assert "Thành phố nằm bên bờ sông Hồng." in cleaned
    assert "Lịch sử Hà Nội" not in cleaned
    assert "Sách A" not in cleaned


def test_normalize_cleans_markup_and_preserves_endoftext_literal() -> None:
    raw = (
        "Văn bản mở đầu<ref name=\"a\">nguồn</ref><!-- hidden --> có thẻ <br/>"
        "và chuỗi <|endoftext|> nguyên vẹn.\n"
        "{| class=\"wikitable\"\n"
        "|- \n"
        "| ô bảng hỏng\n"
        "|}\n"
        "Đoạn kết hợp lệ."
    )
    cleaned = normalize_text(raw)
    assert "<ref" not in cleaned
    assert "hidden" not in cleaned
    assert "<|endoftext|>" in cleaned
    assert "wikitable" not in cleaned
    assert "ô bảng hỏng" not in cleaned
    assert "Đoạn kết hợp lệ." in cleaned


def test_filter_document_thresholds() -> None:
    good = (
        "Trường Đại học Công nghệ là một trường đại học thành viên của Đại học Quốc gia Hà Nội, "
        "được thành lập năm 2004 tại quận Cầu Giấy, Hà Nội. Nhà trường đào tạo nhiều ngành kỹ thuật "
        "và công nghệ mũi nhọn như công nghệ thông tin, trí tuệ nhân tạo, điện tử viễn thông và cơ kỹ thuật."
    )
    assert len(good) >= 250
    assert filter_document(good, min_chars=250, min_valid_char_ratio=0.85)

    stub = "Bài viết quá ngắn."
    assert not filter_document(stub, min_chars=250)

    non_latin = ("日本語のテスト文章です。" * 40)
    assert not filter_document(non_latin, min_chars=100, min_valid_char_ratio=0.85)

    numbers_only = ("12345 67890 11223 44556 " * 20).strip()
    assert not filter_document(numbers_only, min_chars=250, min_valid_char_ratio=0.85, min_letter_ratio=0.45)


def test_dedup_exact_and_stream_pipeline() -> None:
    base = (
        "Hồ Hoàn Kiếm là một hồ nước ngọt tự nhiên nằm ở trung tâm thành phố Hà Nội, "
        "gắn liền với truyền thuyết vua Lê Thái Tổ trả gươm thần cho Rùa Vàng. "
        "Xung quanh hồ có nhiều di tích lịch sử văn hóa nổi tiếng như đền Ngọc Sơn, cầu Thê Húc, "
        "tháp Bút và đài Nghiên nằm ngay bên bờ hồ."
    )
    nfd_duplicate = unicodedata.normalize("NFD", base) + "\n\nTham khảo\n1. Link"
    raw_docs = [
        base,
        nfd_duplicate,  # Should normalize to identical text and be deduped!
        "Ngắn quá",
        base + " Đoạn bổ sung thêm đủ dài để khác bản gốc về mặt nội dung và kiểm tra giữ lại.",
    ]
    out = list(clean_and_dedup_stream(raw_docs, min_chars=250))
    assert len(out) == 2
    assert list(dedup_exact(["a", "b", "a", "c", "b"])) == ["a", "b", "c"]
    assert list(filter_documents([base, "short"], min_chars=250)) == [base]
