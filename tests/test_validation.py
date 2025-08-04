from markthat.utils.validation import (
    END_MARKER,
    START_MARKER,
    has_markers,
    is_valid_markdown,
    validate,
)


def test_basic_markdown_validity():
    md = """```markdown\n[START COPY TEXT]\n# Title\n[END COPY TEXT]\n```"""
    assert is_valid_markdown(md)
    assert has_markers(md)
    result = validate(md)
    assert result.valid, result.message
