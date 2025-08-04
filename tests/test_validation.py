from markthat_new.utils.validation import (
    START_MARKER,
    END_MARKER,
    is_valid_markdown,
    has_markers,
    validate,
)


def test_basic_markdown_validity():
    md = """```markdown\n[START COPY TEXT]\n# Title\n[END COPY TEXT]\n```"""
    assert is_valid_markdown(md)
    assert has_markers(md)
    result = validate(md)
    assert result.valid, result.message
