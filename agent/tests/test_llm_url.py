from agent.llm.url import normalize_base_url


def test_normalize_base_url_strips_whitespace():
    assert normalize_base_url("  https://api.groq.com/openai/v1  ") == "https://api.groq.com/openai/v1"


def test_normalize_base_url_strips_backticks():
    assert normalize_base_url(" `https://api.groq.com/openai/v1` ") == "https://api.groq.com/openai/v1"


def test_normalize_base_url_strips_repeated_backticks():
    assert normalize_base_url(" ``https://api.groq.com/openai/v1`` ") == "https://api.groq.com/openai/v1"


def test_normalize_base_url_strips_double_quotes():
    assert normalize_base_url(' "https://api.groq.com/openai/v1" ') == "https://api.groq.com/openai/v1"
