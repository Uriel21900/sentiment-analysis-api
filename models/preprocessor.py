"""
Text preprocessor for sentiment analysis.
"""
import re


def preprocess_text(text: str) -> str:
    """
    Preprocess text for sentiment analysis.

    Args:
        text: Input text to preprocess

    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\\w\\s\\.,!?\'\"]', '', text)

    # Normalize whitespace
    text = re.sub(r'\\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    # Test the preprocessor
    test_text = "Check out this http://example.com product @username! #amazing!!!"
    result = preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {result}")
