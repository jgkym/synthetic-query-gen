import pandas as pd
import unicodedata
import json
from dspy import Example
from typing import List
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter


def load_json(filepath: str) -> dict:
    """Load and return JSON content from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content as a dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    """Normalize a string to Unicode NFC form.

    Args:
        text (str): The string to be normalized.

    Returns:
        str: The normalized string.
    """
    return unicodedata.normalize("NFC", text)


def load_and_process_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame and normalize the 'Source' column.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with the 'Source' column normalized.
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    df["Source"] = df["Source"].apply(normalize_text)  # Normalize 'Source' column
    return df


def map_questions_to_sources(csv_path: str) -> dict:
    """Create a dictionary mapping 'Source' to a list of 'Questions'.

    Args:
        csv_path (str): Path to the CSV file containing 'Source' and 'Question' columns.

    Returns:
        dict: Dictionary mapping each 'Source' to a list of associated 'Questions'.
    """
    df = load_and_process_csv(csv_path)
    return df.groupby("Source")["Question"].apply(list).to_dict()


def corpus_to_examples(json_path: str, chunk_size: int = 512) -> List[Example]:
    """Process the corpus with document splitting and return processed documents.

    Args:
        json_path (str): Path to the JSON file containing the corpus.
        chunk_size (int, optional): Size of the document chunks. Defaults to 512.

    Returns:
        list: List of Example objects with processed document content.
    """
    corpus = load_json(json_path)

    document_ids = [normalize_text(doc["document_id"]) for doc in corpus]
    content = [doc["content"] for doc in corpus]

    processor = CorpusProcessor(document_splitter_fn=llama_index_sentence_splitter)
    documents = processor.process_corpus(
        documents=content, document_ids=document_ids, chunk_size=chunk_size
    )

    return [
        Example(source=doc["document_id"], content=doc["content"]) for doc in documents
    ]
