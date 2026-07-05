import os
import time
import re
import numpy as np
import pandas as pd

from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from helpers.output import print_article, print_section_title
from helpers.pinecone_helper import ensure_index


BATCH_SIZE = 100

FIXED_SIZE = 60
FIXED_OVERLAP = 15
SEMANTIC_MAX = 60

SEMANTIC_THRESHOLD = 0.7
MIN_CHUNK_SIZE = 50


def fixed_size_chunking(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_SIZE,
        chunk_overlap=FIXED_OVERLAP,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(str(text).strip())


def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = SEMANTIC_THRESHOLD,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    max_words: int = SEMANTIC_MAX,
) -> List[str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " ").strip())
        if s.strip()
    ]

    if len(sentences) < 2:
        return sentences

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]

        current_words = len(" ".join(current_chunk).split())
        next_words = len(next_sentence.split())

        should_split_by_similarity = (
            sim < threshold and current_words >= min_chunk_size
        )

        should_split_by_size = current_words + next_words > max_words

        if should_split_by_similarity or should_split_by_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_arxiv_id(row) -> str:
    if pd.notna(row.get("arxiv_id")):
        return str(row["arxiv_id"])

    if pd.notna(row.get("id")):
        return str(row["id"])

    return "unknown"


def load_top_longest_papers() -> pd.DataFrame:
    df = pd.read_parquet(config.INPUT_PARQUET)

    df["abstract_length"] = (
        df["abstract"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    top_30_longest = df.sort_values(
        by="abstract_length",
        ascending=False,
    ).head(30).copy()

    print(f"Максимальна довжина анотації: {top_30_longest['abstract_length'].max()} слів.")

    return top_30_longest


def process_and_upsert(
    df_papers: pd.DataFrame,
    strategy_name: str,
    pinecone_index,
    model: SentenceTransformer,
):
    print_section_title(f"\nОбробка стратегії: {strategy_name}")

    all_chunks_data = []

    for _, row in df_papers.iterrows():
        abstract = str(row["abstract"])

        if strategy_name == "fixed":
            chunks = fixed_size_chunking(abstract)
        elif strategy_name == "semantic":
            chunks = semantic_chunking(abstract, model=model)
        else:
            raise ValueError("Невідома стратегія чанкінгу")

        for chunk_num, chunk_text in enumerate(chunks, start=1):
            arxiv_id = get_arxiv_id(row)
            model_input = f"{row['title']} [SEP] {chunk_text}"

            all_chunks_data.append(
                {
                    "id": f"{strategy_name}_{arxiv_id}_chunk_{chunk_num}",
                    "model_input": model_input,
                    "metadata": {
                        "arxiv_id": arxiv_id,
                        "title": str(row.get("title", "")),
                        "chunk_text": str(chunk_text)[:1000],
                        "chunk_number": int(chunk_num),
                        "year": int(row.get("year", 0)),
                        "category": str(row.get("category", "")),
                    },
                }
            )

    print(f"Кількість чанків: {len(all_chunks_data)}")

    for i in tqdm(
        range(0, len(all_chunks_data), BATCH_SIZE),
        desc=f"Upsert {strategy_name}",
    ):
        batch = all_chunks_data[i:i + BATCH_SIZE]
        texts = [item["model_input"] for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        vectors = [
            {
                "id": item["id"],
                "values": embedding.tolist(),
                "metadata": item["metadata"],
            }
            for item, embedding in zip(batch, embeddings)
        ]

        pinecone_index.upsert(vectors=vectors)


def search_chunks(
    query_text: str,
    model: SentenceTransformer,
    indexes: list[tuple[str, object]],
):
    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
    ).tolist()

    print_section_title(f"РЕЗУЛЬТАТИ ПОШУКУ ДЛЯ ЗАПИТУ: {query_text}")

    for name, client in indexes:
        print(f"\n--- Стратегія: {name} ---")

        results = client.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
        )

        for rank, match in enumerate(results["matches"], start=1):
            metadata = match["metadata"]

            print_article(
                rank=rank,
                score=match["score"],
                title=metadata.get("title"),
                category=metadata.get("category"),
                year=metadata.get("year"),
                abstract=metadata.get("chunk_text", "")
            )

            print("-" * 60)


def main():
    load_dotenv()

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    model = SentenceTransformer(config.MODEL_NAME, trust_remote_code=True)

    top_30_longest = load_top_longest_papers()

    idx_fixed_client = ensure_index(pc, config.INDEX_FIXED)
    idx_semantic_client = ensure_index(pc, config.INDEX_SEMANTIC)

    process_and_upsert(
        df_papers=top_30_longest,
        strategy_name="fixed",
        pinecone_index=idx_fixed_client,
        model=model,
    )

    process_and_upsert(
        df_papers=top_30_longest,
        strategy_name="semantic",
        pinecone_index=idx_semantic_client,
        model=model,
    )

    indexes = [
        ("FIXED-SIZE CHUNKS", idx_fixed_client),
        ("SEMANTIC CHUNKS", idx_semantic_client),
    ]

    test_queries = [
        "numerical algorithms and simulation methods",
        "experimental physics data analysis",
        "graph neural networks for representation learning",
    ]

    for query in test_queries:
        search_chunks(
            query_text=query,
            model=model,
            indexes=indexes,
        )


if __name__ == "__main__":
    main()