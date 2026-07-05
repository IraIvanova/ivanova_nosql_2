import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

import config
from helpers.output import print_article, print_section_title
from helpers.pinecone_helper import ensure_index

load_dotenv()

TOP_K = 5

DATASET_PATH = "data/arxiv_subset.parquet"
EMBEDDINGS_PATH = "embeddings/embeddings.npy"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = ensure_index(pc, config.INDEX_NAME)
model = SentenceTransformer(config.MODEL_NAME)

df = pd.read_parquet(config.INPUT_PARQUET)


def search_pinecone(query: str, filter_query=None):
    query_embedding =  model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    return index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_query,
    )

def print_results(title: str, results):
    print_section_title(title)

    for rank, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        print_article(
            rank=rank,
            score=match["score"],
            title=metadata.get("title"),
            category=metadata.get("category"),
            year=metadata.get("year"),
            abstract=metadata.get("abstract"),
        )


def print_local_results(title: str, indices, scores):
    print_section_title(title)

    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]

        print_article(
            rank=rank,
            score=scores[idx],
            title=row["title"],
            category=row["category"],
            year=row["year"],
            abstract=str(row["abstract"]),
        )

def local_similarity_search(query: str):
    embeddings = np.load(config.INPUT_EMBEDDINGS)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    cosine_scores = embeddings @ query_embedding
    # Для нормалізованих ембеддингів cosine similarity = dot product
    dot_scores = cosine_scores
    l2_distance = np.linalg.norm(embeddings - query_embedding, axis=1)

    cosine_top = np.argsort(cosine_scores)[::-1][:TOP_K]
    dot_top = np.argsort(dot_scores)[::-1][:TOP_K]
    l2_top = np.argsort(l2_distance)[:TOP_K]

    print_local_results(
        "Локальний пошук: Cosine similarity",
        cosine_top,
        cosine_scores,
    )

    print_local_results(
        "Локальний пошук: Dot product",
        dot_top,
        dot_scores,
    )

    print_local_results(
        "Локальний пошук: L2-distance",
        l2_top,
        -l2_distance,
    )


def main():
    query = "teaching machines to recognize objects in pictures"

    print_results(
        "Cемантичний пошук у Pinecone",
        search_pinecone(query),
    )

    recent_articles_result = search_pinecone(
            "reinforcement learning",
            {
                "category": {"$eq": "cs.LG"},
                "year": {"$gte": 2021},
            }
        )

    print_results(
        "Статті по reinforcement learning за останні 5 років і категорія cs.LG",
        recent_articles_result
    )

    older_articles_result = search_pinecone(
            "reinforcement learning",
            {
                "year": {"$lt": 2015},
            },
        )

    print_results(
        "Більш старі статті (до 2015 року), будь-яка категорія",
        older_articles_result
    )

    print(
        "\nПорівняння фільтрів:"
        "\n- У першому випадку результати обмежені категорією cs.LG "
        "та новішими статтями, тому видача більш сфокусована на сучасному machine learning."
        "\n- У другому випадку обмеження тільки за роком, тому можуть зʼявлятися "
        "старіші роботи з різних категорій."
    )

    local_similarity_search(query)

    print(
        "\nПорівняння метрик:"
        "\n- Оскільки ембеддинги нормалізовані, cosine similarity і dot product "
        "зазвичай дають однаковий або майже однаковий топ результатів."
        "\n- L2-distance також часто дає схожі результати для нормалізованих векторів, "
        "але сортування йде навпаки: чим менша відстань, тим релевантніший результат."
    )


if __name__ == "__main__":
    main()