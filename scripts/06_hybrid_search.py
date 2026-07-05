import os
import re
from typing import List, Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import config
from helpers.output import print_article, print_section_title

from helpers.pinecone_helper import ensure_index

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


def prepare_documents(df: pd.DataFrame) -> List[str]:
    return (
            df["title"].fillna("").astype(str)
            + " "
            + df["abstract"].fillna("").astype(str)
    ).tolist()


def build_bm25_index(documents: List[str]) -> BM25Okapi:
    tokenized_docs = [tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)


def bm25_search(
        query: str,
        bm25: BM25Okapi,
        df: pd.DataFrame,
        top_k: int = CANDIDATE_K,
) -> List[Dict]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]

        results.append(
            {
                "id": f"paper_{idx}",
                "rank": rank,
                "score": float(scores[idx]),
                "title": row["title"],
                "category": row.get("category"),
                "year": row.get("year"),
                "abstract": row.get("abstract", ""),
            }
        )

    return results


def vector_search(
        query: str,
        model: SentenceTransformer,
        index,
        top_k: int = CANDIDATE_K,
) -> List[Dict]:
    query_vector = model.encode(query).tolist()

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for rank, match in enumerate(response["matches"], start=1):
        metadata = match.get("metadata", {})

        results.append(
            {
                "id": match["id"],
                "rank": rank,
                "score": float(match["score"]),
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "year": metadata.get("year"),
                "abstract": metadata.get("abstract", ""),
            }
        )

    return results


def reciprocal_rank_fusion(
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int = TOP_K,
        rrf_k: int = RRF_K,
) -> List[Dict]:
    fused_scores = {}
    documents = {}

    for result_list in [bm25_results, vector_results]:
        for result in result_list:
            doc_id = result["id"]
            rank = result["rank"]

            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (rrf_k + rank)

            if doc_id not in documents:
                documents[doc_id] = result

    ranked_doc_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:top_k]

    hybrid_results = []
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        result = documents[doc_id].copy()
        result["rank"] = rank
        result["rrf_score"] = fused_scores[doc_id]
        hybrid_results.append(result)

    return hybrid_results


def hybrid_search(
        query: str,
        bm25: BM25Okapi,
        df: pd.DataFrame,
        model: SentenceTransformer,
        index,
        candidate_k: int = CANDIDATE_K,
        top_k: int = TOP_K,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    bm25_results = bm25_search(query, bm25, df, top_k=candidate_k)
    vector_results = vector_search(query, model, index, top_k=candidate_k)

    hybrid_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
        top_k=top_k,
    )

    return bm25_results[:top_k], vector_results[:top_k], hybrid_results


def print_results(title: str, results: List[Dict], use_rrf: bool = False) -> None:
    print_section_title(title)

    for result in results:
        score = result["rrf_score"] if use_rrf else result["score"]

        print_article(
            rank=result["rank"],
            score=score,
            title=result.get("title", ""),
            category=result.get("category"),
            year=result.get("year"),
            abstract=result.get("abstract", ""),
            score_label="RRF score" if use_rrf else "Score",
        )


def run_demo_queries(
        queries: List[str],
        bm25: BM25Okapi,
        df: pd.DataFrame,
        model: SentenceTransformer,
        index,
) -> None:
    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        bm25_results, vector_results, hybrid_results = hybrid_search(
            query=query,
            bm25=bm25,
            df=df,
            model=model,
            index=index,
        )

        print_results("Top-5 BM25 results", bm25_results)
        print_results("Top-5 vector search results", vector_results)
        print_results("Top-5 hybrid RRF results", hybrid_results, use_rrf=True)


def main() -> None:
    load_dotenv()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = ensure_index(pc, config.INDEX_NAME)

    model = SentenceTransformer(config.MODEL_NAME)

    df = pd.read_parquet(config.INPUT_PARQUET).reset_index(drop=True)

    documents = prepare_documents(df)
    bm25 = build_bm25_index(documents)

    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]

    run_demo_queries(
        queries=queries,
        bm25=bm25,
        df=df,
        model=model,
        index=index,
    )


if __name__ == "__main__":
    main()
