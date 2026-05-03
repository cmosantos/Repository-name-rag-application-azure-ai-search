import sys

from azure.core.exceptions import HttpResponseError
from azure.search.documents.models import VectorizedQuery

from src.config import settings
from src.embeddings import generate_grounded_answer, get_embedding
from src.search_index import SEMANTIC_CONFIG_NAME, VECTOR_FIELD_NAME, get_search_client


def retrieve_context(question: str) -> tuple[str, list[dict]]:
    """
    Retrieves relevant document chunks using hybrid search:
    text search + vector search + semantic ranking when available.
    """
    search_client = get_search_client()
    query_vector = get_embedding(question)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=settings.top_k,
        fields=VECTOR_FIELD_NAME,
    )

    select_fields = ["title", "content", "source", "chunk_id"]

    try:
        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG_NAME,
            query_caption="extractive",
            query_answer="extractive",
            top=settings.top_k,
            select=select_fields,
        )
    except HttpResponseError as error:
        print("Semantic ranking query failed. Falling back to hybrid vector search without semantic ranking.")
        print(f"Reason: {error.message}")

        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            top=settings.top_k,
            select=select_fields,
        )

    sources = []

    for result in results:
        item = dict(result)

        sources.append(
            {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "chunk_id": item.get("chunk_id", ""),
                "content": item.get("content", ""),
                "score": item.get("@search.score", None),
                "reranker_score": item.get("@search.reranker_score", None),
            }
        )

    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"Source {index}: {source['title']} | File: {source['source']} | Chunk: {source['chunk_id']}\n"
            f"{source['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context, sources


def ask(question: str) -> None:
    settings.validate()

    print(f"Question: {question}")
    print("Retrieving context from Azure AI Search...")

    context, sources = retrieve_context(question)

    if not sources:
        print("No relevant documents were found in Azure AI Search.")
        return

    print(f"Retrieved sources: {len(sources)}")
    print("Generating grounded answer with Azure OpenAI...")

    answer = generate_grounded_answer(question, context)

    print("\nAnswer")
    print("------")
    print(answer)

    print("\nSources")
    print("-------")

    for index, source in enumerate(sources, start=1):
        print(
            f"{index}. {source['title']} | {source['source']} | Chunk {source['chunk_id']} "
            f"| Score: {source['score']} | Reranker: {source['reranker_score']}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask "Your question here"')
        return

    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print("Please provide a valid question.")
        return

    try:
        ask(question)

    except Exception as error:
        print("Question answering failed.")
        print(str(error))
        raise


if __name__ == "__main__":
    main()
