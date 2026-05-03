import uuid

from azure.core.exceptions import HttpResponseError

from src.chunking import load_text_documents, split_text
from src.config import settings
from src.embeddings import get_embedding
from src.search_index import create_or_update_index, get_search_client


def build_document_id(source: str, chunk_id: int) -> str:
    """
    Creates a stable unique ID for each document chunk.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}-{chunk_id}"))


def build_index_documents() -> list[dict]:
    """
    Reads local documents, chunks them, generates embeddings, and prepares them for upload.
    """
    source_documents = load_text_documents(settings.data_dir)

    if not source_documents:
        raise FileNotFoundError(
            f"No .txt files found in {settings.data_dir}. Add at least one text file to the data folder."
        )

    index_documents = []

    for document in source_documents:
        chunks = split_text(document["content"])

        print(f"Processing {document['source']} with {len(chunks)} chunks.")

        for chunk_id, chunk in enumerate(chunks):
            print(f"Generating embedding for {document['source']} chunk {chunk_id}...")

            embedding = get_embedding(chunk)

            index_documents.append(
                {
                    "id": build_document_id(document["source"], chunk_id),
                    "title": document["title"],
                    "content": chunk,
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "content_vector": embedding,
                }
            )

    return index_documents


def upload_documents(documents: list[dict]) -> None:
    """
    Uploads documents to Azure AI Search.
    """
    search_client = get_search_client()

    result = search_client.upload_documents(documents=documents)

    succeeded = sum(1 for item in result if item.succeeded)
    failed = len(result) - succeeded

    print(f"Uploaded documents: {succeeded}")
    print(f"Failed uploads: {failed}")

    for item in result:
        if not item.succeeded:
            print(f"Failed document key: {item.key}. Error: {item.error_message}")


def main() -> None:
    try:
        settings.validate()

        print("Creating or updating Azure AI Search index...")
        create_or_update_index()

        print("Preparing documents...")
        documents = build_index_documents()

        print("Uploading documents to Azure AI Search...")
        upload_documents(documents)

        print("Ingestion completed successfully.")

    except HttpResponseError as error:
        print("Azure Search request failed.")
        print(error.message)
        raise

    except Exception as error:
        print("Ingestion failed.")
        print(str(error))
        raise


if __name__ == "__main__":
    main()
