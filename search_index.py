from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from src.config import settings


VECTOR_FIELD_NAME = "content_vector"
VECTOR_PROFILE_NAME = "vector-profile"
VECTOR_ALGORITHM_NAME = "hnsw-config"
SEMANTIC_CONFIG_NAME = "semantic-config"


def get_index_client() -> SearchIndexClient:
    settings.validate()

    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def get_search_client() -> SearchClient:
    settings.validate()

    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def create_or_update_index() -> None:
    """
    Creates or updates the Azure AI Search index used by the RAG application.
    """
    index_client = get_index_client()

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.Int32,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name=VECTOR_FIELD_NAME,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=settings.embedding_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM_NAME,
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name=SEMANTIC_CONFIG_NAME,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[
                SemanticField(field_name="content"),
            ],
        ),
    )

    semantic_search = SemanticSearch(
        configurations=[
            semantic_config,
        ]
    )

    index = SearchIndex(
        name=settings.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    result = index_client.create_or_update_index(index)

    print(f"Search index ready: {result.name}")
