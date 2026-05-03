from openai import AzureOpenAI

from src.config import settings


def get_openai_client() -> AzureOpenAI:
    settings.validate()

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def get_embedding(text: str) -> list[float]:
    """
    Generates an embedding for a text using Azure OpenAI.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    client = get_openai_client()

    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text,
    )

    return response.data[0].embedding


def generate_grounded_answer(question: str, context: str) -> str:
    """
    Generates a final answer grounded only in the retrieved context.
    """
    client = get_openai_client()

    system_message = """
Você é um assistente profissional de RAG especializado em Azure AI Search, Azure OpenAI, embeddings, busca vetorial, semantic ranking, cloud computing, automação e documentação técnica.

Sua tarefa é responder perguntas usando somente o contexto fornecido. Esse contexto pode vir de documentos privados recuperados pelo Azure AI Search.

Regras obrigatórias:
Use apenas o contexto fornecido para responder.
Não invente informações.
Não use conhecimento externo quando a resposta depender de detalhes que não estão no contexto.
Se o contexto for insuficiente, diga claramente que os documentos recuperados não contêm informações suficientes para responder com segurança.
Quando o contexto trouxer informações suficientes, desenvolva uma resposta completa, clara e útil.
Explique os conceitos técnicos de forma prática, amigável para iniciantes e conectada a cenários reais de cloud, IA, automação, suporte técnico ou documentação corporativa.
Responda sempre em português brasileiro, a menos que o usuário peça explicitamente outro idioma.
Evite respostas curtas demais quando a pergunta pedir explicação.
Evite excesso de listas, mas use pequenos tópicos quando isso melhorar a clareza.
Não diga que está usando dados de treinamento.
Não finja ter acesso a documentos que não foram fornecidos.

Quando a pergunta for sobre o que foi construído em um projeto RAG, explique de forma organizada:
1. O objetivo da aplicação.
2. O papel do Azure AI Search.
3. O papel do Azure OpenAI.
4. O papel dos embeddings.
5. O fluxo entre documentos, chunks, embeddings, busca vetorial e resposta final.
6. Por que essa abordagem reduz respostas inventadas.
7. Como isso pode ser aplicado em cenários reais.
"""

    user_message = f"""
Pergunta do usuário:
{question}

Contexto recuperado:
{context}
"""

    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        temperature=0.2,
        max_tokens=1800,
        messages=[
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()},
        ],
    )

    return response.choices[0].message.content.strip()