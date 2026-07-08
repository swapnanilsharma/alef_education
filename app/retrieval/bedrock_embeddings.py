"""Amazon Bedrock embedding client for Titan text embeddings."""

from __future__ import annotations

import json
import logging

import boto3

from app.core.config import EMBEDDING_MODEL_ID, LLM_MAX_TOKENS, LLM_MODEL_ID
logger = logging.getLogger(__name__)


class BedrockEmbeddingService:
    """Client wrapper for generating embeddings through Amazon Bedrock."""

    def __init__(self, model_id: str = EMBEDDING_MODEL_ID) -> None:
        """Initialize the Bedrock runtime client.

        Args:
            model_id: Bedrock model identifier used for embedding generation.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime")
        logger.info("Initialized Bedrock embedding client | model_id=%s", self.model_id)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks.

        Args:
            texts: Text chunks to embed.

        Returns:
            list[list[float]]: Embedding vector per input text, preserving order.
        """
        logger.info("Embedding batch requested | model_id=%s | batch_size=%s", self.model_id, len(texts))
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        """Generate a single embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            list[float]: Embedding vector returned by Bedrock.

        Raises:
            ValueError: If the Bedrock response does not include an embedding vector.
        """
        logger.debug("Embedding single text | model_id=%s | char_count=%s", self.model_id, len(text))
        request_body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body,
        )
        payload = json.loads(response["body"].read())

        if "embedding" in payload:
            logger.debug("Received embedding response | model_id=%s | dimension=%s", self.model_id, len(payload["embedding"]))
            return payload["embedding"]

        embeddings_by_type = payload.get("embeddingsByType", {})
        if isinstance(embeddings_by_type, dict):
            float_embedding = embeddings_by_type.get("float")
            if float_embedding:
                logger.debug("Received float embedding response | model_id=%s | dimension=%s", self.model_id, len(float_embedding))
                return float_embedding

        raise ValueError("Bedrock response did not include an embedding vector.")


class BedrockLLMService:
    """Client wrapper for text generation through Amazon Bedrock (Amazon Nova)."""

    def __init__(self, model_id: str = LLM_MODEL_ID) -> None:
        """Initialize the Bedrock runtime client for text generation.

        Args:
            model_id: Bedrock model identifier used for completion generation.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime")
        logger.info("Initialized Bedrock LLM client | model_id=%s", self.model_id)

    def generate(self, prompt: str) -> str:
        """Send a prompt to Amazon Nova and return the generated text.

        Args:
            prompt: Full prompt string to send to the model.

        Returns:
            str: Generated text from the model.

        Raises:
            ValueError: If the response contains no content.
        """
        logger.debug("LLM generate requested | model_id=%s | prompt_chars=%s", self.model_id, len(prompt))
        request_body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": LLM_MAX_TOKENS},
        })
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body,
        )
        payload = json.loads(response["body"].read())
        try:
            text = payload["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ValueError("Bedrock LLM response contained no text content.") from exc
        logger.debug("LLM response received | model_id=%s", self.model_id)
        return text.strip()
