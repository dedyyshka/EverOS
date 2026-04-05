"""
Google Gemini Vectorize Service Implementation

Использует google.genai SDK (async) для получения эмбеддингов.
Формат ответа Gemini (result.embeddings[].values) отличается от OpenAI
(response.data[].embedding), поэтому реализуем VectorizeServiceInterface напрямую,
без наследования от BaseVectorizeService.
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from google import genai

from agentic_layer.vectorize_interface import (
    UsageInfo,
    VectorizeError,
    VectorizeServiceInterface,
)

logger = logging.getLogger(__name__)


@dataclass
class GeminiVectorizeConfig:
    """Конфигурация для Gemini embedding сервиса."""

    api_key: str = ""
    model: str = "gemini-embedding-2-preview"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    dimensions: int = 768

    # Не используются Gemini, но нужны для совместимости с фабрикой
    base_url: str = ""
    encoding_format: str = "float"


class GeminiVectorizeService(VectorizeServiceInterface):
    """Gemini embedding сервис через google.genai SDK (async)."""

    def __init__(self, config: GeminiVectorizeConfig | None = None):
        if config is None:
            config = GeminiVectorizeConfig()
        self.config = config
        self._client: genai.Client | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        logger.info(
            "Initialized GeminiVectorizeService | model=%s | dimensions=%d",
            config.model,
            config.dimensions,
        )

    def _ensure_client(self) -> genai.Client:
        """Ленивая инициализация клиента."""
        if self._client is None:
            if not self.config.api_key:
                raise VectorizeError("Gemini API key is not configured.")
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    async def _make_request(
        self,
        texts: list[str],
        instruction: str | None = None,
        is_query: bool = False,
    ) -> list[np.ndarray]:
        """Запрос эмбеддингов через Gemini async API с ретраями."""
        client = self._ensure_client()
        if not self.config.model:
            raise VectorizeError("Embedding model is not configured.")

        # Gemini не поддерживает instruction в формате Instruct/Query,
        # но поддерживает task_type для оптимизации
        embed_config: dict = {}
        if self.config.dimensions and self.config.dimensions > 0:
            embed_config["output_dimensionality"] = self.config.dimensions

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    response = await client.aio.models.embed_content(
                        model=self.config.model,
                        contents=texts,
                        config=embed_config if embed_config else None,
                    )

                    if not response.embeddings:
                        raise VectorizeError("Gemini API: пустой ответ (нет embeddings)")

                    return [
                        np.array(emb.values, dtype=np.float32)
                        for emb in response.embeddings
                    ]

                except VectorizeError:
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        "GeminiVectorizeService API error (attempt %d/%d): %s",
                        attempt + 1,
                        self.config.max_retries,
                        error_msg,
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise VectorizeError(
                        f"Gemini API request failed after {self.config.max_retries} attempts: {error_msg}"
                    )

        # unreachable, но для типизации
        raise VectorizeError("Gemini API: unexpected control flow")

    async def get_embedding(
        self, text: str, instruction: str | None = None, is_query: bool = False
    ) -> np.ndarray:
        """Эмбеддинг одного текста."""
        embeddings = await self._make_request([text], instruction, is_query)
        return embeddings[0]

    async def get_embedding_with_usage(
        self, text: str, instruction: str | None = None, is_query: bool = False
    ) -> tuple[np.ndarray, UsageInfo | None]:
        """Эмбеддинг с usage info. Gemini не возвращает usage — всегда None."""
        embedding = await self.get_embedding(text, instruction, is_query)
        return embedding, None

    async def get_embeddings(
        self,
        texts: list[str],
        instruction: str | None = None,
        is_query: bool = False,
    ) -> list[np.ndarray]:
        """Эмбеддинги для списка текстов с батчингом."""
        if not texts:
            return []

        if len(texts) <= self.config.batch_size:
            return await self._make_request(texts, instruction, is_query)

        embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            result = await self._make_request(batch, instruction, is_query)
            embeddings.extend(result)
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)
        return embeddings

    async def get_embeddings_batch(
        self,
        text_batches: list[list[str]],
        instruction: str | None = None,
        is_query: bool = False,
    ) -> list[list[np.ndarray]]:
        """Эмбеддинги для нескольких батчей параллельно."""
        tasks = [
            self.get_embeddings(batch, instruction, is_query) for batch in text_batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings_batches: list[list[np.ndarray]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Error processing batch %d: %s", i, result)
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)
        return embeddings_batches

    def get_model_name(self) -> str:
        """Имя текущей модели."""
        return self.config.model

    async def close(self):
        """Освобождение ресурсов."""
        self._client = None
