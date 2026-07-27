"""Text embedding for semantic episode recall.

Wraps ``BAAI/bge-small-en-v1.5`` (384-dim) via fastembed — an ONNX runtime that
avoids the multi-GB torch dependency; the quantized model is ~130MB and stays
resident after first use. The model is loaded lazily on the first ``embed`` so
importing this module (and running tests with a fake embedder) stays cheap.

NOTE: the first ``embed`` downloads the model from HuggingFace. In an offline
environment, pre-populate the fastembed cache or set ``FASTEMBED_CACHE_PATH`` /
``HF_HOME`` to a directory that already contains the model. Where huggingface.co
is blocked, set ``HF_ENDPOINT=https://hf-mirror.com`` to fetch via a mirror; the
Xet backend (``cas-server.xethub.hf.co``) is not mirrored, so this module also
defaults ``HF_HUB_DISABLE_XET=1`` to force the classic (mirror-proxied) LFS path.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


class Embedder:
    """Lazy bge-small embedder returning L2-normalized 384-dim float vectors."""

    def __init__(self, model_name: str = MODEL_NAME, dim: int = EMBED_DIM) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            # Classic LFS download works via both direct HF and mirrors; the Xet
            # storage backend does not, so default it off unless overridden.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        import numpy as np

        vectors = list(self._model.embed(list(texts)))
        return [np.asarray(v, dtype="float32").tolist() for v in vectors]
