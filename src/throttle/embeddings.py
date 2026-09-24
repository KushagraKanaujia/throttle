"""Direct ONNX embedding module with frozen contract.

Provides sync embedding functions using pre-exported ONNX weights from HuggingFace Hub.
No torch conversion, single-flight loading with threading.Lock.
"""

import logging
from threading import Lock
from typing import Optional

import importlib.util

# The heavy optional dependencies are only imported when the embedding tier is
# actually used. Importing onnxruntime at CLI startup slowed every command and
# could abort the process at interpreter shutdown (exit 134, libc++
# recursive_mutex) even for commands that never embed anything.
EMBEDDINGS_AVAILABLE = all(
    importlib.util.find_spec(m) is not None
    for m in ("numpy", "onnxruntime", "tokenizers", "huggingface_hub")
)
np = None  # type: ignore
ort = None  # type: ignore
Tokenizer = None  # type: ignore
hf_hub_download = None  # type: ignore


def _import_backends() -> None:
    """Import numpy/onnxruntime/tokenizers/huggingface_hub on first use."""
    global np, ort, Tokenizer, hf_hub_download
    if ort is not None:
        return
    import numpy as _np
    import onnxruntime as _ort
    from tokenizers import Tokenizer as _Tokenizer
    from huggingface_hub import hf_hub_download as _hf_hub_download
    np, ort, Tokenizer, hf_hub_download = _np, _ort, _Tokenizer, _hf_hub_download

logger = logging.getLogger(__name__)

# Module-level singleton state
_embedder: Optional["_DirectEmbedder"] = None
_load_lock = Lock()
_failure_count = 0
# Reason the model could not be loaded (e.g. offline with no cached weights).
# Once set, loading is not retried for the lifetime of the process, so a
# missing model does not trigger a network download attempt on every request.
_load_error: Optional[str] = None

_DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _download(model_id: str, filename: str) -> str:
    """Resolve a model file, preferring the local HF cache over the network."""
    try:
        return hf_hub_download(repo_id=model_id, filename=filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id=model_id, filename=filename)


class _DirectEmbedder:
    """Direct ONNX embedder using pre-exported weights."""

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Install with: pip install throttle-pro[embeddings]"
            )

        _import_backends()
        # Pre-exported ONNX model and tokenizer (local cache first, then Hub)
        model_path = _download(model_id, "onnx/model.onnx")
        tokenizer_path = _download(model_id, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.session = ort.InferenceSession(model_path)

    def embed_one(self, text: str) -> "np.ndarray":
        """Generate L2-normalized 384-dim float32 embedding for one text."""
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        # Mean pooling with attention mask
        token_embeddings = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(np.sum(attention_mask, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        emb = (summed / counts)[0].astype(np.float32)

        # L2 normalize for cosine via dot product
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_batch(self, texts: list[str]) -> "np.ndarray":
        """Generate embeddings for batch of texts.

        Returns (len(texts), 384) array. Failed rows filled with zeros.
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        results = np.zeros((len(texts), 384), dtype=np.float32)

        for i, text in enumerate(texts):
            try:
                results[i] = self.embed_one(text)
            except Exception as e:
                logger.warning(f"Failed to embed text at index {i}: {e}")
                # Row already initialized to zeros
                pass

        return results


def _get_embedder() -> Optional["_DirectEmbedder"]:
    """Get or create the singleton embedder instance. Thread-safe single-flight."""
    global _embedder, _load_error

    if not EMBEDDINGS_AVAILABLE:
        return None

    if _embedder is not None:
        return _embedder
    if _load_error is not None:
        return None

    with _load_lock:
        # Double-check after acquiring lock
        if _embedder is not None:
            return _embedder
        if _load_error is not None:
            return None

        try:
            _embedder = _DirectEmbedder()
            return _embedder
        except Exception as e:
            _load_error = f"{type(e).__name__}: {e}"
            logger.error(
                f"Failed to initialize embedder ({_load_error}). "
                "Semantic embedding tier disabled for this process."
            )
            return None


def is_model_available() -> bool:
    """Return True if the embedding model is installed and loads successfully.

    Triggers the (one-time) model load. Returns False when the optional
    dependencies are missing or the weights cannot be obtained (for example
    no network access and nothing in the local HuggingFace cache).
    """
    return _get_embedder() is not None


def get_load_error() -> Optional[str]:
    """Why the embedding model is unavailable, or None if it is (or may be) usable."""
    if not EMBEDDINGS_AVAILABLE:
        return (
            "Embedding dependencies not installed "
            "(pip install throttle-pro[embeddings])"
        )
    return _load_error


def get_embedding(text: str) -> Optional["np.ndarray"]:
    """Get L2-normalized 384-dim float32 embedding for text.

    Returns None if embeddings unavailable or loading fails.
    """
    global _failure_count

    embedder = _get_embedder()
    if embedder is None:
        return None

    try:
        return embedder.embed_one(text)
    except Exception as e:
        _failure_count += 1
        logger.warning(f"Embedding failed: {e}")
        return None


def get_embeddings(texts: list[str]) -> "np.ndarray":
    """Get embeddings for list of texts.

    Returns (len(texts), 384) array. Failed rows filled with np.zeros((384,)).
    Never returns None - if embeddings unavailable, returns all zeros.
    """
    global _failure_count

    if EMBEDDINGS_AVAILABLE:
        try:
            _import_backends()
        except ImportError:
            pass
    if not EMBEDDINGS_AVAILABLE or np is None:
        # numpy may be missing entirely; return a plain zero matrix if possible
        try:
            import numpy as _np
        except ImportError:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Install with: pip install throttle-pro[embeddings]"
            )
        return _np.zeros((len(texts), 384), dtype=_np.float32)

    embedder = _get_embedder()
    if embedder is None:
        return np.zeros((len(texts), 384), dtype=np.float32)

    try:
        return embedder.embed_batch(texts)
    except Exception as e:
        _failure_count += 1
        logger.error(f"Batch embedding failed: {e}")
        return np.zeros((len(texts), 384), dtype=np.float32)


def get_failure_count() -> int:
    """Get the count of embedding failures since module load."""
    return _failure_count
