import logging
import os
from importlib import import_module
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


class FailureMemoryStore:
    def __init__(
        self,
        persist_dir="results/failure_memory_db",
        collection_name="quality_failures",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        chromadb = import_module("chromadb")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.encoder = None
        hashing_vectorizer_cls = getattr(
            import_module("sklearn.feature_extraction.text"),
            "HashingVectorizer",
        )
        self.fallback_encoder = hashing_vectorizer_cls(
            n_features=384,
            alternate_sign=False,
            norm="l2",
        )
        try:
            sentence_transformer_cls = getattr(
                import_module("sentence_transformers"),
                "SentenceTransformer",
            )
            self.encoder = sentence_transformer_cls(embedding_model)
        except Exception as e:
            logger.warning(
                "Could not load sentence-transformers model (%s). Using local fallback embeddings.",
                e,
            )

    def build_document(self, file_name, decision_payload):
        reason = decision_payload.get("msg") or decision_payload.get("decision") or "Unknown issue"
        return f"Image {file_name} failed quality check: {reason}"

    def build_metadata(self, profile, batch_id, file_name, row_data, release_decision):
        decision_payload = row_data.get("decision", {}) or {}
        image_meta = row_data.get("image_meta", {}) or {}
        metrics = row_data.get("metrics", {}) or {}
        return {
            "profile": profile,
            "batch_id": str(batch_id),
            "file": file_name,
            "release_decision": release_decision,
            "model_decision": str(decision_payload.get("decision", "UNKNOWN")),
            "error_code": str(decision_payload.get("code", "UNKNOWN")),
            "message": str(decision_payload.get("msg", "")),
            "avg_brightness": float(metrics.get("avg_brightness", metrics.get("brightness", 0.0))),
            "sharpness": float(metrics.get("sharpness", 0.0)),
            "latency_ms": float(row_data.get("latency_ms", 0.0)),
            "width": int(image_meta.get("width", 0)),
            "height": int(image_meta.get("height", 0)),
            "pixel_count": int(image_meta.get("pixel_count", 0)),
            "file_size_kb": float(image_meta.get("file_size_kb", 0.0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def store_failure_case(self, failure_id, document, metadata):
        embedding = self._encode_texts([document])[0]
        self.collection.upsert(
            ids=[failure_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query_similar_failures(self, query_text, top_k=3):
        query_embedding = self._encode_texts([query_text])[0]
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def _encode_texts(self, texts):
        if self.encoder is not None:
            return [vec.tolist() for vec in self.encoder.encode(texts)]
        sparse_vectors = self.fallback_encoder.transform(texts)
        return sparse_vectors.toarray().tolist()
