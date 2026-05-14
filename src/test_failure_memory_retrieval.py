import logging

from util.failure_memory import FailureMemoryStore

logger = logging.getLogger(__name__)


def main():
    store = FailureMemoryStore()

    seed_document = "這張照片太黑了，整體亮度不足，判定為 Under-exposed。"
    seed_metadata = {
        "profile": "dev",
        "batch_id": "manual_seed",
        "file": "sample_dark.png",
        "release_decision": "NO_GO",
        "model_decision": "Under-exposed",
        "error_code": "ERR_LIGHT_DARK_002",
        "message": "Image is too dark",
        "avg_brightness": 18.2,
        "sharpness": 11.4,
        "latency_ms": 3.2,
        "width": 128,
        "height": 128,
        "pixel_count": 16384,
        "file_size_kb": 6.8,
        "timestamp": "manual-seed",
    }
    store.store_failure_case("manual_seed_dark_case", seed_document, seed_metadata)

    query = "這張照片太黑了"
    result = store.query_similar_failures(query, top_k=3)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    logger.info("Query: %s", query)
    if not documents:
        logger.info("No similar failure cases found.")
        return

    logger.info("Top similar failure cases:")
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        logger.info(
            "%s. file=%s | release=%s | distance=%.4f",
            idx,
            meta.get("file"),
            meta.get("release_decision"),
            dist,
        )
        logger.info("   document=%s", doc)


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
