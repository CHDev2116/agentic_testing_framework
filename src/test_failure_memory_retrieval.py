from util.failure_memory import FailureMemoryStore


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

    print(f"Query: {query}")
    if not documents:
        print("No similar failure cases found.")
        return

    print("Top similar failure cases:")
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(
            f"{idx}. file={meta.get('file')} | release={meta.get('release_decision')} | "
            f"distance={dist:.4f}"
        )
        print(f"   document={doc}")


if __name__ == "__main__":
    main()
