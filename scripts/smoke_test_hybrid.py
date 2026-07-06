from app.hybrid_retriever import HybridRetriever


def main() -> None:
    retriever = HybridRetriever()

    query = "What does Section 3.2 require?"

    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query)

    print(f"Retrieved {len(results)} results.\n")

    for i, result in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result {i}")
        print("=" * 70)
        print(f"Chunk ID      : {result['chunk_id']}")
        print(f"Document      : {result['document_name']}")
        print(f"Section       : {result['section']}")
        print(f"Section Title : {result['section_title']}")
        print(f"Pages         : {result['pages']}")
        print(f"Score         : {result['score']:.4f}")
        print(f"Preview:\n{result['chunk_text'][:300]}")
        print()


if __name__ == "__main__":
    main()