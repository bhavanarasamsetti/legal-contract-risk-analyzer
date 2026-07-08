from app.analyzer import RiskAnalyzer

def main():
    analyzer = RiskAnalyzer(top_k=5)

    question = "What are the data breach notification obligations?"

    result = analyzer.analyze(question)

    print("\nQUESTION")
    print(result["question"])

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(
            f"{source['document_name']} | "
            f"Pages: {source['pages']} | "
            f"Section: {source['section']}"
        )

    print("\nTOKEN USAGE")
    print(result["total_tokens"])


if __name__ == "__main__":
    main()