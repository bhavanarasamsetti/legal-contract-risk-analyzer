from app.risk_scoring import StructuredRiskScorer


def main():
    scorer = StructuredRiskScorer(top_k=5)

    question = "What are the highest legal risks in this contract?"

    result = scorer.score(question)

    print("\nQUESTION")
    print(result.question)

    print("\nMODEL")
    print(result.model)

    print("\nRISK ASSESSMENTS")
    print("-" * 80)

    for i, clause in enumerate(result.clauses, start=1):
        print(f"\nClause {i}")
        print(f"Document       : {clause.document_name}")
        print(f"Pages          : {clause.page_numbers}")
        print(f"Title          : {clause.clause_title}")
        print(f"Risk Score     : {clause.risk_score}/10")
        print(f"Risk Level     : {clause.risk_level}")
        print(f"Explanation    : {clause.explanation}")
        print(f"Recommendation : {clause.recommendation}")

    print("\nTOKEN USAGE")
    print(result.total_tokens)


if __name__ == "__main__":
    main()