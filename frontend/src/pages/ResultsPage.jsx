import { useLocation, useNavigate } from "react-router-dom";

import Navbar from "../components/Navbar";

import RiskScoreCard from "../components/RiskScoreCard";
import QuestionInfo from "../components/QuestionInfo";
import SourceDocuments from "../components/SourceDocuments";
import DetailedAnalysis from "../components/DetailedAnalysis";
import ReferencedClauses from "../components/ReferencedClauses";
import AISummaryCard from "../components/AISummaryCard";

import "../styles/results.css";

function ResultsPage() {

  const location = useLocation();

  const navigate = useNavigate();

  const result = location.state;

  if (!result) {

    navigate("/");

    return null;

  }

  return (
    <>
      <Navbar />

      <main className="results-page">

        <div
          className="back-link"
          onClick={() => navigate("/")}
        >
          ← Back to Contracts
        </div>

        <div className="results-header">

          <h1>
            Contract Analysis Results
          </h1>

          <p>

            Analysis of

            <strong>

              {" "}

              {result.contract_name ||
                "Employment Agreement"}

            </strong>

          </p>

        </div>

        <div className="results-grid">

          <div className="left-column">

            <RiskScoreCard />

            <QuestionInfo
              question={result.question}
            />

            <SourceDocuments
              sources={result.sources}
            />

          </div>

          <div className="right-column">

            <AISummaryCard
              answer={result.answer}
            />

            <DetailedAnalysis />

            <ReferencedClauses
              sources={result.sources}
            />

          </div>

        </div>

      </main>

    </>
  );
}

export default ResultsPage;