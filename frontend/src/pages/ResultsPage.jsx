import { useLocation, useNavigate } from "react-router-dom";

import Navbar from "../components/Navbar";

import RiskScoreCard from "../components/RiskScoreCard";

import SourceDocuments from "../components/SourceDocuments";
import DetailedAnalysis from "../components/DetailedAnalysis";

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

       <div className="results-actions">

  <div
    className="back-link"
    onClick={() => navigate("/")}
  >
    ← Back to Contracts
  </div>

  <button
    className="new-analysis-btn"
    onClick={() => navigate("/")}
  >
    + Analyze Another Contract
  </button>

</div>

        <div className="results-header">

          <h1>
            Contract Analysis Results
          </h1>

         

        </div>

        <div className="results-grid">

          <div className="left-column">

            <RiskScoreCard
    score={result.risk_score}
    level={result.risk_level}
    confidence={result.confidence}
/>

           

            <SourceDocuments
              sources={result.sources}
            />

          </div>

          <div className="right-column">

            <AISummaryCard
              answer={result.answer}
            />

           {result.recommendations &&
 result.recommendations.length > 0 && (

    <DetailedAnalysis
        findings={result.recommendations}
    />

)}

          

          </div>

        </div>

      </main>

    </>
  );
}

export default ResultsPage;