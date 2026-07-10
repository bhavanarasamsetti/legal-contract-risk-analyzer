import { AlertTriangle, BarChart3 } from "lucide-react";

function RiskScoreCard({
  score = 0,
  level = "Low",
  confidence,
}) {

 const percentage = score;

  return (
    <div className="risk-card">

      <div className="risk-heading">

        <BarChart3 size={18} />

        <span>OVERALL RISK SCORE</span>

      </div>

      <div
        className="risk-circle"
        style={{
          background: `conic-gradient(
            #C58A25 ${percentage * 3.6}deg,
            #F3F4F6 0deg
          )`,
        }}
      >

        <div className="risk-circle-inner">

          <h1>{score}</h1>

          

        </div>

      </div>

      <div className={`risk-badge ${level.toLowerCase()}`}>
        <AlertTriangle size={16} />

        {level} Risk

      </div>

      <div className="confidence-box">

    <div className="confidence-title">
        Confidence
    </div>

    <div className="confidence-value">
        {confidence}%
    </div>

    <div className="confidence-text">
        Based on retrieved contract evidence.
    </div>

</div>

    </div>
  );
}

export default RiskScoreCard;