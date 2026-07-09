import { ShieldCheck, AlertTriangle } from "lucide-react";

function RiskScoreCard() {
  return (
    <div className="risk-card">

      <div className="card-label">
        OVERALL RISK SCORE
      </div>

      <div className="risk-circle">

        <svg viewBox="0 0 120 120">

          <circle
            cx="60"
            cy="60"
            r="46"
            className="circle-bg"
          />

          <circle
            cx="60"
            cy="60"
            r="46"
            className="circle-progress"
          />

        </svg>

        <div className="risk-score">

          <div className="score">
            7.4
          </div>

          <div className="outof">
            / 10
          </div>

        </div>

      </div>

      <div className="risk-pill">

        <AlertTriangle size={16} />

        Medium Risk

      </div>

      <div className="risk-divider"></div>

      <div className="risk-stats">

        <div>

          <span>High Risk</span>

          <strong className="high">
            1 clause
          </strong>

        </div>

        <div>

          <span>Medium Risk</span>

          <strong className="medium">
            2 clauses
          </strong>

        </div>

        <div>

          <span>Low Risk</span>

          <strong className="low">
            1 clause
          </strong>

        </div>

      </div>

    </div>
  );
}

export default RiskScoreCard;