import { Sparkles } from "lucide-react";

function AISummaryCard({ answer }) {
  return (
    <div className="summary-card">

      <div className="summary-title">

        <div className="summary-icon">
          <Sparkles size={18} />
        </div>

        <span>AI SUMMARY</span>

      </div>

      <p className="summary-text">
        {answer}
      </p>

    </div>
  );
}

export default AISummaryCard;