import { Sparkles } from "lucide-react";

function AISummaryCard({ answer }) {

  const cleanedAnswer = answer
    .replace(/\*\*/g, "")        
    .replace(/\[\d+\]/g, "")     
    .replace(/\b\d+\.\s/g, ""); 

  return (
    <div className="summary-card">

      <div className="summary-title">

        <div className="summary-icon">
          <Sparkles size={18} />
        </div>

        <span>EXECUTIVE SUMMARY</span>

      </div>

      <p className="summary-text">
        {cleanedAnswer}
      </p>

    </div>
  );
}

export default AISummaryCard;