import {
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";

function DetailedAnalysis({ findings = [] }) {
  return (
    <div className="analysis-card">

      <div className="analysis-heading">
        <AlertTriangle size={18} />
        <span>RECOMMENDATIONS</span>
      </div>

      {findings.map((item, index) => {

        const severity = (item.severity || "Low").toLowerCase();

        const Icon =
          severity === "high"
            ? AlertTriangle
            : severity === "medium"
            ? AlertCircle
            : CheckCircle2;

        return (

          <div
            key={index}
            className={`recommendation-card ${severity}`}
          >

            <div className="recommendation-header">

              <div className="recommendation-left">

                <Icon size={20} />

                <div>

                  <h3>{item.title}</h3>

                  <span className={`severity-pill ${severity}`}>
                    {item.severity}
                  </span>

                </div>

              </div>

            </div>

            <p>{item.description}</p>

          </div>

        );

      })}

    </div>
  );
}

export default DetailedAnalysis;