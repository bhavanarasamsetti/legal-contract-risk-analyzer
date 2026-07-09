import {
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";

function DetailedAnalysis() {

  const findings = [

    {
      level: "High",
      color: "high-risk",
      icon: <AlertTriangle size={20} />,
      title: "Broad Confidentiality Obligations",
      text:
        "The confidentiality clause extends for three years after employment and applies to a wide range of business information.",
    },

    {
      level: "Medium",
      color: "medium-risk",
      icon: <AlertCircle size={20} />,
      title: "Return of Company Documents",
      text:
        "Employees must return all confidential documents and electronic records immediately upon termination.",
    },

    {
      level: "Low",
      color: "low-risk",
      icon: <CheckCircle2 size={20} />,
      title: "GDPR Compliance",
      text:
        "The agreement references GDPR obligations and employee responsibility when handling personal data.",
    },

  ];

  return (

    <div className="analysis-card">

      <div className="analysis-title">

        <AlertTriangle size={18} />

        <span>DETAILED ANALYSIS</span>

      </div>

      {findings.map((item,index)=>(

        <div
          key={index}
          className={`finding ${item.color}`}
        >

          <div className="finding-header">

            {item.icon}

            <div>

              <strong>
                {item.title}
              </strong>

              <p>{item.level} Risk</p>

            </div>

          </div>

          <div className="finding-body">

            {item.text}

          </div>

        </div>

      ))}

    </div>

  );

}

export default DetailedAnalysis;