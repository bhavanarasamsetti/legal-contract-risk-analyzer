import { FileText, ExternalLink } from "lucide-react";

function ReferencedClauses() {

  const clauses = [

    {
      section: "Section 9.1",
      title: "Confidentiality Obligations",
      page: "Page 12",
    },

    {
      section: "Section 9.2",
      title: "Return of Company Property",
      page: "Page 12",
    },

    {
      section: "Section 8.3",
      title: "Termination Conditions",
      page: "Page 9",
    },

  ];

  return (

    <div className="clauses-card">

      <div className="clauses-title">

        <FileText size={18} />

        <span>REFERENCED CLAUSES</span>

      </div>

      {clauses.map((clause,index)=>(

        <div
          key={index}
          className="clause-item"
        >

          <div className="clause-left">

            <div className="clause-section">

              {clause.section}

            </div>

            <h4>

              {clause.title}

            </h4>

            <p>

              {clause.page}

            </p>

          </div>

          <ExternalLink size={18} />

        </div>

      ))}

    </div>

  );

}

export default ReferencedClauses;