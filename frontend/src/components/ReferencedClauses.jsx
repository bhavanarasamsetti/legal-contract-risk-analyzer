import { FileText } from "lucide-react";

function ReferencedClauses({ sources = [] }) {

  return (

    <div className="clauses-card">

      <div className="clauses-heading">

        <FileText size={18} />

        <span>REFERENCED CLAUSES</span>

      </div>

      {sources.length === 0 ? (

        <p className="empty-text">

          No clauses referenced.

        </p>

      ) : (

        sources.map((source, index) => (

          <div
            className="clause-card"
            key={index}
          >

            <div className="clause-top">

              <div className="clause-section">

                Section {source.section}

              </div>

              <div className="clause-page">

                Page {source.pages.join(", ")}

              </div>

            </div>

            <h3>

              {source.section_title ||
                "Contract Clause"}

            </h3>

            <p>

              {source.document_name}

            </p>

          </div>

        ))

      )}

    </div>

  );

}

export default ReferencedClauses;