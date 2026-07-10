import { FileText } from "lucide-react";

function SourceDocuments({ sources = [] }) {

  return (

    <div className="result-card">

      <div className="card-heading">

        <FileText size={18} />

        <span>SOURCES</span>

      </div>

      {sources.length > 0 && (

        <p className="source-file">

          {sources[0].document_name}

        </p>

      )}

      {sources.map((doc, index) => (

        <div
          key={index}
          className="source-item"
        >

          <strong>

            Section {doc.section}

          </strong>

          <small>

            Page{doc.pages.length > 1 ? "s" : ""}{" "}

            {doc.pages.join(", ")}

          </small>

        </div>

      ))}

    </div>

  );

}

export default SourceDocuments;