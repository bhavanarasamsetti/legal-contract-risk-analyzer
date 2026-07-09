import { FileText } from "lucide-react";

function SourceDocuments({ sources = [] }) {

  return (

    <div className="result-card">

      <div className="card-heading">

        <FileText size={18} />

        <span>SOURCES</span>

      </div>

      {sources.map((source) => (

        <div
          className="source-item"
          key={source.chunk_id}
        >

          <strong>

            {source.document_name}

          </strong>

          <small>

            Section {source.section}

            {" • "}

            Pages {source.pages.join(", ")}

          </small>

        </div>

      ))}

    </div>

  );

}

export default SourceDocuments;