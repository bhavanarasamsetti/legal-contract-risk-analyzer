import "../styles/upload.css";

function UploadCard() {
  return (
    <section className="hero-section">

      <div className="hero-badge">
        ⚡ RAG-Powered Analysis
      </div>

      <h1 className="hero-title">
        AI-Powered Legal
        <br />
        Contract Review
      </h1>

      <p className="hero-description">
        Analyze legal contracts using Retrieval-Augmented Generation (RAG).
        <br />
        Ask natural language questions and receive grounded answers with
        cited contract clauses.
      </p>

    </section>
  );
}

export default UploadCard;