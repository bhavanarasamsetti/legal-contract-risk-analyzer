import "../styles/about.css";

function AboutPage() {
  return (
    <main className="about-page">

      <section className="about-hero">

        <h1>About LexAI</h1>

      <p>
  LexAI is an AI-powered Legal Contract Risk Analyzer that combines
  Retrieval-Augmented Generation (RAG), Hybrid Search, and Large Language
  Models to analyze legal agreements. It generates executive summaries,
  identifies legal risks, provides confidence-based recommendations, and
  answers contract-specific questions using only retrieved contract evidence.
</p>

      </section>

      <section className="about-section">

        <h2>Features</h2>

        <div className="feature-grid">

          <div className="feature-card">
<h3>📄 Contract Analysis</h3>
<p>Analyze Employment Agreements, Data Processing Agreements, and SaaS DPAs using AI.</p>
          </div>

          <div className="feature-card">
           <h3>🔍 Hybrid Retrieval</h3>
<p>Combines semantic vector search, BM25 keyword search, and Reciprocal Rank Fusion for accurate retrieval.</p>
          </div>

          <div className="feature-card">
            <h3>⚖️ Risk Assessment</h3>
<p>Generates overall legal risk scores, confidence scores, findings, and actionable recommendations.</p>
          </div>

          <div className="feature-card">
            <h3>📚 Grounded AI Answers</h3>
<p>Every response is generated only from retrieved contract evidence to reduce hallucinations.</p>
          </div>

          <div className="feature-card">
            <h3>📊 RAG Evaluation</h3>
<p>Retrieval and generation quality are evaluated using RAGAS with faithfulness, relevance, precision, and recall metrics.</p>
          </div>

          <div className="feature-card">
           <h3>📈 Observability</h3>
<p>Every retrieval and LLM interaction is traced with Langfuse for debugging and performance monitoring.</p>
          </div>

        </div>

      </section>

      <section className="about-section">

        <h2>Technology Stack</h2>

        <div className="tech-stack">

        <span className="tech-badge react">React</span>

<span className="tech-badge fastapi">FastAPI</span>

<span className="tech-badge openai">GPT-4o Mini</span>

<span className="tech-badge openai">OpenAI Embeddings</span>

<span className="tech-badge pinecone">Pinecone</span>

<span className="tech-badge langchain">LangChain</span>

<span className="tech-badge rag">Hybrid RAG</span>

<span className="tech-badge rag">BM25</span>

<span className="tech-badge rag">RRF</span>

<span className="tech-badge rag">Langfuse</span>

<span className="tech-badge rag">RAGAS</span>

        </div>

      </section>

      <section className="about-section">

        <h2>How LexAI Works</h2>

        <div className="steps">

          <div className="step-card">
            <div className="step-number">1</div>
           <h4>Ingest</h4>

<p>Upload a contract or choose one of the built-in legal agreement samples.</p>
          </div>

          <div className="step-card">
            <div className="step-number">2</div>
<h4>Retrieve</h4>

<p>Hybrid Retrieval combines OpenAI embeddings, Pinecone, BM25, and Reciprocal Rank Fusion to locate the most relevant clauses.</p>
          </div>

          <div className="step-card">
            <div className="step-number">3</div>
            <h4>Analyze</h4>

<p>GPT-4o Mini performs grounded legal analysis to generate summaries, risk scores, findings, confidence scores, and recommendations.</p>
          </div>

          <div className="step-card">
            <div className="step-number">4</div>
<h4>Evaluate</h4>

<p>Results are traced with Langfuse and the retrieval pipeline is evaluated using RAGAS to ensure reliable AI responses.</p>
          </div>

        </div>

      </section>

      <div className="about-footer">

       Built using React • FastAPI • GPT-4o Mini • OpenAI Embeddings • Pinecone • Hybrid RAG • BM25 • RRF • Langfuse • RAGAS

      </div>

    </main>
  );
}

export default AboutPage;