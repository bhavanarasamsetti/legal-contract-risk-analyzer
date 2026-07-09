import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, ChevronRight, Zap } from "lucide-react";
import { analyzeContract } from "../services/api";
import "../styles/question.css";

const sampleQuestions = {
  "employment_agreement.pdf": [
    "Summarize this employment agreement.",
    "What confidentiality rules must the employee follow?",
    "What happens when the employee leaves the company?",
  ],

  "data_processing_agreement.pdf": [
    "How should a data breach be handled?",
    "What are the company's GDPR responsibilities?",
    "What security measures protect personal data?",
  ],

  "atlassian_customer_dpa.pdf": [
    "What happens if a security incident occurs?",
    "How does Atlassian protect customer data?",
    "Summarize the main privacy and compliance requirements.",
  ],
};

function QuestionCard({ selectedContract }) {
  const navigate = useNavigate();

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const examples = sampleQuestions[selectedContract] || [];

  async function handleAnalyze() {
    if (!question.trim()) {
      alert("Please enter a question.");
      return;
    }

    try {
      setLoading(true);

      const result = await analyzeContract(
        question,
        selectedContract
      );

      navigate("/results", {
        state: result,
      });

    } catch (error) {
      console.error(error);
      alert("Failed to analyze contract.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="question-section">

      <p className="question-label">
        ASK AI
      </p>

      <h2>
        Ask a question about the contract
      </h2>

      <p className="question-description">
        Type your own question or choose a suggestion below to begin analysis.
      </p>

      <div className="search-box">

        <Search size={22} />

        <input
          type="text"
          placeholder="e.g. What are the termination conditions for this contract?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

      </div>

      <p className="suggestion-title">
        SUGGESTED QUESTIONS
      </p>

      <div className="question-list">

        {examples.map((item) => (

          <button
            key={item}
            className={`question-item ${
              question === item ? "active" : ""
            }`}
            onClick={() => setQuestion(item)}
          >

            <span>{item}</span>

            <ChevronRight size={18} />

          </button>

        ))}

      </div>

      <button
        className="analyze-btn"
        disabled={loading}
        onClick={handleAnalyze}
      >

        <Zap size={18} />

        {loading ? "Analyzing..." : "Analyze Contract"}

      </button>

    </section>
  );
}

export default QuestionCard;