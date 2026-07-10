import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Search,
  ChevronRight,
  Zap,
} from "lucide-react";
import { analyzeContract} from "../services/api";
import "../styles/question.css";

const sampleQuestions = {
  "employment_agreement.pdf": [
    "Summarize this employment agreement.",
    "What confidentiality obligations must the employee follow?",
    "What legal risks should I review before signing this employment agreement?"
  ],

 "data_processing_agreement.pdf": [
  "Summarize this data processing agreement.",                  
  "What GDPR compliance obligations are included?",             
  "What are the biggest legal and liability risks in this agreement?" 
],

 "atlassian_customer_dpa.pdf": [
  "Summarize this Data Processing Addendum.",                   
  "What customer data protection responsibilities are included?", 
  "What security, liability, and compliance risks should I review before accepting this agreement?" 
]
};

function QuestionCard({
  selectedContract,
  uploadedContract,
}) {
  const navigate = useNavigate();

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);


const examples =
  uploadedContract
    ? []
    : sampleQuestions[selectedContract] || [];
  

  async function handleAnalyze() {
    if (!question.trim()) {
      alert("Please enter a question.");
      return;
    }

    try {
      setLoading(true);

      const contract = uploadedContract || selectedContract;

if (!contract) {
    alert("Please select a sample contract or upload a PDF.");
    return;
}

const result = await analyzeContract(
    question,
    contract
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

      {examples.length > 0 && (
  <>
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
  </>
)}
      <button
        className="analyze-btn"
        disabled={loading}
        onClick={handleAnalyze}
      >

        <Zap size={18} />

        {loading ? "Analyzing..." : "Analyze Contract"}

      </button>
     {loading && (
  <div className="analysis-status">

    <div className="analysis-spinner"></div>

    <div>

      <strong>AI is reviewing your contract...</strong>

      <p>
        Analyzing legal clauses, evaluating risks, and preparing your report.
This usually takes 3–5 seconds.
      </p>

    </div>

  </div>
)}
    </section>
  );
}

export default QuestionCard;