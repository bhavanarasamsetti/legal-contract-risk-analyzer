import { MessageSquare } from "lucide-react";

function QuestionInfo({ question }) {
  return (
    <div className="result-card">

      <div className="card-heading">

        <MessageSquare size={18} />

        <span>QUESTION</span>

      </div>

      <p className="question-text">
        {question}
      </p>

    </div>
  );
}

export default QuestionInfo;