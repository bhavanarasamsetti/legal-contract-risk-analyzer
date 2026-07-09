const API_URL = "http://127.0.0.1:8000";

export async function analyzeContract(question, selectedContract) {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      filter: {
        document_name: {
          $eq: selectedContract,
        },
      },
    }),
  });

  if (!response.ok) {
    throw new Error("Analysis failed.");
  }

  return await response.json();
}