const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

console.log("VITE_API_URL =", import.meta.env.VITE_API_URL);
console.log("API_URL =", API_URL);
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

export async function uploadContract(file) {

  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Upload failed.");
  }

  return await response.json();
}