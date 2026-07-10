import { useState } from "react";

import Navbar from "../components/Navbar";
import UploadCard from "../components/UploadCard";
import SampleContracts from "../components/SampleContracts";

import QuestionCard from "../components/QuestionCard";

function UploadPage() {
 const [selectedContract, setSelectedContract] = useState(null);
const [uploadedContract, setUploadedContract] = useState(null);
const [uploading, setUploading] = useState(false);
  return (
    <>
      <Navbar />

      <main
        style={{
          background: "#FCFCFD",
          minHeight: "100vh",
          paddingBottom: "80px",
        }}
      >
        {/* Hero */}
        <UploadCard />

        {/* Sample Cards */}
  <SampleContracts
  selectedContract={selectedContract}
  setSelectedContract={setSelectedContract}
  uploadedContract={uploadedContract}
  setUploadedContract={setUploadedContract}
  uploading={uploading}
  setUploading={setUploading}
/>


{(selectedContract || uploadedContract) && (
  <QuestionCard
    selectedContract={selectedContract}
    uploadedContract={uploadedContract}
  />
)}
   {uploading && (
 <div
  style={{
    display: "flex",
    justifyContent: "center",
    marginTop: "60px",
    marginBottom: "60px",
    width: "100%",
  }}
>
    <div className="analysis-status">

      <div className="analysis-spinner"></div>

      <div>

       <strong>Preparing your contract for AI analysis...</strong>

        <p>
          Reading the PDF, extracting text, creating embeddings and indexing it
          for AI search. This usually takes 5–10 seconds.
        </p>

      </div>

    </div>
  </div>
)}     
      </main>
    </>
  );
}

export default UploadPage;