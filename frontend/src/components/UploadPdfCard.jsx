import { FileUp } from "lucide-react";
import { useState } from "react";
import { uploadContract } from "../services/api";

function UploadPdfCard({
  uploadedContract,
  setUploadedContract,
  setSelectedContract,
  uploading,
  setUploading,
}) {

  
  
  const [uploadedName,setUploadedName]=useState("");

  async function handleUpload(e) {

    const file = e.target.files[0];

    if (!file) return;

   try {

  setUploading(true);

  // Immediately clear sample selection
  setSelectedContract(null);

  const result = await uploadContract(file);

setUploadedContract(result.document_name);

setUploadedName(result.filename);
      alert(`${result.filename} uploaded successfully`);

    } catch (err) {

      console.error(err);

      alert("Upload failed.");

    } finally {

      setUploading(false);

    }

  }

  return (

  <div
  className={`contract-card ${
    uploadedContract ? "active" : ""
  }`}
>

    <div className="file-icon">
      <FileUp size={26} />
    </div>

   <h2>Upload Contract</h2>

 <p>

{uploadedContract
    ? uploadedName
    : "Upload any legal contract and analyze it using AI."}

</p>

    <div className="tags">

      <span>Custom</span>
      <span>PDF</span>
      <span>AI</span>

    </div>

   <label
  className={uploadedContract ? "selected-button" : ""}
  style={{
    width: "100%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    cursor: "pointer",
    marginTop: "28px",
    height: "48px",
    borderRadius: "12px",
    border: uploadedContract ? "none" : "1px solid #D1D5DB",

background: uploadedContract ? "#5B46B2" : "#FFFFFF",

color: uploadedContract ? "#FFFFFF" : "#111827",
    fontWeight: 600,
  }}
>

{
  uploading
    ? "Uploading..."
    : uploadedContract
    ? "✓ Selected"
    : "Upload Contract"
}

  <input
    hidden
    type="file"
    accept=".pdf"
    onChange={handleUpload}
  />

</label>


   
  </div>

);

}

export default UploadPdfCard;