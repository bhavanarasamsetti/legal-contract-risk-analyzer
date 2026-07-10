import "../styles/sampleContracts.css";
import UploadPdfCard from "./UploadPdfCard";

const contracts = [
  {
    id: "employment_agreement.pdf",
    title: "Employment Agreement",
    description:
      "Standard employment contract covering salary, confidentiality, and termination provisions.",
    tags: ["Salary", "Confidentiality", "Termination"],
  },
  {
    id: "data_processing_agreement.pdf",
    title: "Data Processing Agreement",
    description:
      "GDPR-compliant agreement outlining processor responsibilities and breach notification procedures.",
    tags: ["GDPR", "Data Breach", "Processor"],
  },
  {
    id: "atlassian_customer_dpa.pdf",
    title: "Enterprise DPA Sample",
    description: "Enterprise-grade data processing agreement for GDPR compliance.",
    tags: ["Security", "Compliance", "Privacy"],
  },
];

function SampleContracts({
  selectedContract,
  setSelectedContract,
  uploadedContract,
  setUploadedContract,
  uploading,
  setUploading,
}) {
  return (
    <section className="contracts-section">

      <h3 className="contracts-title">
        CHOOSE A SAMPLE CONTRACT
      </h3>

      <div className="contracts-grid">

        {contracts.map((contract) => (

          <div
            key={contract.id}
            className={`contract-card ${
              selectedContract === contract.id ? "active" : ""
            }`}
          >

            {selectedContract === contract.id && (
              <div className="selected-badge">
                ✓ Selected
              </div>
            )}

            <div className="file-icon">
              📄
            </div>

            <h2>{contract.title}</h2>

            <p>{contract.description}</p>

            <div className="tags">

              {contract.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}

            </div>

         <button
  disabled={uploading}
  onClick={() => {

    if (uploading) return;

    setSelectedContract(contract.id);

    setUploadedContract(null);

  }}
  className={
    selectedContract === contract.id
      ? "selected-button"
      : ""
  }
>
              {selectedContract === contract.id
                ? "✓ Using This Sample"
                : "Use Sample"}
            </button>

          </div>

        ))}
  <UploadPdfCard
  uploadedContract={uploadedContract}
  setUploadedContract={setUploadedContract}
  setSelectedContract={setSelectedContract}
  uploading={uploading}
  setUploading={setUploading}
/>
      </div>

    </section>
  );
}

export default SampleContracts;