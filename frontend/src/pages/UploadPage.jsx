import { useState } from "react";

import Navbar from "../components/Navbar";
import UploadCard from "../components/UploadCard";
import SampleContracts from "../components/SampleContracts";
import QuestionCard from "../components/QuestionCard";

function UploadPage() {
  const [selectedContract, setSelectedContract] = useState(null);

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
        />

        {/* Question Section */}
        {selectedContract && (
          <QuestionCard
            selectedContract={selectedContract}
          />
        )}
      </main>
    </>
  );
}

export default UploadPage;