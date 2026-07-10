import { BrowserRouter, Routes, Route } from "react-router-dom";

import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";
import AboutPage from "./pages/AboutPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route
    path="/about"
    element={<AboutPage />}
/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;