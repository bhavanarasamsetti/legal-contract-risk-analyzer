import "../styles/navbar.css";
import { Shield } from "lucide-react";

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <div className="logo-box">
            <Shield size={20} strokeWidth={2.3} />
          </div>

          <span className="logo-text">
            LexAI
          </span>

          <span className="beta-badge">
            BETA
          </span>
        </div>

       <div className="navbar-links">

  <a href="#">
    Docs
  </a>

  <a
    href="https://github.com/"
    target="_blank"
    rel="noreferrer"
  >
    GitHub
  </a>

</div>
      </div>
    </header>
  );
}

export default Navbar;