import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Sun, Moon, Activity, Info } from 'lucide-react';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import './index.css';

function App() {
  const [theme, setTheme] = useState('light');

  // Initialize theme from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <Router>
      <div className="app-layout">
        <nav style={navStyle}>
          <div style={navContainerStyle}>
            <Link to="/" style={logoStyle}>
              <Activity size={24} color="var(--primary-brand)" />
              <span style={{ fontWeight: 800, letterSpacing: '-0.5px' }}>MoSSGA AI</span>
            </Link>
            
            <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
              <Link to="/about" style={navLinkStyle}>
                <Info size={18} /> About
              </Link>
              <button onClick={toggleTheme} className="btn-secondary" style={{ padding: '0.5rem' }}>
                {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
              </button>
            </div>
          </div>
        </nav>

        <main className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/about" element={
              <div className="card">
                <h2>About MoSSGA</h2>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  Hybrid Semantic and Multi-Model Approach for AI-Driven Skill Gap Analysis in Workforce Informatics.
                  <br/><br/>
                  This research-grade application utilizes Sentence-BERT and a NetworkX Knowledge Graph fused with XGBoost predictors to assess workforce skill deficits dynamically.
                </p>
              </div>
            } />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

// Inline styles for simple structural layout to keep components clean
const navStyle = {
  borderBottom: '1px solid var(--border-color)',
  backgroundColor: 'var(--bg-card)',
  position: 'sticky',
  top: 0,
  zIndex: 100
};

const navContainerStyle = {
  maxWidth: '1200px',
  margin: '0 auto',
  padding: '1rem',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center'
};

const logoStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
  textDecoration: 'none',
  color: 'var(--text-primary)',
  fontSize: '1.25rem'
};

const navLinkStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.4rem',
  textDecoration: 'none',
  color: 'var(--text-secondary)',
  fontWeight: 500,
  transition: 'color 0.2s'
};

export default App;
