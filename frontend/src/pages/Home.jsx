import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, FileText, ArrowRight, Github, Briefcase, AlignLeft, Sparkles } from 'lucide-react';
import axios from 'axios';

const Home = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    targetRole: '',
    jobDescription: '',
    githubUrl: '',
    manualSkills: ''
  });
  const [files, setFiles] = useState([]);

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setFiles([...e.target.files]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files) setFiles([...e.dataTransfer.files]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const data = new FormData();
    data.append('target_role', formData.targetRole);
    data.append('job_description', formData.jobDescription);
    data.append('github_url', formData.githubUrl);
    data.append('manual_skills', formData.manualSkills);
    
    files.forEach(file => {
      data.append('files', file);
    });

    try {
      // Use environment variable for production deployments (like Vercel to Render)
      const baseUrl = import.meta.env.VITE_API_URL || '';
      const response = await axios.post(`${baseUrl}/api/analyze`, data, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      navigate('/dashboard', { state: { report: response.data.data } });
    } catch (error) {
      console.error("Error analyzing skills:", error);
      alert("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-container" style={{ maxWidth: '800px', margin: '0 auto', animation: 'fadeIn 0.5s ease' }}>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h1 className="text-gradient" style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
          <Sparkles color="var(--primary-brand)" size={32} /> Hybrid AI Skill Gap Analysis
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
          Upload your resume and job description. Our Multi-Model architecture (SBERT + Knowledge Graph + XGBoost) will analyze your proficiency and predict your optimal career trajectory.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div>
            <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Briefcase size={16} /> Target Role
            </label>
            <input 
              type="text" 
              name="targetRole"
              placeholder="e.g., Data Scientist"
              className="input-field"
              value={formData.targetRole}
              onChange={handleInputChange}
            />
          </div>
          <div>
            <label className="label">GitHub Profile URL (Optional)</label>
            <div style={{ position: 'relative' }}>
              <Github size={18} style={{ position: 'absolute', top: '12px', left: '12px', color: 'var(--text-secondary)' }} />
              <input 
                type="url" 
                name="githubUrl"
                placeholder="https://github.com/username"
                className="input-field"
                style={{ paddingLeft: '2.5rem' }}
                value={formData.githubUrl}
                onChange={handleInputChange}
              />
            </div>
          </div>
        </div>

        <div>
           <label className="label">Upload Resume (PDF/DOCX)</label>
           <div 
             onDragOver={(e) => e.preventDefault()}
             onDrop={handleDrop}
             style={{
               border: '2px dashed var(--border-color)',
               borderRadius: '12px',
               padding: '2rem',
               textAlign: 'center',
               backgroundColor: 'var(--bg-secondary)',
               cursor: 'pointer',
               transition: 'border-color 0.2s'
             }}
           >
             <input 
               type="file" 
               multiple 
               accept=".pdf,.docx" 
               onChange={handleFileChange}
               style={{ display: 'none' }}
               id="file-upload"
             />
             <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
               <UploadCloud size={40} color="var(--primary-brand)" />
               <h4 style={{ color: 'var(--text-primary)' }}>Click to upload or drag & drop</h4>
               <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>PDF or DOCX documents up to 10MB</p>
             </label>
             {files.length > 0 && (
               <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                 {files.map((f, i) => (
                   <span key={i} style={{ backgroundColor: 'var(--bg-primary)', padding: '0.25rem 0.75rem', borderRadius: '20px', fontSize: '0.8rem', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                     <FileText size={14} /> {f.name}
                   </span>
                 ))}
               </div>
             )}
           </div>
        </div>

        <div>
          <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <AlignLeft size={16} /> Job Description (Target Skills)
          </label>
          <textarea 
            name="jobDescription"
            rows={5}
            placeholder="Paste the job description here..."
            className="input-field"
            value={formData.jobDescription}
            onChange={handleInputChange}
            style={{ resize: 'vertical' }}
          />
        </div>

        <button 
          type="submit" 
          className="btn-primary" 
          disabled={loading || (!formData.jobDescription && files.length === 0 && !formData.githubUrl)}
          style={{ width: '100%', justifyContent: 'center', padding: '1rem', fontSize: '1.1rem', marginTop: '1rem' }}
        >
          {loading ? 'Running Hybrid Analysis...' : 'Generate AI Diagnostics'}
          {!loading && <ArrowRight size={20} />}
        </button>
      </form>
    </div>
  );
};

export default Home;
