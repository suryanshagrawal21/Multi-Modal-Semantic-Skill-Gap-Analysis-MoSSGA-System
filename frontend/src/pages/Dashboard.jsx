import React from 'react';
import { useLocation, Navigate, Link } from 'react-router-dom';
import { ArrowLeft, CheckCircle, AlertTriangle, Brain, Target, Compass, Cpu, TrendingUp, Award } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const Dashboard = () => {
  const location = useLocation();
  const report = location.state?.report;

  if (!report) {
    return <Navigate to="/" />;
  }

  const hybridScore = report.mossga_score || 0;
  const matchPct = (hybridScore / 100).toFixed(2);
  const gapData = report.gap_analysis || {};
  const explainable = report.explainable_insights || [];
  const careerPath = report.career_path || {};
  
  // Format data for radar chart (matched vs missing)
  const allSkillsSet = new Set([
    ...(report.resume_skills || []),
    ...(gapData.missing_with_severity?.map(s => s.skill) || [])
  ]);
  
  const radarData = Array.from(allSkillsSet).slice(0, 8).map(skill => {
    const isMissing = gapData.missing_with_severity?.find(s => s.skill === skill);
    return {
      skill: skill,
      proficiency: isMissing ? 0 : 100,
      required: isMissing ? (isMissing.importance * 100) : 100
    };
  });

  return (
    <div style={{ animation: 'fadeIn 0.5s ease' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 className="text-gradient">MoSSGA Diagnostics</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Advanced Hybrid Graph-ML Pipeline Analysis</p>
        </div>
        <Link to="/" className="btn-secondary">
          <ArrowLeft size={18} /> New Analysis
        </Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* SCORE GAUGE CARD */}
        <div className="card" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
           <h3 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
             <Cpu size={20} color="var(--primary-brand)" /> Hybrid AI Score
           </h3>
           <div style={{ position: 'relative', width: '160px', height: '160px', borderRadius: '50%', background: `conic-gradient(var(--primary-brand) ${hybridScore}%, var(--bg-secondary) ${hybridScore}%)`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
             <div style={{ position: 'absolute', width: '140px', height: '140px', backgroundColor: 'var(--bg-card)', borderRadius: '50%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--text-primary)' }}>{hybridScore}</span>
             </div>
           </div>
           <p style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Semantic + Graph + XGBoost weighted model</p>
        </div>

        {/* RADAR CHART */}
        <div className="card">
           <h3 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
             <Target size={20} /> Skill Competency Profile
           </h3>
           <div style={{ height: '200px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                  <PolarGrid stroke="var(--border-color)" />
                  <PolarAngleAxis dataKey="skill" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                  <Radar name="You" dataKey="proficiency" stroke="var(--primary-brand)" fill="var(--primary-brand)" fillOpacity={0.5} />
                  <Radar name="Required" dataKey="required" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.2} />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                </RadarChart>
              </ResponsiveContainer>
           </div>
        </div>

      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
         
         <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* EXPLAINABLE AI */}
            <div className="card">
              <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Brain color="var(--primary-brand)" /> Explainable AI Insights</h2>
              <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                 {explainable.length > 0 ? explainable.map((item, idx) => (
                   <div key={idx} style={{ padding: '1rem', borderLeft: '4px solid var(--danger)', backgroundColor: 'var(--bg-secondary)', borderRadius: '0 8px 8px 0' }}>
                     <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                       <h4 style={{ textTransform: 'capitalize' }}>{item.skill}</h4>
                     </div>
                     <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{item.insight}</p>
                     {item.missing_prerequisites && item.missing_prerequisites.length > 0 && (
                       <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--danger)', fontWeight: 500 }}>
                         Graph Prerequisite Gap: {item.missing_prerequisites.join(', ')}
                       </div>
                     )}
                   </div>
                 )) : (
                   <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                     <CheckCircle size={20} /> No critical missing skills flagged by the AI!
                   </div>
                 )}
              </div>
            </div>
         </div>

         <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
             {/* WORKFORCE PROGRESSION */}
             <div className="card">
               <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Compass color="var(--primary-brand)" /> Career Trajectory</h2>
               
               <div style={{ marginTop: '1.5rem' }}>
                 <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                   <Award size={16} /> Next Best Skills to Learn:
                 </h4>
                 <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                   {careerPath.next_skills?.map((skill, idx) => (
                     <span key={idx} style={{ backgroundColor: 'rgba(37, 99, 235, 0.1)', color: 'var(--primary-brand)', padding: '4px 12px', borderRadius: '16px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(37, 99, 235, 0.2)' }}>
                       {skill}
                     </span>
                   )) || <span style={{ color: 'var(--text-secondary)' }}>No predictive path generated.</span>}
                 </div>
               </div>

               <div style={{ marginTop: '2rem' }}>
                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <TrendingUp size={16} /> Skill Progression Model:
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                    <div style={{ borderLeft: '3px solid var(--success)', paddingLeft: '1rem' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Beginner</span>
                      <div>{report.career_intelligence?.learning_roadmap?.Beginner?.join(', ') || '...'}</div>
                    </div>
                    <div style={{ borderLeft: '3px solid var(--warning)', paddingLeft: '1rem' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Intermediate</span>
                      <div>{report.career_intelligence?.learning_roadmap?.Intermediate?.join(', ') || '...'}</div>
                    </div>
                    <div style={{ borderLeft: '3px solid var(--danger)', paddingLeft: '1rem' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Advanced</span>
                      <div>{report.career_intelligence?.learning_roadmap?.Advanced?.join(', ') || '...'}</div>
                    </div>
                  </div>
               </div>
             </div>
         </div>

      </div>

    </div>
  );
};

export default Dashboard;
