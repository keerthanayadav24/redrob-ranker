import React, { useState } from 'react';
import axios from 'axios';
import { Sparkles, Briefcase } from 'lucide-react';
import './index.css';

const API = 'http://localhost:8000';

function App() {
  const [candidates, setCandidates]               = useState([]);
  const [loading, setLoading]                     = useState(false);
  const [error, setError]                         = useState(null);
  const [selectedCandidate, setSelectedCandidate] = useState(null);

  const handleRank = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API}/api/top100`);
      const data = res.data;
      setCandidates(data);
      if (data.length > 0) setSelectedCandidate(data[0]);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Could not reach the backend. Make sure `uvicorn backend.main:app --reload` is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  // Dashboard stats derived from results
  const stats = candidates.length > 0 ? {
    total:         candidates.length,
    avgScore:      candidates.reduce((s, c) => s + c.score, 0) / candidates.length,
    strongHire:    candidates.filter(c => c.decision === 'Strong Hire').length,
    immediate:     candidates.filter(c => c.interview_priority === 'Immediate').length,
    highRisk:      candidates.filter(c => c.risk === 'High').length,
    avgConfidence: candidates.reduce((s, c) => s + (c.confidence || 0), 0) / candidates.length,
  } : null;

  const rankClass = (rank) =>
    rank === 1 ? 'top-1' : rank === 2 ? 'top-2' : rank === 3 ? 'top-3' : 'standard';

  const decisionClass = (d) =>
    (d || '').toLowerCase().replace(/\s/g, '-');

  return (
    <div className="app-container">

      {/* ── Header ── */}
      <header className="header">
        <span className="hero-tag">AI Recruiter Copilot</span>
        <h1>Hire Smarter with Explainable AI</h1>
        <p>Semantic ranking · Behaviour analysis · Recruiter intelligence · Explainable recommendations</p>
      </header>

      {/* ── Dashboard stats ── */}
      {stats && (
        <div className="dashboard-stats">
          {[
            ['Candidates',  stats.total],
            ['Avg Match',   Math.round(stats.avgScore * 100) + '%'],
            ['Strong Hire', stats.strongHire],
            ['Immediate',   stats.immediate],
            ['High Risk',   stats.highRisk],
            ['Confidence',  Math.round(stats.avgConfidence) + '%'],
          ].map(([label, value]) => (
            <div className="stat-card" key={label}>
              <span>{label}</span>
              <h3>{value}</h3>
            </div>
          ))}
        </div>
      )}

      {/* ── Main grid ── */}
      <main className="main-grid">

        {/* Left: trigger panel */}
        <div className="jd-section glass-panel">
          <h2><Briefcase size={22} color="var(--primary)" /> Ranking Control</h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.95rem', lineHeight: 1.6 }}>
            The AI ranker has already analysed all 100,000 candidates against the
            Senior AI Engineer JD. Click the button to load the top-100 shortlist.
          </p>
          <div style={{ background: 'var(--surface-2)', borderRadius: 14, padding: '1rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>
            <strong style={{ color: 'var(--primary)' }}>What's happening under the hood:</strong>
            <ul style={{ marginTop: 8, paddingLeft: 18, lineHeight: 2 }}>
              <li>TF-IDF semantic similarity vs JD text</li>
              <li>5-dimension scoring (career · skills · availability · location · bonus)</li>
              <li>Honeypot &amp; keyword-stuffer detection</li>
              <li>All JD disqualifiers encoded explicitly</li>
            </ul>
          </div>
          <button className="btn-primary" onClick={handleRank} disabled={loading}>
            {loading
              ? <><div className="loader" /> Analysing Candidates...</>
              : <><Sparkles size={18} /> Load Top-100 Shortlist</>}
          </button>
          {error && (
            <div style={{ color: 'var(--danger)', background: '#fff0f0', borderRadius: 10, padding: '0.9rem', fontSize: '0.88rem' }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* Right: results */}
        <div className="results-section">
          <div className="results-header">
            <h2>Top Candidates</h2>
            <span style={{ color: 'var(--text-light)' }}>
              {candidates.length > 0 ? `Top ${candidates.length} of 100,000` : 'Waiting…'}
            </span>
          </div>

          {loading ? (
            <div className="loading-state glass-panel">
              <div className="loader" style={{ width: 40, height: 40 }} />
              <p>Scanning candidate pool and generating semantic match scores…</p>
            </div>
          ) : (
            <div className="results-layout">

              {/* Candidate list */}
              <div className="candidate-list">
                {candidates.map((c, idx) => {
                  const profile = c.raw_candidate?.profile || {};
                  return (
                    <div
                      key={c.candidate_id}
                      className={`candidate-card glass-panel ${selectedCandidate?.candidate_id === c.candidate_id ? 'selected-card' : ''}`}
                      onClick={() => setSelectedCandidate(c)}
                    >
                      <div className="card-header">
                        <div style={{ display: 'flex', gap: '0.9rem', alignItems: 'flex-start' }}>
                          <div className={`rank-badge ${rankClass(idx + 1)}`}>{idx + 1}</div>
                          <div className="candidate-info">
                            <h3>{profile.current_title || '—'}</h3>
                            <p>{profile.years_of_experience}y exp · {profile.location}</p>
                          </div>
                        </div>
                        <div className="score-display">
                          <div className={`decision-badge ${decisionClass(c.decision)}`}>{c.decision}</div>
                          <div className="score-value">{Math.round(c.score * 100)}%</div>
                          <div className="score-label">Overall Match</div>
                          <div className="confidence-text">Confidence {c.confidence}%</div>
                        </div>
                      </div>
                      <div className="card-body">
                        <div className="candidate-summary">{c.summary}</div>
                        {c.top_skills?.length > 0 && (
                          <div className="skills-container" style={{ marginTop: 8 }}>
                            {c.top_skills.slice(0, 4).map((sk, i) => (
                              <span key={i} className="skill-tag skill-expert">{sk}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {candidates.length === 0 && !loading && !error && (
                  <div className="loading-state glass-panel" style={{ padding: '3rem' }}>
                    <p>Click "Load Top-100 Shortlist" to see results.</p>
                  </div>
                )}
              </div>

              {/* Candidate detail panel */}
              {selectedCandidate && (
                <div className="candidate-details glass-panel">
                  <div className="detail-header">
                    <div>
                      <h2>{selectedCandidate.raw_candidate?.profile?.current_title}</h2>
                      <p>
                        {selectedCandidate.raw_candidate?.profile?.years_of_experience}y ·{' '}
                        {selectedCandidate.raw_candidate?.profile?.location}
                      </p>
                    </div>
                    <div className={`decision-badge ${decisionClass(selectedCandidate.decision)}`}>
                      {selectedCandidate.decision}
                    </div>
                  </div>

                  <h1 className="detail-score">{Math.round(selectedCandidate.score * 100)}%</h1>
                  <p className="detail-summary">{selectedCandidate.summary}</p>
                  <hr />

                  <h3>AI Match Breakdown</h3>
                  {[
                    ['Semantic Match',  selectedCandidate.semantic_score],
                    ['Skill Coverage',  selectedCandidate.skill_score],
                    ['Career Evidence', selectedCandidate.experience_score],
                    ['Availability',    selectedCandidate.behaviour_score],
                    ['Location',        selectedCandidate.location_score],
                    ['Bonus Signals',   selectedCandidate.notice_score],
                  ].map(([label, val]) => (
                    <div className="metric" key={label}>
                      <span>{label}</span>
                      <progress value={Math.round((val || 0) * 100)} max="100" />
                      <b>{Math.round((val || 0) * 100)}%</b>
                    </div>
                  ))}

                  <hr />
                  <h3>Matched Skill Clusters</h3>
                  <div className="skills-container" style={{ marginTop: 8 }}>
                    {(selectedCandidate.matched_clusters || []).map((cl, i) => (
                      <span key={i} className="skill-tag skill-expert">{cl}</span>
                    ))}
                    {(selectedCandidate.matched_clusters || []).length === 0 && (
                      <span style={{ color: 'var(--danger)', fontSize: '0.85rem' }}>No required clusters matched</span>
                    )}
                  </div>

                  <hr />
                  <h3>Recruiter Insights</h3>
                  <div className="insights-list">
                    {(selectedCandidate.insights || []).map((item, i) => (
                      <div key={i} className="insight-item">{item}</div>
                    ))}
                  </div>

                  <hr />
                  <h3>Recommendation</h3>
                  <div className="recommendation-box">
                    <strong>{selectedCandidate.recommendation}</strong>
                    <p style={{ marginTop: 8 }}>{selectedCandidate.reasoning}</p>
                  </div>

                  <hr />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-light)' }}>
                    <span>Priority: <strong>{selectedCandidate.interview_priority}</strong></span>
                    <span>Risk: <strong style={{ color: selectedCandidate.risk === 'High' ? 'var(--danger)' : 'var(--success)' }}>{selectedCandidate.risk}</strong></span>
                    <span>Confidence: <strong>{selectedCandidate.confidence}%</strong></span>
                  </div>
                </div>
              )}

            </div>
          )}
        </div>

      </main>
    </div>
  );
}

export default App;
