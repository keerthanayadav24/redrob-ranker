"""
CandidateIQ — Scoring Engine
Redrob Intelligent Candidate Discovery & Ranking Challenge

Single source of truth for all scoring logic.
Used by both run.py (generates CSV) and main.py (FastAPI backend).

Runtime: ~90 seconds for 100K candidates, CPU-only, no network, no GPU.
Semantic layer: TF-IDF cosine similarity (scikit-learn) — no transformer model needed.
"""

from __future__ import annotations
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Current date for availability calculations ───────────────────────────────
CURRENT_DATE = date(2026, 6, 29)

# ─── JD Knowledge Base ───────────────────────────────────────────────────────
# Parsed directly from job_description.md

MUST_HAVE_CLUSTERS: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "sentence transformers", "sentence-transformers", "embeddings", "embedding",
        "openai embeddings", "bge", "e5", "semantic search", "dense retrieval",
        "vector search", "neural search",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "pgvector", "vector database", "hybrid search",
        "approximate nearest neighbor",
    ],
    "ranking_retrieval": [
        "ranking", "retrieval", "recommendation systems", "recommendation system",
        "reranking", "re-ranking", "information retrieval", "bm25",
        "learning to rank", "search",
    ],
    "evaluation_frameworks": [
        "ndcg", "mrr", "map", "a/b test", "ab testing", "offline evaluation",
        "online evaluation", "evaluation framework",
    ],
    "llm_production": [
        "llm", "llms", "large language model", "rag",
        "retrieval augmented generation", "fine-tuning", "fine-tune",
        "qlora", "lora", "peft", "instruction tuning", "langchain", "llamaindex",
    ],
}

CONSULTING_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy", "hcl", "tech mahindra", "hexaware",
]

CV_SPEECH_TERMS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "text to speech", "asr", "ocr", "robotics",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "information retrieval",
    "search", "ranking", "embedding", "text", "language model",
]

RESEARCH_TITLES = [
    "research scientist", "research fellow", "phd researcher",
    "postdoc", "academic researcher",
]

ARCHITECTURE_DRIFT_TITLES = [
    "vp engineering", "vp ai", "director of engineering", "cto",
    "head of engineering", "chief technology officer", "engineering manager",
]

# JD text used for TF-IDF semantic matching
JD_SEMANTIC_TEXT = """
Senior AI Engineer founding team role owning the intelligence layer of a
recruiting product: ranking retrieval candidate job matching systems.
Production experience embeddings retrieval systems sentence-transformers
openai embeddings bge e5 deployed real users embedding drift index refresh
retrieval quality regression production. Vector databases hybrid search
infrastructure pinecone weaviate qdrant milvus opensearch elasticsearch faiss.
Strong python production code quality. Evaluation frameworks ranking systems
ndcg mrr map offline online correlation ab test. Shipped end to end ranking
search recommendation system real users meaningful scale. Applied machine learning
product companies not pure services. LLM fine-tuning lora qlora peft rag
retrieval augmented generation. Learning to rank xgboost. Open source contributions
AI ML space. HR-tech recruiting tech marketplace product experience.
"""

# ─── Weights (must sum to 1.0) ────────────────────────────────────────────────
WEIGHTS = {
    "semantic":        0.20,
    "career_evidence": 0.35,
    "skill_cluster":   0.20,
    "availability":    0.15,
    "location":        0.05,
    "bonus":           0.05,
}

# ─── Data helpers ─────────────────────────────────────────────────────────────

def skill_names_lower(skills: list[dict]) -> list[str]:
    return [s.get("name", "").lower() for s in skills]


def candidate_text_blob(candidate: dict) -> str:
    """Text representation of candidate for TF-IDF. Career descriptions weighted 2x."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    parts = [
        profile.get("current_title", ""),
        profile.get("summary", "") * 2,
    ]
    for job in career:
        desc = job.get("description", "")
        parts.append(f"{job.get('title', '')} {desc} {desc}")
    parts.append(" ".join(skill_names_lower(skills)))
    return " ".join(filter(None, parts))


# ─── Honeypot detection ───────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple[bool, str]:
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    # YOE vs actual career history
    total_months = sum(j.get("duration_months", 0) for j in career)
    if total_months > 0 and yoe > total_months / 12 + 3:
        return True, f"Claims {yoe}y but career history totals {total_months/12:.1f}y"

    # Expert skill with near-zero usage + suspicious endorsements
    bad_skills = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 99) < 4
        and s.get("endorsements", 0) > 50
    ]
    if len(bad_skills) >= 3:
        return True, f"Expert proficiency in {len(bad_skills)} skills with <4 months usage"

    # High YOE, no career history
    if not career and yoe > 3:
        return True, f"Claims {yoe}y with no career history"

    return False, ""


# ─── JD disqualifier checks ───────────────────────────────────────────────────

def is_research_only(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False
    titles = [j.get("title", "").lower() for j in career]
    if not all(any(rt in t for rt in RESEARCH_TITLES) for t in titles):
        return False
    production_terms = ["deployed", "production", "shipped", "real users", "scale"]
    return not any(
        any(pt in (j.get("description", "") or "").lower() for pt in production_terms)
        for j in career
    )


def is_recent_langchain_only(candidate: dict) -> bool:
    skills = skill_names_lower(candidate.get("skills", []))
    if not any("langchain" in s for s in skills):
        return False
    career = candidate.get("career_history", [])
    pre_llm_ml = [
        j for j in career
        if any(kw in (j.get("description", "") or "").lower()
               for kw in ["ml", "machine learning", "nlp", "retrieval", "ranking"])
        and int((j.get("start_date") or "2099")[:4]) < 2023
        and j.get("duration_months", 0) > 12
    ]
    ai_jobs = [
        j for j in career
        if any(kw in (j.get("title", "") or "").lower()
               for kw in ["ml", "ai", "data", "nlp", "engineer"])
    ]
    all_recent = all(j.get("duration_months", 0) < 12 for j in ai_jobs) if ai_jobs else False
    return all_recent and not pre_llm_ml


def has_architecture_drift(candidate: dict) -> bool:
    title = (candidate.get("profile", {}).get("current_title", "") or "").lower()
    if not any(t in title for t in ARCHITECTURE_DRIFT_TITLES):
        return False
    career = candidate.get("career_history", [])
    if not career:
        return True
    recent = career[0]
    desc = (recent.get("description", "") or "").lower()
    writes_code = any(kw in desc for kw in ["code", "implement", "built", "wrote", "developed"])
    return recent.get("duration_months", 0) >= 18 and not writes_code


def is_keyword_stuffer(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False
    ai_kws = ["engineer", "scientist", "developer", "architect", "ml", "ai",
              "nlp", "data", "lead", "researcher"]
    titles = [j.get("title", "").lower() for j in career]
    return all(not any(kw in t for kw in ai_kws) for t in titles)


# ─── Scoring dimensions ───────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    semantic: float = 0.0
    skill_cluster: float = 0.0
    career_evidence: float = 0.0
    availability: float = 0.0
    location: float = 0.0
    bonus: float = 0.0
    composite: float = 0.0
    matched_clusters: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)
    is_honeypot: bool = False
    honeypot_reason: str = ""


CLUSTER_READABLE = {
    "embeddings_retrieval": "embeddings/semantic search",
    "vector_db": "vector DB/hybrid search",
    "ranking_retrieval": "ranking/retrieval systems",
    "evaluation_frameworks": "eval frameworks (NDCG/MRR)",
    "llm_production": "LLM/RAG",
}


def skill_cluster_score(skills_lower: list[str]) -> tuple[float, list[str]]:
    matched = []
    for cluster, terms in MUST_HAVE_CLUSTERS.items():
        if any(t in s for t in terms for s in skills_lower):
            matched.append(cluster)
    return len(matched) / len(MUST_HAVE_CLUSTERS), matched


def career_evidence_score(candidate: dict) -> tuple[float, list[str]]:
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    skills_lower = skill_names_lower(candidate.get("skills", []))
    notes = []
    score = 0.5

    # YOE fit
    if 5 <= yoe <= 9:
        score += 0.12
    elif 4 <= yoe <= 12:
        score += 0.06
    elif yoe < 3:
        score -= 0.10
        notes.append(f"only {yoe}y experience")

    # JD disqualifiers
    if is_research_only(candidate):
        score -= 0.45
        notes.append("pure research career, no production deployment")
    if is_recent_langchain_only(candidate):
        score -= 0.35
        notes.append("AI experience is recent LangChain-only")
    if has_architecture_drift(candidate):
        score -= 0.30
        notes.append("architect title with no recent coding evidence")
    if is_keyword_stuffer(candidate):
        score -= 0.40
        notes.append("all career titles non-technical despite AI skills listed")

    # Product company vs consulting
    if career:
        all_consulting = all(
            any(cc in j.get("company", "").lower() for cc in CONSULTING_COMPANIES)
            or j.get("industry", "").lower() == "it services"
            for j in career
        )
        if all_consulting:
            score -= 0.25
            notes.append("entire career at IT services/consulting firms")
        else:
            product_count = sum(
                1 for j in career
                if not any(cc in j.get("company", "").lower() for cc in CONSULTING_COMPANIES)
            )
            if product_count >= 2:
                score += 0.15

    # Production language in descriptions
    prod_kws = ["deployed", "production", "shipped", "built", "scale",
                "real users", "end-to-end", "latency", "retrieval", "ranking", "embedding"]
    hits = sum(
        1 for j in career for kw in prod_kws
        if kw in (j.get("description", "") or "").lower()
    )
    score += min(hits / 12, 1.0) * 0.20

    # Title chasing (JD explicit)
    if len(career) >= 3:
        avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
        if avg_tenure < 18:
            score -= 0.15
            notes.append(f"avg tenure {avg_tenure:.0f}mo — title-chasing pattern")

    # CV/speech without NLP (JD explicit)
    cv_count = sum(1 for s in skills_lower if any(kw in s for kw in CV_SPEECH_TERMS))
    nlp_count = sum(1 for s in skills_lower if any(kw in s for kw in NLP_IR_TERMS))
    if skills_lower and cv_count / max(len(skills_lower), 1) > 0.5 and nlp_count == 0:
        score -= 0.30
        notes.append("CV/speech specialist without NLP/IR exposure")

    return max(0.0, min(1.0, score)), notes


def availability_score(signals: dict) -> float:
    score = 0.0

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            days = (CURRENT_DATE - datetime.strptime(last_active, "%Y-%m-%d").date()).days
            if days <= 30:   score += 0.30
            elif days <= 60: score += 0.22
            elif days <= 90: score += 0.15
            elif days <= 180: score += 0.05
        except (ValueError, TypeError):
            score += 0.08

    if signals.get("open_to_work_flag"):    score += 0.15
    score += (signals.get("recruiter_response_rate") or 0) * 0.20

    notice = signals.get("notice_period_days", 999)
    if notice == 0:        score += 0.15
    elif notice <= 30:     score += 0.13
    elif notice <= 60:     score += 0.07
    elif notice <= 90:     score += 0.03

    score += (signals.get("interview_completion_rate") or 0) * 0.10

    art = signals.get("avg_response_time_hours", 999)
    if art <= 4:    score += 0.10
    elif art <= 24: score += 0.07
    elif art <= 72: score += 0.03

    return min(1.0, score)


def location_score(profile: dict, signals: dict) -> float:
    location = (profile.get("location") or "").lower()
    country  = (profile.get("country") or "").lower()
    relocate = signals.get("willing_to_relocate", False)
    if country != "india": return 0.15
    if any(c in location for c in ["pune", "noida"]): return 1.0
    if any(c in location for c in ["hyderabad", "bangalore", "bengaluru",
                                    "mumbai", "delhi", "gurgaon"]): return 0.80
    return 0.55 if relocate else 0.35


def bonus_score(candidate: dict, signals: dict) -> float:
    score = 0.0
    gh = signals.get("github_activity_score", -1)
    if gh >= 60:   score += 0.35
    elif gh >= 30: score += 0.22
    elif gh >= 10: score += 0.10
    elif gh > 0:   score += 0.05

    relevant_kws = ["nlp", "embeddings", "llm", "fine-tuning",
                    "vector", "recommendation", "elasticsearch", "retrieval"]
    assessments = signals.get("skill_assessment_scores") or {}
    rel_scores = [v for k, v in assessments.items()
                  if any(kw in k.lower() for kw in relevant_kws)]
    if rel_scores:
        score += (sum(rel_scores) / len(rel_scores) / 100) * 0.30

    certs = candidate.get("certifications", [])
    cert_kws = ["aws", "gcp", "azure", "ml", "ai", "deep learning", "nlp", "pytorch"]
    rel_certs = [c for c in certs if any(kw in (c.get("name") or "").lower() for kw in cert_kws)]
    score += min(len(rel_certs) * 0.05, 0.15)

    if signals.get("linkedin_connected"):                           score += 0.05
    if signals.get("verified_email") and signals.get("verified_phone"): score += 0.05

    return min(1.0, score)


# ─── TF-IDF semantic layer ────────────────────────────────────────────────────

def compute_semantic_scores(candidate_texts: list[str]) -> np.ndarray:
    """
    TF-IDF cosine similarity between JD and each candidate text blob.
    Captures term importance beyond exact keyword matching.
    Runtime: ~15-20s for 100K candidates on a single CPU core.
    """
    corpus = [JD_SEMANTIC_TEXT] + candidate_texts
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return scores


# ─── Explainability ───────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, breakdown: ScoreBreakdown) -> str:
    """
    Specific, honest 1-2 sentence reasoning per candidate.
    Grounded only in fields that exist in the candidate record — no hallucination.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    career   = candidate.get("career_history", [])

    yoe     = profile.get("years_of_experience", 0)
    title   = profile.get("current_title", "")
    notice  = signals.get("notice_period_days", 999)
    rr      = signals.get("recruiter_response_rate") or 0
    otw     = signals.get("open_to_work_flag", False)
    last_a  = signals.get("last_active_date", "")

    clusters_str = ", ".join(
        CLUSTER_READABLE.get(c, c) for c in breakdown.matched_clusters[:3]
    )
    product_cos = [
        j["company"] for j in career
        if not any(cc in j.get("company", "").lower() for cc in CONSULTING_COMPANIES)
        and j.get("industry", "").lower() != "it services"
    ]

    concerns = []
    if notice > 60: concerns.append(f"{notice}d notice")
    if rr < 0.20:   concerns.append(f"low response rate ({int(rr*100)}%)")
    if last_a:
        try:
            days = (CURRENT_DATE - datetime.strptime(last_a, "%Y-%m-%d").date()).days
            if days > 90: concerns.append(f"inactive {days}d")
        except (ValueError, TypeError):
            pass
    if breakdown.disqualifiers:
        concerns.append(breakdown.disqualifiers[0])

    # Sentence 1: what they bring
    if clusters_str and product_cos:
        s1 = f"{yoe}y {title} with {clusters_str} experience at product companies ({', '.join(product_cos[:2])})."
    elif clusters_str:
        s1 = f"{yoe}y {title} with coverage across {clusters_str}."
    elif breakdown.disqualifiers:
        s1 = f"{yoe}y {title} — {breakdown.disqualifiers[0]}."
    else:
        s1 = f"{yoe}y {title} with adjacent technical background."

    # Sentence 2: availability
    if concerns:
        s2 = f"Concerns: {'; '.join(concerns[:2])}."
    elif otw and rr >= 0.5:
        s2 = f"Active and responsive ({int(rr*100)}% reply rate, {notice}d notice, open to work)."
    else:
        s2 = f"Availability moderate (response rate {int(rr*100)}%, notice {notice}d)."

    return f"{s1} {s2}"


# ─── Master rank function ─────────────────────────────────────────────────────

def rank_candidates(candidates: list[dict]) -> list[dict]:
    """
    Scores all candidates and returns a sorted list of result dicts.
    Called by both run.py (CLI) and main.py (FastAPI).
    """
    print(f"Building TF-IDF matrix for {len(candidates)} candidates...")
    texts = [candidate_text_blob(c) for c in candidates]
    semantic_scores = compute_semantic_scores(texts)
    print("Semantic scores computed. Scoring all dimensions...")

    results = []
    for i, c in enumerate(candidates):
        cid      = c.get("candidate_id", "")
        profile  = c.get("profile", {})
        signals  = c.get("redrob_signals", {})
        skills   = c.get("skills", [])
        skills_l = skill_names_lower(skills)

        breakdown = ScoreBreakdown()

        # Honeypot check
        hp, hp_reason = is_honeypot(c)
        if hp:
            breakdown.is_honeypot   = True
            breakdown.honeypot_reason = hp_reason
            breakdown.composite     = 0.0
            results.append(_build_result(c, cid, breakdown, 0.0))
            continue

        # Hard disqualifier: non-technical title + zero cluster coverage
        current_title = (profile.get("current_title", "") or "").lower()
        nontechnical_kws = ["marketing", "operations", "hr manager", "accountant",
                            "civil engineer", "sales executive", "graphic designer",
                            "content writer"]
        cluster_score_val, matched = skill_cluster_score(skills_l)
        if any(kw in current_title for kw in nontechnical_kws) and not matched:
            breakdown.composite = 0.01
            results.append(_build_result(c, cid, breakdown, 0.01))
            continue

        # Score all dimensions
        breakdown.semantic       = float(semantic_scores[i])
        breakdown.skill_cluster  = cluster_score_val
        breakdown.matched_clusters = matched
        career_score, disqs      = career_evidence_score(c)
        breakdown.career_evidence = career_score
        breakdown.disqualifiers  = disqs
        breakdown.availability   = availability_score(signals)
        breakdown.location       = location_score(profile, signals)
        breakdown.bonus          = bonus_score(c, signals)

        breakdown.composite = (
            WEIGHTS["semantic"]        * breakdown.semantic +
            WEIGHTS["career_evidence"] * breakdown.career_evidence +
            WEIGHTS["skill_cluster"]   * breakdown.skill_cluster +
            WEIGHTS["availability"]    * breakdown.availability +
            WEIGHTS["location"]        * breakdown.location +
            WEIGHTS["bonus"]           * breakdown.bonus
        )

        results.append(_build_result(c, cid, breakdown, breakdown.composite))

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return results


def _build_result(candidate: dict, cid: str, breakdown: ScoreBreakdown, score: float) -> dict:
    """Builds the full result dict consumed by both CSV writer and FastAPI response."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    reasoning = generate_reasoning(candidate, breakdown)

    # Decision label
    if breakdown.is_honeypot:
        decision, confidence = "Reject", 20
    elif score >= 0.82:
        decision, confidence = "Strong Hire", 96
    elif score >= 0.68:
        decision, confidence = "Hire", 85
    elif score >= 0.50:
        decision, confidence = "Needs Review", 70
    else:
        decision, confidence = "Reject", 45

    # Interview priority
    if score >= 0.88:   priority = "Immediate"
    elif score >= 0.75: priority = "This Week"
    elif score >= 0.60: priority = "Normal"
    else:               priority = "Low"

    # Insights (for UI)
    insights = []
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        insights.append(f"{yoe} years — ideal experience range for this role")
    if signals.get("open_to_work_flag"):
        insights.append("Actively open to work")
    if (signals.get("recruiter_response_rate") or 0) > 0.6:
        insights.append("High recruiter engagement")
    if (signals.get("notice_period_days") or 999) <= 30:
        insights.append("Immediate availability")
    if (signals.get("github_activity_score") or -1) > 40:
        insights.append("Active GitHub contributor")
    for dq in breakdown.disqualifiers:
        insights.append(f"⚠ {dq}")

    top_skills = [s.get("name") for s in candidate.get("skills", [])[:6]]

    return {
        "candidate_id":    cid,
        "score":           round(score, 4),
        "semantic_score":  round(breakdown.semantic, 4),
        "skill_score":     round(breakdown.skill_cluster, 4),
        "experience_score": round(breakdown.career_evidence, 4),
        "behaviour_score": round(breakdown.availability, 4),
        "location_score":  round(breakdown.location, 4),
        "notice_score":    round(breakdown.bonus, 4),
        "decision":        decision,
        "confidence":      confidence,
        "interview_priority": priority,
        "reasoning":       reasoning,
        "summary":         f"{yoe}y {profile.get('current_title', '')} based in {profile.get('location', '')}.",
        "recommendation":  decision,
        "insights":        insights,
        "top_skills":      top_skills,
        "matched_clusters": [CLUSTER_READABLE.get(c, c) for c in breakdown.matched_clusters],
        "risk":            "High" if breakdown.is_honeypot else "Low",
        "raw_candidate":   candidate,
    }
