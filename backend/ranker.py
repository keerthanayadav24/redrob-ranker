import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Cache candidate embeddings
CACHED_EMBEDDINGS = None
CACHED_IDS = None

# Load the model globally so it's only loaded once
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    # If network is unavailable and not cached, this will fail.
    # The hackathon requires no network *during* ranking, so the model must be downloaded beforehand.
    model = None


def get_jd_embedding(jd_text: str):
    if model:
        return model.encode(jd_text, convert_to_tensor=False)
    return None


def extract_candidate_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    text_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", "")
    ]

    # Add recent experience
    for exp in candidate.get("career_history", [])[:2]:
        text_parts.append(exp.get("title", ""))
        text_parts.append(exp.get("description", ""))

    # Add top skills
    skills = candidate.get("skills", [])
    skill_names = [s.get("name", "") for s in skills[:5]]
    text_parts.extend(skill_names)

    certifications = " ".join(
    cert.get("name", "")
    for cert in candidate.get("certifications", [])
    )

    return " ".join(text_parts)

def build_candidate_embeddings(candidates):
    global CACHED_EMBEDDINGS
    global CACHED_IDS

    # Already built
    if CACHED_EMBEDDINGS is not None:
        return CACHED_EMBEDDINGS

    print("\nBuilding candidate embeddings... (only once)")

    texts = [
        extract_candidate_text(c)
        for c in candidates
    ]

    CACHED_EMBEDDINGS = model.encode(
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    CACHED_IDS = [
        c["candidate_id"]
        for c in candidates
    ]

    print("Embeddings cached.")

    return CACHED_EMBEDDINGS

import re

def normalize(text):
    if not text:
        return ""
    return text.lower().strip()


def extract_years_from_jd(jd_text):
    match = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*years', jd_text.lower())
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r'(\d+)\+?\s*years', jd_text.lower())
    if match:
        y = int(match.group(1))
        return y, y + 2

    return None, None

def skill_match_score(candidate, jd_text):
    """
    JD-aware skill matching based on Redrob's AI Engineer JD.
    Gives higher weight to must-have production skills.
    """

    candidate_skills = {
        s.get("name", "").lower().strip()
        for s in candidate.get("skills", [])
    }

    profile = candidate.get("profile", {})
    profile_text = (
        profile.get("headline", "") + " " +
        profile.get("summary", "")
    ).lower()

    career_text = " ".join(
        job.get("description", "").lower()
        for job in candidate.get("career_history", [])
    )

    searchable = (
        " ".join(candidate_skills)
        + " "
        + profile_text
        + " "
        + career_text
    )

    # -----------------------------
    # Must-have skills (High Weight)
    # -----------------------------
    must_have = {
        "python": 10,
        "embedding": 10,
        "embeddings": 10,
        "retrieval": 10,
        "ranking": 9,
        "sentence transformer": 8,
        "sentence-transformers": 8,
        "vector database": 8,
        "qdrant": 8,
        "faiss": 8,
        "milvus": 8,
        "pinecone": 8,
        "weaviate": 8,
        "elasticsearch": 7,
        "opensearch": 7,
        "llm": 8,
        "fine tuning": 8,
        "fine-tuning": 8,
        "rag": 8,
        "evaluation": 7,
        "ndcg": 7,
        "mrr": 7,
        "map": 7,
    }

    # -----------------------------
    # Nice-to-have
    # -----------------------------
    nice_to_have = {
        "langchain": 3,
        "docker": 3,
        "kubernetes": 3,
        "airflow": 2,
        "spark": 2,
        "aws": 2,
        "gcp": 2,
        "mlflow": 2,
        "huggingface": 3,
        "transformers": 3,
        "pytorch": 3,
        "tensorflow": 3,
    }

    score = 0
    max_score = 0

    for skill, weight in must_have.items():
        max_score += weight
        if skill in searchable:
            score += weight

    for skill, weight in nice_to_have.items():
        max_score += weight
        if skill in searchable:
            score += weight

    return score / max_score

def experience_score(candidate, jd_text):

    min_exp, max_exp = extract_years_from_jd(jd_text)

    years = candidate["profile"].get(
        "years_of_experience",
        0
    )

    if min_exp is None:
        return 1

    if min_exp <= years <= max_exp:
        return 1

    if years < min_exp:
        return max(0, years / min_exp)

    return 0.9

def behaviour_score(candidate):

    sig = candidate.get("redrob_signals", {})

    score = 0

    score += sig.get("profile_completeness_score", 0) /100

    score += sig.get("recruiter_response_rate", 0)

    score += 0.2 if sig.get("open_to_work_flag") else 0

    score /= 2.2

    return score

def location_score(candidate, jd_text):

    location = normalize(
        candidate["profile"].get("location", "")
    )

    jd = normalize(jd_text)

    if location in jd:
        return 1

    return 0.5

def notice_score(candidate):

    notice = candidate.get(
        "redrob_signals",
        {}
    ).get(
        "notice_period_days",
        90
    )

    if notice <= 30:
        return 1

    if notice <= 60:
        return 0.8

    if notice <= 90:
        return 0.6

    return 0.3



from datetime import datetime

def is_honeypot(candidate):
    """
    Detect suspicious/fake candidate profiles.
    Returns True if profile looks unrealistic.
    """

    score = 0

    # 1. Expert skill with almost no experience

    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)

        if prof in ["expert", "advanced"] and duration <= 3:
            score += 2

    # 2. Unrealistic years of experience

    profile = candidate.get("profile", {})
    years = profile.get("years_of_experience", 0)

    if years > 35:
        score += 3

    
    # 3. Overlapping jobs

    jobs = candidate.get("career_history", [])

    parsed = []

    for job in jobs:

        try:
            start = datetime.strptime(
                job["start_date"],
                "%Y-%m-%d"
            )

            end = job.get("end_date")

            if end:
                end = datetime.strptime(end, "%Y-%m-%d")
            else:
                end = datetime.today()

            parsed.append((start, end))

        except:
            continue

    parsed.sort()

    for i in range(1, len(parsed)):

        previous_end = parsed[i - 1][1]
        current_start = parsed[i][0]

        if current_start < previous_end:
            score += 3

    
    # 4. Keyword stuffing

    searchable = (
        profile.get("summary", "")
        + " "
        + profile.get("headline", "")
    ).lower()

    keywords = [
        "llm",
        "rag",
        "embedding",
        "embeddings",
        "retrieval",
        "ranking",
        "langchain",
        "qdrant",
        "faiss",
        "vector"
    ]

    keyword_hits = sum(
        searchable.count(k)
        for k in keywords
    )

    if keyword_hits > 20:
        score += 3

    
    # 5. Very low profile quality

    signals = candidate.get("redrob_signals", {})

    if (
        signals.get("profile_completeness_score", 100) < 15
        and
        signals.get("recruiter_response_rate", 1) < 0.05
    ):
        score += 2

    # Final Decision

    return score >= 5


def calculate_behavioral_multiplier(signals: Dict[str, Any]) -> float:
    """Calculate a multiplier based on user engagement signals."""
    if not signals:
        return 0.5

    mult = 1.0

    # Recruiter response rate (0 to 1) -> up to +0.2
    rr = signals.get("recruiter_response_rate", 0)
    mult += (rr * 0.2)

    # Inactive penalty
    # We can't strictly parse dates without datetime easily, but we know if
    # they are open to work
    if signals.get("open_to_work_flag"):
        mult += 0.1

    # Completeness
    completeness = signals.get("profile_completeness_score", 0)
    mult += (completeness / 100.0) * 0.1

    return mult


def generate_reasoning(candidate, semantic_score, is_trap, behaviour):

    if is_trap:
        return (
            "⚠ Potential profile anomaly detected. "
            "Ranking reduced because of suspicious experience or skill patterns."
        )

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    strengths = []
    concerns = []

    # Experience

    yoe = profile.get("years_of_experience", 0)

    if yoe >= 5:
        strengths.append(f"{yoe} years of relevant experience")
    else:
        concerns.append("Below preferred experience")

    
    # Skills
    
    searchable = (
        profile.get("headline", "")
        + " "
        + profile.get("summary", "")
        + " "
        + " ".join(
            s.get("name", "")
            for s in candidate.get("skills", [])
        )
    ).lower()

    important = [
        "python",
        "embeddings",
        "retrieval",
        "ranking",
        "qdrant",
        "faiss",
        "milvus",
        "pinecone",
        "vector",
        "llm",
        "rag",
        "transformers",
    ]

    matched = []

    for skill in important:
        if skill in searchable:
            matched.append(skill)

    if matched:
        strengths.append(
            "Skills: " + ", ".join(matched[:4])
        )
    else:
        concerns.append("Missing core AI ranking skills")

    
    # Behaviour
   
    if signals.get("open_to_work_flag"):
        strengths.append("Open to work")

    response = signals.get("recruiter_response_rate", 0)

    if response >= 0.5:
        strengths.append("High recruiter response rate")

    notice = signals.get("notice_period_days", 90)

    if notice > 90:
        concerns.append(f"{notice}-day notice period")

    
    # Semantic Fit
    
    if semantic_score > 0.70:
        verdict = "Excellent match."
    elif semantic_score > 0.55:
        verdict = "Strong match."
    elif semantic_score > 0.40:
        verdict = "Moderate match."
    else:
        verdict = "Limited match."

    reason = verdict

    if strengths:
        reason += " Strengths: " + "; ".join(strengths[:3]) + "."

    if concerns:
        reason += " Concerns: " + "; ".join(concerns[:2]) + "."

    return reason

def get_candidate_insights(candidate, final_score):

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    insights = []

    # Experience
    yoe = profile.get("years_of_experience", 0)

    if yoe >= 5:
        insights.append("✔ Experience matches senior role")
    else:
        insights.append("⚠ Experience below preferred range")

    # Open to work
    if signals.get("open_to_work_flag"):
        insights.append("✔ Actively open to work")

    # Recruiter response
    rr = signals.get("recruiter_response_rate", 0)

    if rr > 0.6:
        insights.append("✔ High recruiter engagement")

    # Notice
    notice = signals.get("notice_period_days", 90)

    if notice <= 30:
        insights.append("✔ Immediate availability")
    elif notice > 90:
        insights.append("⚠ Long notice period")

    # Recommendation
    if final_score >= 0.80:
        verdict = "Strong Hire"

    elif final_score >= 0.65:
        verdict = "Hire"

    else:
        verdict = "Needs Review"

    return {
        "verdict": verdict,
        "insights": insights
    }

def get_decision_score(final_score, trap):

    if trap:
        return {
            "decision": "Reject",
            "color": "red",
            "confidence": 20
        }

    if final_score >= 0.85:
        return {
            "decision": "Strong Hire",
            "color": "green",
            "confidence": 98
        }

    if final_score >= 0.70:
        return {
            "decision": "Hire",
            "color": "lime",
            "confidence": 88
        }

    if final_score >= 0.55:
        return {
            "decision": "Needs Review",
            "color": "yellow",
            "confidence": 72
        }

    return {
        "decision": "Reject",
        "color": "red",
        "confidence": 40
    }

def recruiter_summary(candidate):

    profile = candidate.get("profile", {})

    return (
        f"{profile.get('current_title')} with "
        f"{profile.get('years_of_experience')} years experience."
    )

def interview_priority(score):

    if score >= 0.90:
        return "Immediate"

    if score >= 0.80:
        return "This Week"

    if score >= 0.70:
        return "Normal"

    return "Low"

def rank_candidates(
        candidates: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
    jd_emb = get_jd_embedding(jd_text)

    # Compute embeddings in batches
    if model and jd_emb is not None:
        cand_embs = build_candidate_embeddings(candidates)

        # Cosine similarity
        jd_norm = jd_emb / np.linalg.norm(jd_emb)
        cand_norms = cand_embs / \
            np.linalg.norm(cand_embs, axis=1, keepdims=True)
        similarities = np.dot(cand_norms, jd_norm)
    else:
        # Fallback if model fails (e.g., no internet to download model)
        # Just use basic keyword overlap for fallback
        jd_words = set(jd_text.lower().split())
        similarities = []
        for c in candidates:
            t = extract_candidate_text(c)
            t_words = set(t.lower().split())
            overlap = len(jd_words.intersection(t_words))
            similarities.append(overlap / (len(t_words) + 1))

# Get Top 1000 Semantic Matches
        TOP_K = 1000

        top_indices = np.argsort(similarities)[::-1][:TOP_K]

        results = []

        for i in top_indices:

            c = candidates[i]

            sem_score = float(similarities[i])

                # Check honeypot
        trap = is_honeypot(c)

        # Calculate all scores
        semantic = sem_score
        skill = skill_match_score(c, jd_text)
        experience = experience_score(c, jd_text)
        behaviour = behaviour_score(c)
        location = location_score(c, jd_text)
        notice = notice_score(c)

        # Weighted final score
        final_score = (
        semantic * 0.40 +
        skill * 0.25 +
        experience * 0.15 +
        behaviour * 0.10 +
        location * 0.05 +
        notice * 0.05
       )

        # Honeypot penalty
        if trap:
            final_score *= 0.30

        # Keep this variable name because generate_reasoning() expects it
        mult = behaviour

        reasoning = generate_reasoning(c, semantic, trap, mult)

        skills = [
            s.get("name")
            for s in c.get("skills", [])
        ][:5]

        required = [
            "Python",
            "Embeddings",
            "Retrieval",
            "Ranking",
            "FAISS",
            "Qdrant"
        ]

        candidate_skill_names = {
            s.get("name","").lower()
            for s in c.get("skills",[])
        }

        missing = [
            skill
            for skill in required
            if skill.lower() not in candidate_skill_names
        ]

        insight = get_candidate_insights(
            c,
            final_score
        )

        decision = get_decision_score(
            final_score,
            trap
        )

        summary = recruiter_summary(c)

        priority = interview_priority(final_score)

        results.append ({
            "candidate_id": c.get("candidate_id"),

            "score": round(final_score, 4),

            "semantic_score": round(semantic, 4),

            "skill_score": round(skill, 4),

            "experience_score": round(experience, 4),

            "behaviour_score": round(behaviour, 4),

            "location_score": round(location, 4),

            "notice_score": round(notice, 4),

            "top_skills": skills,

            "risk": (
                "High"
                if trap
                else "Low"
            ),

            "recommendation": (
                "Strong Hire"
                if final_score > 0.80
                else
                "Hire"
                if final_score > 0.65
                else
                "Maybe"
            ),

            "reasoning": reasoning,

            "raw_candidate": c,

            "missing_skills": missing,

            "verdict": insight["verdict"],

            "insights": insight["insights"],

            "decision": decision["decision"],

            "confidence": decision["confidence"],

            "decision_color": decision["color"],

            "summary": summary,

            "interview_priority": priority,
        }
       )
        
    # Sort
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
