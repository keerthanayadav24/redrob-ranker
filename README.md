# Redrob AI Candidate Ranker

An AI-powered recruitment platform that intelligently ranks candidates based on a job description using structured scoring, resume analysis, and explainable AI.

---

## Features

- AI-powered candidate ranking
- Resume parsing and analysis
- Skill matching
- Experience scoring
- Explainable ranking reasons
- Interactive recruiter dashboard
- Top 100 candidate generation
- Submission CSV generation

---

## Tech Stack

### Frontend
- React (Vite)
- CSS
- JavaScript

### Backend
- FastAPI
- Python

### AI
- Custom scoring engine
- Resume feature extraction
- Explainable ranking model

---

## Project Structure

```
backend/
frontend/
data/
run.py
requirements.txt
validate_submission.py
```

---

## Installation

### Backend

```bash
pip install -r requirements.txt
python backend/main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Generate Rankings

```bash
python run.py
```

---

## Validate Submission

```bash
python validate_submission.py data/team_keerthana.csv
```

---

## Dataset

The original candidate dataset is not included in this repository because it exceeds GitHub's file size limit.

Place the dataset inside:

```
data/candidates.jsonl
```

before running the ranking pipeline.

---

## Submission Output

The generated submission is available in:

```
data/team_keerthana.csv
```

---

## Authors

Keerthana S Yadav
