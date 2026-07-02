from pydantic import BaseModel
from typing import List, Optional, Dict


class Profile(BaseModel):
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


class CareerHistory(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: Optional[str] = None


class Skill(BaseModel):
    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int] = None


class Certification(BaseModel):
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    language: str
    proficiency: str


class ExpectedSalary(BaseModel):
    min: float
    max: float


class RedrobSignals(BaseModel):
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float] = {}
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: ExpectedSalary
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


class Candidate(BaseModel):
    candidate_id: str
    profile: Profile
    career_history: List[CareerHistory]
    education: List[Education] = []
    skills: List[Skill] = []
    certifications: List[Certification] = []
    languages: List[Language] = []
    redrob_signals: RedrobSignals
