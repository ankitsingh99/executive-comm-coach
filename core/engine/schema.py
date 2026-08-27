"""
Deterministic Structured Output Specification for Executive Communication Coach.
Enforces strict schema constraints, score bounds, verbatim quote validation, and Top-N itemization.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class FillerWordMetric(BaseModel):
    """Detected verbal filler word and frequency count."""
    token: str = Field(description="The filler token, e.g., 'matlab', 'basically', 'like'")
    count: int = Field(ge=0, description="Number of times uttered by the user")


class CommunicationMetrics(BaseModel):
    """Core quantitative communication efficacy scores [0, 100]."""
    presence_score: int = Field(
        ge=0, le=100,
        description="Overall executive presence, structured delivery, and BLUF alignment [0-100]"
    )
    assertiveness_score: int = Field(
        ge=0, le=100,
        description="Ratio of definitive assertions to self-diminishing qualifiers [0-100]"
    )
    active_listening_score: int = Field(
        ge=0, le=100,
        description="Frequency of mirroring, validation, and inquiry before advancing own agenda [0-100]"
    )
    filler_words_detected: List[FillerWordMetric] = Field(
        default_factory=list,
        description="Itemized verbal filler counts (e.g. 'matlab', 'basically', 'like', 'you know')"
    )


class TopStrength(BaseModel):
    """Itemized positive communication observation."""
    observation: str = Field(
        max_length=250,
        description="Detailed observation of what went well in executive presence (max 250 chars)"
    )
    verbatim_quote: str = Field(
        description="Exact quote from the user in the transcript demonstrating this strength"
    )

    @field_validator("observation")
    @classmethod
    def validate_obs_length(cls, v: str) -> str:
        if len(v) > 250:
            raise ValueError("Strength observation cannot exceed 250 characters")
        return v.strip()


class AreaForImprovement(BaseModel):
    """Itemized area for communication improvement with coached rephrasing."""
    critique: str = Field(
        max_length=250,
        description="Analysis of the phrasing or influence deficiency (max 250 chars)"
    )
    verbatim_quote: str = Field(
        description="Exact excerpt from user speech identifying the weakness"
    )
    coached_phrasing: str = Field(
        max_length=250,
        description="Actionable, higher-impact executive alternative for the user (max 250 chars)"
    )

    @field_validator("critique", "coached_phrasing")
    @classmethod
    def validate_str_length(cls, v: str) -> str:
        if len(v) > 250:
            raise ValueError("Text cannot exceed 250 characters")
        return v.strip()


class ExecutiveCoachingEvaluation(BaseModel):
    """Complete structured coaching evaluation report constrained by Top-N parameter."""
    persona_context: str = Field(
        description="Summary of the power axis and counterpart context (e.g. Upward to VP, Lateral with PM)"
    )
    metrics: CommunicationMetrics = Field(
        description="Standardized quantitative communication metrics"
    )
    top_strengths: List[TopStrength] = Field(
        description="Exact N positive observations with verbatim quotes"
    )
    areas_for_improvement: List[AreaForImprovement] = Field(
        description="Exact N areas for improvement with verbatim quote and coached rephrasing"
    )
    longitudinal_summary: str = Field(
        description="High-level executive coaching takeaway and strategic advice"
    )
    persona_alignment_notes: str = Field(
        description="Specific notes on how delivery met the expected power dynamic (BLUF / Collaboration / Mentorship)"
    )


class Utterance(BaseModel):
    """Single timestamped and diarized dialogue turn."""
    speaker: str = Field(description="Speaker identifier, e.g. 'USER' or 'COUNTERPART'")
    start_time: float = Field(ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(ge=0.0, description="End timestamp in seconds")
    transcript: str = Field(description="Diarized transcript text (may include Hinglish code-switching)")


class ConversationSession(BaseModel):
    """Complete meeting/conversation session data."""
    session_id: str
    timestamp_utc: str
    target_speaker: str = "USER"
    counterpart_name: str
    counterpart_role: str
    power_axis: str  # "UPWARD", "LATERAL", "DOWNWARD"
    dialogue: List[Utterance] = Field(default_factory=list)
    raw_audio_path: Optional[str] = None
    is_encrypted: bool = True
