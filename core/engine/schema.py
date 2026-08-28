"""
Deterministic Structured Output Specification for Executive Communication Coach.
Enforces strict schema constraints, score bounds, and Top-N itemization.
Supports both Pydantic and standard library dataclasses for universal portability.
"""

from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Portable fallback using standard dataclasses
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    res[k] = v.model_dump()
                elif isinstance(v, list):
                    res[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
                else:
                    res[k] = v
            return res

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            return cls(**data)

    def Field(*args, **kwargs):
        return kwargs.get("default", None)

    def field_validator(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


class FillerWordMetric(BaseModel):
    """Detected verbal filler word and frequency count."""
    token: str = ""
    count: int = 0

    def __init__(self, token: str = "", count: int = 0, **kwargs):
        super().__init__(token=token, count=count, **kwargs)
        self.token = token
        self.count = count


class SpeakerAcousticProfile(BaseModel):
    """Acoustic voice characteristics and tone classification for an individual speaker."""
    speaker_id: str = "SPEAKER_01"
    mean_pitch_hz: float = 0.0
    pitch_range_hz: float = 0.0
    energy_rms: float = 0.0
    speech_rate_wpm: float = 0.0
    tone_label: str = "Calm & Measured"
    talk_time_percentage: float = 100.0
    confidence_score: float = 1.0

    def __init__(
        self,
        speaker_id: str = "SPEAKER_01",
        mean_pitch_hz: float = 0.0,
        pitch_range_hz: float = 0.0,
        energy_rms: float = 0.0,
        speech_rate_wpm: float = 0.0,
        tone_label: str = "Calm & Measured",
        talk_time_percentage: float = 100.0,
        confidence_score: float = 1.0,
        **kwargs
    ):
        super().__init__(
            speaker_id=speaker_id,
            mean_pitch_hz=mean_pitch_hz,
            pitch_range_hz=pitch_range_hz,
            energy_rms=energy_rms,
            speech_rate_wpm=speech_rate_wpm,
            tone_label=tone_label,
            talk_time_percentage=talk_time_percentage,
            confidence_score=confidence_score,
            **kwargs
        )
        self.speaker_id = speaker_id
        self.mean_pitch_hz = float(mean_pitch_hz)
        self.pitch_range_hz = float(pitch_range_hz)
        self.energy_rms = float(energy_rms)
        self.speech_rate_wpm = float(speech_rate_wpm)
        self.tone_label = tone_label
        self.talk_time_percentage = float(talk_time_percentage)
        self.confidence_score = float(confidence_score)


class AcousticAnalysisResult(BaseModel):
    """Aggregate acoustic voice and speaker detection analysis."""
    detected_speaker_count: int = 1
    is_multi_speaker: bool = False
    speakers: List[SpeakerAcousticProfile] = []
    overall_tone: str = "Calm & Measured"
    turn_taking_events: int = 0

    def __init__(
        self,
        detected_speaker_count: int = 1,
        is_multi_speaker: bool = False,
        speakers: Optional[List[SpeakerAcousticProfile]] = None,
        overall_tone: str = "Calm & Measured",
        turn_taking_events: int = 0,
        **kwargs
    ):
        super().__init__(
            detected_speaker_count=detected_speaker_count,
            is_multi_speaker=is_multi_speaker,
            speakers=speakers or [],
            overall_tone=overall_tone,
            turn_taking_events=turn_taking_events,
            **kwargs
        )
        self.detected_speaker_count = int(detected_speaker_count)
        self.is_multi_speaker = bool(is_multi_speaker)
        self.speakers = speakers or []
        self.overall_tone = overall_tone
        self.turn_taking_events = int(turn_taking_events)


class CommunicationMetrics(BaseModel):
    """Core quantitative communication efficacy scores [0, 100]."""
    presence_score: int = 0
    assertiveness_score: int = 0
    active_listening_score: int = 0
    filler_words_detected: List[FillerWordMetric] = []
    acoustic_analysis: Optional[AcousticAnalysisResult] = None

    def __init__(
        self,
        presence_score: int = 0,
        assertiveness_score: int = 0,
        active_listening_score: int = 0,
        filler_words_detected: Optional[List[FillerWordMetric]] = None,
        **kwargs
    ):
        super().__init__(
            presence_score=presence_score,
            assertiveness_score=assertiveness_score,
            active_listening_score=active_listening_score,
            filler_words_detected=filler_words_detected or [],
            **kwargs
        )
        self.presence_score = max(0, min(100, int(presence_score)))
        self.assertiveness_score = max(0, min(100, int(assertiveness_score)))
        self.active_listening_score = max(0, min(100, int(active_listening_score)))
        self.filler_words_detected = filler_words_detected or []


class TopStrength(BaseModel):
    """Itemized positive communication observation."""
    observation: str = ""
    verbatim_quote: str = ""

    def __init__(self, observation: str = "", verbatim_quote: str = "", **kwargs):
        super().__init__(observation=observation[:250], verbatim_quote=verbatim_quote, **kwargs)
        self.observation = observation[:250]
        self.verbatim_quote = verbatim_quote


class AreaForImprovement(BaseModel):
    """Itemized area for communication improvement with coached rephrasing."""
    critique: str = ""
    verbatim_quote: str = ""
    coached_phrasing: str = ""

    def __init__(
        self,
        critique: str = "",
        verbatim_quote: str = "",
        coached_phrasing: str = "",
        **kwargs
    ):
        super().__init__(
            critique=critique[:250],
            verbatim_quote=verbatim_quote,
            coached_phrasing=coached_phrasing[:250],
            **kwargs
        )
        self.critique = critique[:250]
        self.verbatim_quote = verbatim_quote
        self.coached_phrasing = coached_phrasing[:250]


class ActionItem(BaseModel):
    """Extracted action item, commitment, or scheduled follow-up from spoken dialogue."""
    owner: str = "USER"
    task: str = ""
    due_time_or_date: Optional[str] = None
    verbatim_quote: str = ""
    category: str = "Follow-up"
    urgency: str = "Normal"

    def __init__(
        self,
        owner: str = "USER",
        task: str = "",
        due_time_or_date: Optional[str] = None,
        verbatim_quote: str = "",
        category: str = "Follow-up",
        urgency: str = "Normal",
        **kwargs
    ):
        super().__init__(
            owner=owner,
            task=task[:250],
            due_time_or_date=due_time_or_date,
            verbatim_quote=verbatim_quote,
            category=category,
            urgency=urgency,
            **kwargs
        )
        self.owner = owner
        self.task = task[:250]
        self.due_time_or_date = due_time_or_date
        self.verbatim_quote = verbatim_quote
        self.category = category
        self.urgency = urgency


class ExecutiveCoachingEvaluation(BaseModel):
    """Complete structured coaching evaluation report constrained by Top-N parameter."""
    persona_context: str = ""
    metrics: CommunicationMetrics = None
    top_strengths: List[TopStrength] = []
    areas_for_improvement: List[AreaForImprovement] = []
    action_items: List[ActionItem] = []
    longitudinal_summary: str = ""
    persona_alignment_notes: str = ""

    def __init__(
        self,
        persona_context: str = "",
        metrics: Optional[CommunicationMetrics] = None,
        top_strengths: Optional[List[TopStrength]] = None,
        areas_for_improvement: Optional[List[AreaForImprovement]] = None,
        action_items: Optional[List[ActionItem]] = None,
        longitudinal_summary: str = "",
        persona_alignment_notes: str = "",
        **kwargs
    ):
        super().__init__(
            persona_context=persona_context,
            metrics=metrics,
            top_strengths=top_strengths or [],
            areas_for_improvement=areas_for_improvement or [],
            action_items=action_items or [],
            longitudinal_summary=longitudinal_summary,
            persona_alignment_notes=persona_alignment_notes,
            **kwargs
        )
        self.persona_context = persona_context
        self.metrics = metrics or CommunicationMetrics()
        self.top_strengths = top_strengths or []
        self.areas_for_improvement = areas_for_improvement or []
        self.action_items = action_items or []
        self.longitudinal_summary = longitudinal_summary
        self.persona_alignment_notes = persona_alignment_notes


class Utterance(BaseModel):
    """Single timestamped and diarized dialogue turn."""
    speaker: str = "USER"
    start_time: float = 0.0
    end_time: float = 0.0
    transcript: str = ""

    def __init__(
        self,
        speaker: str = "USER",
        start_time: float = 0.0,
        end_time: float = 0.0,
        transcript: str = "",
        **kwargs
    ):
        super().__init__(
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            transcript=transcript,
            **kwargs
        )
        self.speaker = speaker
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.transcript = transcript


class ConversationSession(BaseModel):
    """Complete meeting/conversation session data."""
    session_id: str = ""
    timestamp_utc: str = ""
    target_speaker: str = "USER"
    counterpart_name: str = ""
    counterpart_role: str = ""
    power_axis: str = "LATERAL"
    dialogue: List[Utterance] = []
    raw_audio_path: Optional[str] = None
    is_encrypted: bool = True

    def __init__(
        self,
        session_id: str = "",
        timestamp_utc: str = "",
        target_speaker: str = "USER",
        counterpart_name: str = "",
        counterpart_role: str = "",
        power_axis: str = "LATERAL",
        dialogue: Optional[List[Utterance]] = None,
        raw_audio_path: Optional[str] = None,
        is_encrypted: bool = True,
        **kwargs
    ):
        super().__init__(
            session_id=session_id,
            timestamp_utc=timestamp_utc,
            target_speaker=target_speaker,
            counterpart_name=counterpart_name,
            counterpart_role=counterpart_role,
            power_axis=power_axis,
            dialogue=dialogue or [],
            raw_audio_path=raw_audio_path,
            is_encrypted=is_encrypted,
            **kwargs
        )
        self.session_id = session_id
        self.timestamp_utc = timestamp_utc
        self.target_speaker = target_speaker
        self.counterpart_name = counterpart_name
        self.counterpart_role = counterpart_role
        self.power_axis = power_axis
        self.dialogue = dialogue or []
        self.raw_audio_path = raw_audio_path
        self.is_encrypted = is_encrypted
