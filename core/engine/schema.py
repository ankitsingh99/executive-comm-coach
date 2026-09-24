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
        **kwargs,
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
            **kwargs,
        )
        self.speaker_id = speaker_id
        self.mean_pitch_hz = float(mean_pitch_hz)
        self.pitch_range_hz = float(pitch_range_hz)
        self.energy_rms = float(energy_rms)
        self.speech_rate_wpm = float(speech_rate_wpm)
        self.tone_label = tone_label
        self.talk_time_percentage = float(talk_time_percentage)
        self.confidence_score = float(confidence_score)


class AcousticFillerEvent(BaseModel):
    """Detected non-phonetic acoustic vocal hesitation event from raw audio waveforms."""

    token: str = "umm"  # umm, aah, aaaaa, uhh, hmm, err
    start_time: float = 0.0
    end_time: float = 0.0
    duration_sec: float = 0.0
    confidence: float = 0.90
    speaker: str = "USER"
    acoustic_type: str = "nasal_murmur"  # nasal_murmur, vowel_elongation, open_pause, vocal_hesitation

    def __init__(
        self,
        token: str = "umm",
        start_time: float = 0.0,
        end_time: float = 0.0,
        duration_sec: float = 0.0,
        confidence: float = 0.90,
        speaker: str = "USER",
        acoustic_type: str = "nasal_murmur",
        **kwargs,
    ):
        super().__init__(
            token=token,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            confidence=confidence,
            speaker=speaker,
            acoustic_type=acoustic_type,
            **kwargs,
        )
        self.token = str(token)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.duration_sec = float(duration_sec)
        self.confidence = float(confidence)
        self.speaker = str(speaker)
        self.acoustic_type = str(acoustic_type)


class AcousticAnalysisResult(BaseModel):
    """Aggregate acoustic voice, speaker detection, and hesitation analysis."""

    detected_speaker_count: int = 1
    is_multi_speaker: bool = False
    speakers: List[SpeakerAcousticProfile] = []
    overall_tone: str = "Calm & Measured"
    turn_taking_events: int = 0
    overlapping_speech_events: int = 0
    overlap_duration_total_sec: float = 0.0
    acoustic_fillers: List[AcousticFillerEvent] = []

    def __init__(
        self,
        detected_speaker_count: int = 1,
        is_multi_speaker: bool = False,
        speakers: Optional[List[SpeakerAcousticProfile]] = None,
        overall_tone: str = "Calm & Measured",
        turn_taking_events: int = 0,
        overlapping_speech_events: int = 0,
        overlap_duration_total_sec: float = 0.0,
        acoustic_fillers: Optional[List[AcousticFillerEvent]] = None,
        **kwargs,
    ):
        super().__init__(
            detected_speaker_count=detected_speaker_count,
            is_multi_speaker=is_multi_speaker,
            speakers=speakers or [],
            overall_tone=overall_tone,
            turn_taking_events=turn_taking_events,
            overlapping_speech_events=overlapping_speech_events,
            overlap_duration_total_sec=overlap_duration_total_sec,
            acoustic_fillers=acoustic_fillers or [],
            **kwargs,
        )
        self.detected_speaker_count = int(detected_speaker_count)
        self.is_multi_speaker = bool(is_multi_speaker)
        self.speakers = speakers or []
        self.overall_tone = overall_tone
        self.turn_taking_events = int(turn_taking_events)
        self.overlapping_speech_events = int(overlapping_speech_events)
        self.overlap_duration_total_sec = float(overlap_duration_total_sec)
        self.acoustic_fillers = acoustic_fillers or []


class CommunicationMetrics(BaseModel):
    """Core quantitative communication efficacy scores [0, 100]."""

    presence_score: int = 0
    assertiveness_score: int = 0
    active_listening_score: int = 0
    filler_words_detected: List[FillerWordMetric] = []
    interruption_count: int = 0
    overlap_count: int = 0
    acoustic_analysis: Optional[AcousticAnalysisResult] = None

    def __init__(
        self,
        presence_score: int = 0,
        assertiveness_score: int = 0,
        active_listening_score: int = 0,
        filler_words_detected: Optional[List[FillerWordMetric]] = None,
        interruption_count: int = 0,
        overlap_count: int = 0,
        **kwargs,
    ):
        super().__init__(
            presence_score=presence_score,
            assertiveness_score=assertiveness_score,
            active_listening_score=active_listening_score,
            filler_words_detected=filler_words_detected or [],
            interruption_count=interruption_count,
            overlap_count=overlap_count,
            **kwargs,
        )
        self.presence_score = max(0, min(100, int(presence_score)))
        self.assertiveness_score = max(0, min(100, int(assertiveness_score)))
        self.active_listening_score = max(0, min(100, int(active_listening_score)))
        self.filler_words_detected = filler_words_detected or []
        self.interruption_count = int(interruption_count)
        self.overlap_count = int(overlap_count)


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

    def __init__(self, critique: str = "", verbatim_quote: str = "", coached_phrasing: str = "", **kwargs):
        super().__init__(
            critique=critique[:250], verbatim_quote=verbatim_quote, coached_phrasing=coached_phrasing[:250], **kwargs
        )
        self.critique = critique[:250]
        self.verbatim_quote = verbatim_quote
        self.coached_phrasing = coached_phrasing[:250]


class KeyHighlight(BaseModel):
    """Itemized key highlight, strategic decision, or takeaway extracted from transcription."""

    headline: str = ""
    takeaway: str = ""
    speaker: str = "SPEAKER"
    verbatim_quote: str = ""
    category: str = "Key Takeaway"
    importance: str = "Normal"

    def __init__(
        self,
        headline: str = "",
        takeaway: str = "",
        speaker: str = "SPEAKER",
        verbatim_quote: str = "",
        category: str = "Key Takeaway",
        importance: str = "Normal",
        **kwargs,
    ):
        super().__init__(
            headline=headline[:250],
            takeaway=takeaway[:300],
            speaker=speaker,
            verbatim_quote=verbatim_quote,
            category=category,
            importance=importance,
            **kwargs,
        )
        self.headline = headline[:250]
        self.takeaway = takeaway[:300]
        self.speaker = speaker
        self.verbatim_quote = verbatim_quote
        self.category = category
        self.importance = importance


class ActionItem(BaseModel):
    """Extracted action item, commitment, or scheduled follow-up from spoken dialogue."""

    owner: str = "USER"
    task: str = ""
    due_time_or_date: Optional[str] = None
    resolved_datetime: Optional[str] = None
    target_time_inferred_ampm: Optional[str] = None
    verbatim_quote: str = ""
    category: str = "Follow-up"
    urgency: str = "Normal"

    def __init__(
        self,
        owner: str = "USER",
        task: str = "",
        due_time_or_date: Optional[str] = None,
        resolved_datetime: Optional[str] = None,
        target_time_inferred_ampm: Optional[str] = None,
        verbatim_quote: str = "",
        category: str = "Follow-up",
        urgency: str = "Normal",
        **kwargs,
    ):
        super().__init__(
            owner=owner,
            task=task[:250],
            due_time_or_date=due_time_or_date,
            resolved_datetime=resolved_datetime,
            target_time_inferred_ampm=target_time_inferred_ampm,
            verbatim_quote=verbatim_quote,
            category=category,
            urgency=urgency,
            **kwargs,
        )
        self.owner = owner
        self.task = task[:250]
        self.due_time_or_date = due_time_or_date
        self.resolved_datetime = resolved_datetime
        self.target_time_inferred_ampm = target_time_inferred_ampm
        self.verbatim_quote = verbatim_quote
        self.category = category
        self.urgency = urgency


# PotentialTask is an alias for ActionItem
PotentialTask = ActionItem


class TranscriptionAnalysisResult(BaseModel):
    """Complete structured intelligence extracted from transcription."""

    session_id: str = ""
    timestamp_utc: str = ""
    summary: str = ""
    key_highlights: List[KeyHighlight] = []
    potential_tasks: List[ActionItem] = []
    topics_discussed: List[str] = []
    sentiment_tone: str = "Neutral & Constructive"

    def __init__(
        self,
        session_id: str = "",
        timestamp_utc: str = "",
        summary: str = "",
        key_highlights: Optional[List[KeyHighlight]] = None,
        potential_tasks: Optional[List[ActionItem]] = None,
        topics_discussed: Optional[List[str]] = None,
        sentiment_tone: str = "Neutral & Constructive",
        **kwargs,
    ):
        super().__init__(
            session_id=session_id,
            timestamp_utc=timestamp_utc,
            summary=summary,
            key_highlights=key_highlights or [],
            potential_tasks=potential_tasks or [],
            topics_discussed=topics_discussed or [],
            sentiment_tone=sentiment_tone,
            **kwargs,
        )
        self.session_id = session_id
        self.timestamp_utc = timestamp_utc
        self.summary = summary
        self.key_highlights = key_highlights or []
        self.potential_tasks = potential_tasks or []
        self.topics_discussed = topics_discussed or []
        self.sentiment_tone = sentiment_tone


class EmotionalTrajectoryPoint(BaseModel):
    """Emotional trajectory and psychological tension marker across conversational turns."""

    timestamp_sec: float = 0.0
    speaker: str = "USER"
    emotion_label: str = "Neutral & Composed"
    valence_score: float = 0.0  # -1.0 (highly negative/anxious) to +1.0 (highly positive/enthusiastic)
    tension_level: str = "LOW"  # LOW, MEDIUM, HIGH
    pacing_wpm: float = 0.0

    def __init__(
        self,
        timestamp_sec: float = 0.0,
        speaker: str = "USER",
        emotion_label: str = "Neutral & Composed",
        valence_score: float = 0.0,
        tension_level: str = "LOW",
        pacing_wpm: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            timestamp_sec=timestamp_sec,
            speaker=speaker,
            emotion_label=emotion_label,
            valence_score=valence_score,
            tension_level=tension_level,
            pacing_wpm=pacing_wpm,
            **kwargs,
        )
        self.timestamp_sec = float(timestamp_sec)
        self.speaker = str(speaker)
        self.emotion_label = str(emotion_label)
        self.valence_score = float(valence_score)
        self.tension_level = str(tension_level)
        self.pacing_wpm = float(pacing_wpm)


class ConversationalDynamicsMetric(BaseModel):
    """Comprehensive conversational dynamics, pacing, talk-time parity, and inquiry/advocacy balance."""

    user_talk_time_pct: float = 50.0
    counterpart_talk_time_pct: float = 50.0
    user_words_total: int = 0
    counterpart_words_total: int = 0
    average_turn_latency_ms: float = 0.0
    ask_vs_tell_ratio: float = 1.0  # Inquiry to Advocacy ratio (inquiries / max(1, directives))
    inquiry_count: int = 0
    directive_count: int = 0
    brevity_potential_pct: float = 0.0  # Estimated reduction percentage by eliminating redundancies & fillers
    deep_listening_score: int = 0  # 0-100 reflective listening vs passive nod quotient
    vocal_tension_index: str = "Calm & Grounded"

    def __init__(
        self,
        user_talk_time_pct: float = 50.0,
        counterpart_talk_time_pct: float = 50.0,
        user_words_total: int = 0,
        counterpart_words_total: int = 0,
        average_turn_latency_ms: float = 0.0,
        ask_vs_tell_ratio: float = 1.0,
        inquiry_count: int = 0,
        directive_count: int = 0,
        brevity_potential_pct: float = 0.0,
        deep_listening_score: int = 0,
        vocal_tension_index: str = "Calm & Grounded",
        **kwargs,
    ):
        super().__init__(
            user_talk_time_pct=user_talk_time_pct,
            counterpart_talk_time_pct=counterpart_talk_time_pct,
            user_words_total=user_words_total,
            counterpart_words_total=counterpart_words_total,
            average_turn_latency_ms=average_turn_latency_ms,
            ask_vs_tell_ratio=ask_vs_tell_ratio,
            inquiry_count=inquiry_count,
            directive_count=directive_count,
            brevity_potential_pct=brevity_potential_pct,
            deep_listening_score=deep_listening_score,
            vocal_tension_index=vocal_tension_index,
            **kwargs,
        )
        self.user_talk_time_pct = float(user_talk_time_pct)
        self.counterpart_talk_time_pct = float(counterpart_talk_time_pct)
        self.user_words_total = int(user_words_total)
        self.counterpart_words_total = int(counterpart_words_total)
        self.average_turn_latency_ms = float(average_turn_latency_ms)
        self.ask_vs_tell_ratio = float(ask_vs_tell_ratio)
        self.inquiry_count = int(inquiry_count)
        self.directive_count = int(directive_count)
        self.brevity_potential_pct = float(brevity_potential_pct)
        self.deep_listening_score = max(0, min(100, int(deep_listening_score)))
        self.vocal_tension_index = str(vocal_tension_index)


class AgreementPoint(BaseModel):
    """Explicit consensus or mutual alignment point finalized during conversation."""

    headline: str = ""
    agreed_solution: str = ""
    speaker_turn: str = ""
    verbatim_quote: str = ""

    def __init__(
        self,
        headline: str = "",
        agreed_solution: str = "",
        speaker_turn: str = "",
        verbatim_quote: str = "",
        **kwargs,
    ):
        super().__init__(
            headline=headline[:250],
            agreed_solution=agreed_solution[:300],
            speaker_turn=speaker_turn,
            verbatim_quote=verbatim_quote,
            **kwargs,
        )
        self.headline = headline[:250]
        self.agreed_solution = agreed_solution[:300]
        self.speaker_turn = speaker_turn
        self.verbatim_quote = verbatim_quote


class UnresolvedOpenLoop(BaseModel):
    """Unresolved tension, lingering objection, or open loop requiring follow-up resolution."""

    concern_topic: str = ""
    raised_by: str = ""
    context: str = ""
    recommended_followup: str = ""

    def __init__(
        self,
        concern_topic: str = "",
        raised_by: str = "",
        context: str = "",
        recommended_followup: str = "",
        **kwargs,
    ):
        super().__init__(
            concern_topic=concern_topic[:250],
            raised_by=raised_by,
            context=context[:300],
            recommended_followup=recommended_followup[:300],
            **kwargs,
        )
        self.concern_topic = concern_topic[:250]
        self.raised_by = raised_by
        self.context = context[:300]
        self.recommended_followup = recommended_followup[:300]


class ExecutiveCoachingEvaluation(BaseModel):
    """Complete structured coaching evaluation report constrained by Top-N parameter."""

    persona_context: str = ""
    metrics: CommunicationMetrics = None
    top_strengths: List[TopStrength] = []
    areas_for_improvement: List[AreaForImprovement] = []
    action_items: List[ActionItem] = []
    key_highlights: List[KeyHighlight] = []
    longitudinal_summary: str = ""
    persona_alignment_notes: str = ""
    dynamics: Optional[ConversationalDynamicsMetric] = None
    emotional_trajectory: List[EmotionalTrajectoryPoint] = []
    agreements: List[AgreementPoint] = []
    unresolved_loops: List[UnresolvedOpenLoop] = []

    def __init__(
        self,
        persona_context: str = "",
        metrics: Optional[CommunicationMetrics] = None,
        top_strengths: Optional[List[TopStrength]] = None,
        areas_for_improvement: Optional[List[AreaForImprovement]] = None,
        action_items: Optional[List[ActionItem]] = None,
        key_highlights: Optional[List[KeyHighlight]] = None,
        longitudinal_summary: str = "",
        persona_alignment_notes: str = "",
        dynamics: Optional[ConversationalDynamicsMetric] = None,
        emotional_trajectory: Optional[List[EmotionalTrajectoryPoint]] = None,
        agreements: Optional[List[AgreementPoint]] = None,
        unresolved_loops: Optional[List[UnresolvedOpenLoop]] = None,
        **kwargs,
    ):
        super().__init__(
            persona_context=persona_context,
            metrics=metrics,
            top_strengths=top_strengths or [],
            areas_for_improvement=areas_for_improvement or [],
            action_items=action_items or [],
            key_highlights=key_highlights or [],
            longitudinal_summary=longitudinal_summary,
            persona_alignment_notes=persona_alignment_notes,
            dynamics=dynamics,
            emotional_trajectory=emotional_trajectory or [],
            agreements=agreements or [],
            unresolved_loops=unresolved_loops or [],
            **kwargs,
        )
        self.persona_context = persona_context
        self.metrics = metrics or CommunicationMetrics()
        self.top_strengths = top_strengths or []
        self.areas_for_improvement = areas_for_improvement or []
        self.action_items = action_items or []
        self.key_highlights = key_highlights or []
        self.longitudinal_summary = longitudinal_summary
        self.persona_alignment_notes = persona_alignment_notes
        self.dynamics = dynamics or ConversationalDynamicsMetric()
        self.emotional_trajectory = emotional_trajectory or []
        self.agreements = agreements or []
        self.unresolved_loops = unresolved_loops or []


class Utterance(BaseModel):
    """Single timestamped and diarized dialogue turn."""

    speaker: str = "USER"
    start_time: float = 0.0
    end_time: float = 0.0
    transcript: str = ""
    is_overlapping: bool = False
    overlap_duration_sec: float = 0.0
    interrupted_speaker: Optional[str] = None

    def __init__(
        self,
        speaker: str = "USER",
        start_time: float = 0.0,
        end_time: float = 0.0,
        transcript: str = "",
        is_overlapping: bool = False,
        overlap_duration_sec: float = 0.0,
        interrupted_speaker: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            transcript=transcript,
            is_overlapping=is_overlapping,
            overlap_duration_sec=overlap_duration_sec,
            interrupted_speaker=interrupted_speaker,
            **kwargs,
        )
        self.speaker = speaker
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.transcript = transcript
        self.is_overlapping = bool(is_overlapping)
        self.overlap_duration_sec = float(overlap_duration_sec)
        self.interrupted_speaker = interrupted_speaker


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
        **kwargs,
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
            **kwargs,
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
