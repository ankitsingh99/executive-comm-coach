"""
Multimodal Communication Context & Power Axis Calibrator.
Analyzes vocal acoustics (pitch modulation, energy dynamics & attenuation,
turn-taking pacing, cross-talk overlap) combined with linguistic dialogue markers
to automatically gauge and calibrate the relational communication context (Power Axis).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from ..asr_diarization.indic_normalizer import IndicNormalizer
    from .persona_ontology import PowerAxis
    from .schema import AcousticAnalysisResult, Utterance
except (ImportError, ValueError):
    from asr_diarization.indic_normalizer import IndicNormalizer
    from engine.persona_ontology import PowerAxis
    from engine.schema import AcousticAnalysisResult, Utterance


@dataclass
class ContextInferenceResult:
    """Inferred communication context with acoustic & linguistic rationale."""

    recommended_axis: PowerAxis
    confidence_score: float
    acoustic_rationale: str
    semantic_rationale: str
    recommended_name: str
    recommended_role: str
    axis_scores: Dict[str, float] = field(default_factory=dict)


class CommunicationContextCalibrator:
    """
    Multimodal context calibration engine.
    Fuses on-device acoustic voice modulation and conversational discourse patterns.
    """

    # Semantic Keyword Dictionaries for Relational Contexts
    CASUAL_MARKERS = [
        r"\bbhai\b",
        r"\byaar\b",
        r"\barre\b",
        r"\bchalo\b",
        r"\btheek hai\b",
        r"\bhangout\b",
        r"\bcoffee\b",
        r"\bparty\b",
        r"\bweekend\b",
        r"\bchill\b",
        r"\bcool\b",
        r"\blol\b",
        r"\bhaha\b",
        r"\bdude\b",
        r"\bbro\b",
        r"\bmatlab\b",
        r"\bbasically\b",
        r"\bkya haal\b",
        r"\bsahi hai\b",
    ]

    LATERAL_TECH_MARKERS = [
        r"\bsync\b",
        r"\bsprint\b",
        r"\bdeploy\b",
        r"\brelease\b",
        r"\bcode\b",
        r"\bapi\b",
        r"\barchitecture\b",
        r"\btimeline\b",
        r"\bmilestone\b",
        r"\bfeature\b",
        r"\bdatabase\b",
        r"\breview\b",
        r"\bblocker\b",
        r"\bpull request\b",
        r"\bpr\b",
        r"\bmerge\b",
        r"\bstakeholder\b",
        r"\bteam\b",
        r"\bcolleague\b",
        r"\bmeeting\b",
    ]

    UPWARD_EXECUTIVE_MARKERS = [
        r"\bstrategy\b",
        r"\broi\b",
        r"\brevenue\b",
        r"\bbudget\b",
        r"\bheadcount\b",
        r"\bquarter\b",
        r"\bq[1-4]\b",
        r"\bdeliverables\b",
        r"\bboard\b",
        r"\bvp\b",
        r"\bdirector\b",
        r"\bcxo\b",
        r"\bleadership\b",
        r"\bproposal\b",
        r"\bexecutive\b",
        r"\bpresentation\b",
        r"\bstrategic\b",
        r"\bkpi\b",
        r"\bokr\b",
    ]

    DOWNWARD_COACHING_MARKERS = [
        r"\bfeedback\b",
        r"\blearn\b",
        r"\bgrowth\b",
        r"\bimprovement\b",
        r"\bguidance\b",
        r"\bhelp you\b",
        r"\bsuggest\b",
        r"\b1-on-1\b",
        r"\bone on one\b",
        r"\bcareer\b",
        r"\bmentor\b",
        r"\bgoals\b",
        r"\bnext time\b",
        r"\bdo this\b",
    ]

    CONFLICT_NEGOTIATION_MARKERS = [
        r"\bunfair\b",
        r"\bunfairness\b",
        r"\bmana kar\b",
        r"\bdisagree\b",
        r"\bdispute\b",
        r"\bproblem\b",
        r"\bissue\b",
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bcost\b",
        r"\bdelay\b",
        r"\brefuse\b",
        r"\bconflict\b",
        r"\bargument\b",
        r"\bwhy is\b",
        r"\bblame\b",
        r"\bnegotiate\b",
        r"\bcompromise\b",
        r"\bkhush\b",
    ]

    SOLO_REHEARSAL_MARKERS = [
        r"\btoday i will\b",
        r"\bi am going to present\b",
        r"\bthe goal of this\b",
        r"\bin summary\b",
        r"\bfirstly\b",
        r"\bsecondly\b",
        r"\blet me explain\b",
        r"\bmy presentation\b",
    ]

    @classmethod
    def infer_context(
        cls, acoustic_result: Optional[AcousticAnalysisResult], utterances: List[Utterance]
    ) -> ContextInferenceResult:
        """
        Synthesizes acoustic vocal dynamics and transcribed dialogue text
        to infer the optimal communication coaching context.
        """
        combined_text = IndicNormalizer.normalize_text(" ".join(u.transcript for u in utterances))
        speaker_count = acoustic_result.detected_speaker_count if acoustic_result else 1
        overall_tone = acoustic_result.overall_tone if acoustic_result else "Calm & Measured"

        # 1. Acoustic Modulation & Attenuation Scoring
        scores = {
            PowerAxis.SOLO: 0.0,
            PowerAxis.CASUAL: 0.0,
            PowerAxis.LATERAL: 0.0,
            PowerAxis.UPWARD: 0.0,
            PowerAxis.DOWNWARD: 0.0,
            PowerAxis.CONFLICT: 0.0,
        }

        acoustic_notes = []

        # Analyze speaker turn distribution
        spk_turns = {}
        for u in utterances:
            spk_turns[u.speaker] = spk_turns.get(u.speaker, 0) + 1
        is_monologue = len(spk_turns) <= 1 and speaker_count == 1

        if is_monologue:
            scores[PowerAxis.SOLO] += 4.5
            acoustic_notes.append("Single voice stream with zero turn-taking (Solo rehearsal pattern)")
        else:
            # Multi-speaker dynamics
            scores[PowerAxis.LATERAL] += 2.0
            scores[PowerAxis.CASUAL] += 2.0

            # Inspect speaker profiles
            if acoustic_result and acoustic_result.speakers:
                talk_times = [s.talk_time_percentage for s in acoustic_result.speakers]
                max_talk = max(talk_times) if talk_times else 50.0

                # If one speaker heavily dominates talk-time (> 72%) in multi-speaker
                if max_talk > 72.0:
                    scores[PowerAxis.DOWNWARD] += 2.5
                    scores[PowerAxis.UPWARD] += 1.5
                    acoustic_notes.append(f"Asymmetric talk-time ({max_talk:.0f}% dominant speaker)")
                else:
                    scores[PowerAxis.LATERAL] += 2.0
                    scores[PowerAxis.CASUAL] += 1.5
                    acoustic_notes.append("Balanced reciprocal turn-taking between speakers")

        # Overlapping speech & interruptions
        overlap_events = acoustic_result.overlapping_speech_events if acoustic_result else 0
        overlap_dur = acoustic_result.overlap_duration_total_sec if acoustic_result else 0.0

        if overlap_events >= 2 or overlap_dur > 1.5:
            scores[PowerAxis.CONFLICT] += 3.0
            scores[PowerAxis.CASUAL] += 1.5
            acoustic_notes.append(f"Elevated simultaneous cross-talk ({overlap_dur:.1f}s overlap)")

        # Vocal Tone & Pitch Modulation
        if "Assertive" in overall_tone or "Heightened" in overall_tone:
            scores[PowerAxis.CONFLICT] += 2.0
            scores[PowerAxis.UPWARD] += 1.5
            acoustic_notes.append(f"High vocal energy & assertive inflection ({overall_tone})")
        elif "Dynamic" in overall_tone or "Expressive" in overall_tone:
            scores[PowerAxis.CASUAL] += 3.0
            scores[PowerAxis.LATERAL] += 1.5
            acoustic_notes.append("High pitch variability and dynamic melodic inflection")
        elif "Measured" in overall_tone or "Subdued" in overall_tone:
            scores[PowerAxis.UPWARD] += 2.0
            scores[PowerAxis.SOLO] += 1.5
            acoustic_notes.append("Measured, attenuated vocal pacing")

        # 2. Semantic & Discourse Pattern Scoring
        def _count_matches(patterns: List[str]) -> int:
            return sum(len(re.findall(p, combined_text, flags=re.IGNORECASE)) for p in patterns)

        casual_hits = _count_matches(cls.CASUAL_MARKERS)
        lateral_hits = _count_matches(cls.LATERAL_TECH_MARKERS)
        upward_hits = _count_matches(cls.UPWARD_EXECUTIVE_MARKERS)
        downward_hits = _count_matches(cls.DOWNWARD_COACHING_MARKERS)
        conflict_hits = _count_matches(cls.CONFLICT_NEGOTIATION_MARKERS)
        solo_hits = _count_matches(cls.SOLO_REHEARSAL_MARKERS)

        scores[PowerAxis.CASUAL] += casual_hits * 1.8
        scores[PowerAxis.LATERAL] += lateral_hits * 2.2
        scores[PowerAxis.UPWARD] += upward_hits * 2.5
        scores[PowerAxis.DOWNWARD] += downward_hits * 2.5
        scores[PowerAxis.CONFLICT] += conflict_hits * 3.0
        scores[PowerAxis.SOLO] += solo_hits * 2.0

        semantic_notes = []
        if conflict_hits > 0:
            semantic_notes.append(f"Negotiation/friction indicators ({conflict_hits} markers detected)")
        if casual_hits > 0:
            semantic_notes.append(f"Informal colloquial banter ({casual_hits} markers detected)")
        if lateral_hits > 0:
            semantic_notes.append(f"Peer project & collaboration discourse ({lateral_hits} markers)")
        if upward_hits > 0:
            semantic_notes.append(f"Formal executive terms ({upward_hits} markers)")
        if downward_hits > 0:
            semantic_notes.append(f"Mentoring & feedback guidance ({downward_hits} markers)")

        if not semantic_notes:
            semantic_notes.append("General conversational discourse")

        # 3. Determine Winning Context
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_axis, best_score = ranked[0]

        total_s = sum(scores.values()) + 1e-6
        confidence = min(0.96, max(0.65, float(best_score / total_s * 1.8)))

        # Default naming / role heuristics
        role_map = {
            PowerAxis.SOLO: ("Self (Solo Practice)", "Self"),
            PowerAxis.CASUAL: ("Friend / Colleague", "Informal Contact"),
            PowerAxis.LATERAL: ("Peer Collaborator", "Team Member"),
            PowerAxis.UPWARD: ("Senior Leadership", "Manager / Executive"),
            PowerAxis.DOWNWARD: ("Direct Report", "Mentee / Team Member"),
            PowerAxis.CONFLICT: ("Counterpart", "Negotiation Contact"),
        }
        rec_name, rec_role = role_map[best_axis]

        # Convert score distribution to percentages
        score_dist = {axis.value: round((s / total_s) * 100, 1) for axis, s in ranked}

        return ContextInferenceResult(
            recommended_axis=best_axis,
            confidence_score=round(confidence, 2),
            acoustic_rationale="; ".join(acoustic_notes) if acoustic_notes else "Natural vocal delivery",
            semantic_rationale="; ".join(semantic_notes),
            recommended_name=rec_name,
            recommended_role=rec_role,
            axis_scores=score_dist,
        )
