"""
Conversational Intelligence & Dynamics Engine.
Extracts multi-dimensional behavioral intelligence, psychological tension trajectories,
airtime parity, inquiry-to-advocacy ratios, consensus agreements, and unresolved open loops.
"""

import re
from typing import List, Optional, Tuple

try:
    from .schema import (
        Utterance,
        ConversationalDynamicsMetric,
        EmotionalTrajectoryPoint,
        AgreementPoint,
        UnresolvedOpenLoop,
        AcousticAnalysisResult,
    )
except (ImportError, ValueError):
    from engine.schema import (
        Utterance,
        ConversationalDynamicsMetric,
        EmotionalTrajectoryPoint,
        AgreementPoint,
        UnresolvedOpenLoop,
        AcousticAnalysisResult,
    )


# Active Listening & Paraphrasing Markers
REFLECTION_CUES = [
    r"\bso what you(?:'re| are) saying\b",
    r"\bif i understand correctly\b",
    r"\bto your point\b",
    r"\bas you mentioned\b",
    r"\bbuilding on (?:that|what you said)\b",
    r"\bechoing (?:your|that)\b",
    r"\bi hear you\b",
    r"\byou're suggesting that\b",
    r"\bwhat i'm taking away is\b",
    r"\baapka matlab hai\b",
    r"\bjaise aapne kaha\b",
    r"\bjo aap keh rahe hain\b",
    r"\bsamajh gaya\b",
]

# Passive Acknowledgement Nods
PASSIVE_NODS = [
    r"^(?:yeah|yes|yep|yup|ok|okay|right|sure|mhm|uh-huh|haan|sahi hai|theek hai|achha)\.?$",
    r"^(?:cool|got it|makes sense|noted)\.?$",
]

# Consensus and Agreement Cues
AGREEMENT_CUES = [
    r"\b(?:we agree|agreed|sounds like a plan|deal|sounds good to me|aligned on|same page)\b",
    r"\b(?:let's lock (?:this|that) in|let's go with|let's proceed with|final decision is)\b",
    r"\b(?:perfect, let's do (?:that|it)|we have a consensus|i'm on board)\b",
    r"\b(?:theek hai (?:ye|yahi) karte hain|haan bilkul yahi karenge|done deal|ye finalize karte hain)\b",
]

# Open Loop and Unresolved Tension Cues
OPEN_LOOP_CUES = [
    r"\b(?:still need to figure out|let's revisit|revisit (?:this|later)|not sure about)\b",
    r"\b(?:open question|open concern|pending (?:approval|review|decision)|concern (?:with|regarding)|risk of|blocker)\b",
    r"\b(?:we haven't decided on|table this for now|circle back on|unresolved)\b",
    r"\b(?:baad mein dekhte hain|abhi decide nahi hua|is par clarity chahiye|doubt hai|risk lag raha hai)\b",
]

# Hedging & Filler Words for Brevity Analysis
BREVITY_FLUFF = [
    r"\bbasically\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bliterally\b",
    r"\bactually\b",
    r"\bi mean\b",
    r"\byou know\b",
    r"\bto be honest\b",
    r"\bfor what it's worth\b",
    r"\bmatlab\b",
    r"\bmujhe lagta hai\b",
    r"\bshayad\b",
    r"\bdekha jaye toh\b",
]


class ConversationalIntelligenceEngine:
    """
    Computes comprehensive conversational intelligence analytics,
    turn-taking pacing, psychological tension arcs, agreements, and open loops.
    """

    @classmethod
    def analyze_session(
        cls,
        dialogue: List[Utterance],
        target_speaker: str = "USER",
        acoustic_result: Optional[AcousticAnalysisResult] = None,
    ) -> Tuple[
        ConversationalDynamicsMetric,
        List[EmotionalTrajectoryPoint],
        List[AgreementPoint],
        List[UnresolvedOpenLoop],
    ]:
        """
        Executes full conversational intelligence analysis on timestamped dialogue turns.
        """
        dynamics = cls.compute_conversational_dynamics(dialogue, target_speaker, acoustic_result)
        emotional_trajectory = cls.extract_emotional_trajectory(dialogue, acoustic_result)
        agreements = cls.extract_agreements(dialogue)
        unresolved_loops = cls.extract_open_loops(dialogue)

        return dynamics, emotional_trajectory, agreements, unresolved_loops

    @classmethod
    def compute_conversational_dynamics(
        cls,
        dialogue: List[Utterance],
        target_speaker: str = "USER",
        acoustic_result: Optional[AcousticAnalysisResult] = None,
    ) -> ConversationalDynamicsMetric:
        """
        Calculates airtime parity, turn latency, inquiry-advocacy ratio, brevity potential, and listening depth.
        """
        if not dialogue:
            return ConversationalDynamicsMetric()

        user_synonyms = {"USER", "SELF", "YOU", target_speaker.upper()}
        user_words = 0
        counterpart_words = 0
        user_time_sec = 0.0
        counterpart_time_sec = 0.0

        user_inquiries = 0
        user_directives = 0
        user_fluff_words = 0

        turn_latencies_ms: List[float] = []
        reflection_matches = 0
        counterpart_turns_count = 0

        for idx, u in enumerate(dialogue):
            is_user = u.speaker.strip().upper() in user_synonyms or (
                len(dialogue) == 1 and u.speaker.strip().upper() in ["SPEAKER_00", "SPEAKER_01"]
            )
            words = [w for w in re.findall(r"\b\w+\b", u.transcript)]
            w_count = len(words)
            dur = max(0.0, u.end_time - u.start_time)
            if dur <= 0.0:
                # Estimate duration from word count (~140 wpm = ~2.33 words/sec)
                dur = max(0.5, w_count / 2.33)

            if is_user:
                user_words += w_count
                user_time_sec += dur

                # Inquiry vs Directive Analysis
                transcript = u.transcript.strip()
                # Remove leading fillers to check sentence opening
                clean_start = re.sub(
                    r"^(?:so|well|umm*|uhh*|basically|matlab|dekho|actually|okay|ok)\s*,?\s*",
                    "",
                    transcript,
                    flags=re.IGNORECASE,
                ).strip()
                is_q = (
                    "?" in transcript
                    or bool(
                        re.search(
                            r"^(?:how|what|why|where|when|which|who|whom|whose|can|could|should|would|is|are|am|do|does|did|will|shall|may|might|kya|kyun|kaise|kab|kahan|batao)\b",
                            clean_start,
                            re.IGNORECASE,
                        )
                    )
                    or bool(
                        re.search(
                            r"\b(?:what about|how about|what do you think|kya lagta hai|kya sochte ho|can we|could we|should we)\b",
                            transcript,
                            re.IGNORECASE,
                        )
                    )
                )
                if is_q:
                    user_inquiries += 1
                else:
                    user_directives += 1

                # Fluff & Redundancy Scan for Brevity Potential
                for pat in BREVITY_FLUFF:
                    matches = re.findall(pat, transcript, re.IGNORECASE)
                    user_fluff_words += len(matches) * 2

                # Reflection / Deep Listening (Did user build on previous counterpart turn?)
                if idx > 0 and dialogue[idx - 1].speaker.strip().upper() not in user_synonyms:
                    for r_pat in REFLECTION_CUES:
                        if re.search(r_pat, transcript, re.IGNORECASE):
                            reflection_matches += 1
                            break
            else:
                counterpart_words += w_count
                counterpart_time_sec += dur
                counterpart_turns_count += 1

            # Turn-taking latency computation
            if idx > 0 and dialogue[idx].speaker != dialogue[idx - 1].speaker:
                lat = dialogue[idx].start_time - dialogue[idx - 1].end_time
                if 0.0 <= lat <= 8.0:  # Latency within realistic conversational boundaries
                    turn_latencies_ms.append(lat * 1000.0)

        total_time = user_time_sec + counterpart_time_sec
        if total_time > 0:
            user_pct = round((user_time_sec / total_time) * 100.0, 1)
            cp_pct = round((counterpart_time_sec / total_time) * 100.0, 1)
        else:
            total_w = user_words + counterpart_words
            user_pct = round((user_words / total_w) * 100.0, 1) if total_w > 0 else 100.0
            cp_pct = round((counterpart_words / total_w) * 100.0, 1) if total_w > 0 else 0.0

        # Turn Latency Average
        avg_latency = round(sum(turn_latencies_ms) / len(turn_latencies_ms), 1) if turn_latencies_ms else 450.0

        # Ask vs Tell Ratio
        ask_ratio = round(user_inquiries / max(1, user_directives), 2)

        # Brevity Potential Index (estimated % reducible)
        if user_words > 0:
            brevity_pct = min(45.0, round((user_fluff_words / max(1, user_words)) * 100.0 + 8.0, 1))
        else:
            brevity_pct = 0.0

        # Deep Active Listening Score (0 - 100)
        if counterpart_turns_count > 0:
            deep_listening = min(100, int(60 + (reflection_matches / counterpart_turns_count) * 40))
        else:
            deep_listening = 85  # Solo rehearsal default

        # Vocal Tension Index from Acoustics
        tension_idx = "Calm & Grounded"
        if acoustic_result:
            if "Tense" in acoustic_result.overall_tone or "High Dynamic" in acoustic_result.overall_tone:
                tension_idx = "Elevated Tension / High Strain"
            elif "Monotone" in acoustic_result.overall_tone or "Flat" in acoustic_result.overall_tone:
                tension_idx = "Subdued / Guarded"
            elif "Vibrant" in acoustic_result.overall_tone or "Expressive" in acoustic_result.overall_tone:
                tension_idx = "High Energy & Expressive"
            else:
                tension_idx = "Calm & Grounded"

        return ConversationalDynamicsMetric(
            user_talk_time_pct=user_pct,
            counterpart_talk_time_pct=cp_pct,
            user_words_total=user_words,
            counterpart_words_total=counterpart_words,
            average_turn_latency_ms=avg_latency,
            ask_vs_tell_ratio=ask_ratio,
            inquiry_count=user_inquiries,
            directive_count=user_directives,
            brevity_potential_pct=brevity_pct,
            deep_listening_score=deep_listening,
            vocal_tension_index=tension_idx,
        )

    @classmethod
    def extract_emotional_trajectory(
        cls, dialogue: List[Utterance], acoustic_result: Optional[AcousticAnalysisResult] = None
    ) -> List[EmotionalTrajectoryPoint]:
        """
        Traces psychological tension and emotional valence across chronological dialogue turns.
        """
        if not dialogue:
            return []

        trajectory: List[EmotionalTrajectoryPoint] = []

        for idx, u in enumerate(dialogue):
            text = u.transcript.lower()
            duration = max(0.5, u.end_time - u.start_time)
            words = len(re.findall(r"\b\w+\b", text))
            wpm = round((words / duration) * 60.0, 1)

            # Heuristic sentiment / emotion analysis
            valence = 0.1
            emotion = "Neutral & Composed"
            tension = "LOW"

            if any(
                w in text
                for w in [
                    "great",
                    "awesome",
                    "excited",
                    "excellent",
                    "perfect",
                    "shandar",
                    "mast",
                    "achha",
                    "aligned",
                    "love",
                ]
            ):
                valence = 0.75
                emotion = "Enthusiastic & Aligned"
                tension = "LOW"
            elif any(
                w in text
                for w in [
                    "agree",
                    "yes",
                    "sure",
                    "makes sense",
                    "good",
                    "progress",
                    "clear",
                    "understand",
                    "sahi",
                    "theek",
                ]
            ):
                valence = 0.50
                emotion = "Constructive & Aligned"
                tension = "LOW"
            elif any(
                w in text
                for w in [
                    "problem",
                    "issue",
                    "delay",
                    "risk",
                    "blocker",
                    "concern",
                    "worried",
                    "frustrated",
                    "disagree",
                    "mushkil",
                    "dikkat",
                ]
            ):
                valence = -0.60
                emotion = "Concerned / Challenging"
                tension = "HIGH"
            elif any(w in text for w in ["maybe", "not sure", "possibly", "i guess", "shayad", "lag raha"]):
                valence = -0.15
                emotion = "Tentative / Hesitant"
                tension = "MEDIUM"
            elif "?" in u.transcript or any(
                w in text for w in ["how", "what", "why", "when", "kya", "kaise", "kyun", "kab"]
            ):
                valence = 0.25
                emotion = "Inquiring & Engaged"
                tension = "LOW" if wpm < 160 else "MEDIUM"

            if wpm > 180:
                tension = "HIGH"
                if valence > 0:
                    emotion = "Urgent / High-Paced Advocacy"
                else:
                    emotion = "Elevated Stress / Fast Pacing"

            trajectory.append(
                EmotionalTrajectoryPoint(
                    timestamp_sec=round(u.start_time, 2),
                    speaker=u.speaker,
                    emotion_label=emotion,
                    valence_score=valence,
                    tension_level=tension,
                    pacing_wpm=wpm,
                )
            )

        return trajectory

    @classmethod
    def extract_agreements(cls, dialogue: List[Utterance]) -> List[AgreementPoint]:
        """
        Extracts finalized agreements, consensus points, and mutual alignment decisions.
        """
        agreements: List[AgreementPoint] = []
        for u in dialogue:
            for pat in AGREEMENT_CUES:
                match = re.search(pat, u.transcript, re.IGNORECASE)
                if match:
                    # Clean up quote and summarize
                    sentence = u.transcript.strip()
                    # Headline formulation
                    topic_snip = sentence[:60].replace("\n", " ")
                    headline = f"Consensus: {topic_snip}..." if len(sentence) > 60 else f"Consensus: {sentence}"
                    agreements.append(
                        AgreementPoint(
                            headline=headline,
                            agreed_solution=sentence,
                            speaker_turn=f"{u.speaker} at {round(u.start_time, 1)}s",
                            verbatim_quote=sentence,
                        )
                    )
                    break
        return agreements

    @classmethod
    def extract_open_loops(cls, dialogue: List[Utterance]) -> List[UnresolvedOpenLoop]:
        """
        Extracts open tensions, unresolved blockers, and items flagged for future review.
        """
        open_loops: List[UnresolvedOpenLoop] = []
        for u in dialogue:
            for pat in OPEN_LOOP_CUES:
                match = re.search(pat, u.transcript, re.IGNORECASE)
                if match:
                    sentence = u.transcript.strip()
                    topic = sentence[:50] + "..." if len(sentence) > 50 else sentence
                    open_loops.append(
                        UnresolvedOpenLoop(
                            concern_topic=f"Pending: {topic}",
                            raised_by=u.speaker,
                            context=sentence,
                            recommended_followup=f"Schedule a 10-minute follow-up with {u.speaker} to close this loop before next milestone.",
                        )
                    )
                    break
        return open_loops
