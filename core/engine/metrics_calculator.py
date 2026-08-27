"""
Deterministic Metrics Calculator for Executive Communication Analysis.
Calculates presence, assertiveness, active listening, and filler word statistics.
"""

import re
from typing import List, Dict, Tuple
from .schema import Utterance, FillerWordMetric, CommunicationMetrics


# Recognized filler words in corporate Hinglish / English dialogue
FILLER_PATTERNS = [
    r"\bmatlab\b",
    r"\bbasically\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bactually\b",
    r"\byaani\b",
    r"\barre\b",
    r"\bhaina\b",
    r"\bumm?\b",
    r"\buhh?\b",
    r"\bi mean\b",
    r"\bsort of\b",
    r"\bkind of\b",
]

# Self-diminishing / hedging qualifiers
HEDGING_PATTERNS = [
    r"\bi just think\b",
    r"\bi just wanted to\b",
    r"\bmaybe we could possibly\b",
    r"\bsorry to bother you\b",
    r"\bi might be wrong but\b",
    r"\bi'm not totally sure but\b",
    r"\bif it's not too much trouble\b",
    r"\bjust checking in\b",
    r"\bdoes that make sense\b",
    r"\bif you don't mind\b",
    r"\bperhaps maybe\b",
    r"\bi was just wondering\b",
]

# Strong definitive assertion markers
ASSERTIVE_PATTERNS = [
    r"\bour data demonstrates\b",
    r"\bi recommend\b",
    r"\bwe have decided\b",
    r"\bthe direct impact is\b",
    r"\bthe priority is\b",
    r"\bi propose\b",
    r"\bwe need to focus on\b",
    r"\bthe conclusion is\b",
    r"\bwe will achieve\b",
    r"\bthe blocker is\b",
    r"\bour analysis shows\b",
]

# Active listening & validation markers
ACTIVE_LISTENING_PATTERNS = [
    r"\bbuilding on what you said\b",
    r"\bto confirm\b",
    r"\bif i understand correctly\b",
    r"\bundertood\b",
    r"\bunderstood\b",
    r"\bthat makes sense\b",
    r"\bgood point\b",
    r"\bsahi point hai\b",
    r"\bi see your point\b",
    r"\bwhat do you think about\b",
    r"\bhow do you see this\b",
    r"\bwhat are your thoughts\b",
]


class MetricsCalculator:
    """Calculates quantitative communication benchmarks from dialogue transcripts."""

    @classmethod
    def detect_fillers(cls, text: str) -> List[FillerWordMetric]:
        """Detects and tallies verbal filler words."""
        counts: Dict[str, int] = {}
        for pattern in FILLER_PATTERNS:
            token = pattern.replace(r"\b", "")
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                counts[token] = len(matches)
        
        # Sort by highest frequency
        sorted_fillers = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [FillerWordMetric(token=k, count=v) for k, v in sorted_fillers]

    @classmethod
    def calculate_hedging_vs_assertion(cls, text: str) -> Tuple[int, int]:
        """Returns (hedging_count, assertive_count)."""
        hedging_count = sum(
            len(re.findall(pat, text, flags=re.IGNORECASE))
            for pat in HEDGING_PATTERNS
        )
        assertive_count = sum(
            len(re.findall(pat, text, flags=re.IGNORECASE))
            for pat in ASSERTIVE_PATTERNS
        )
        return hedging_count, assertive_count

    @classmethod
    def calculate_active_listening_signals(cls, user_text: str, counterpart_text: str) -> int:
        """Counts instances of validation, inquiry, and acknowledgment."""
        return sum(
            len(re.findall(pat, user_text, flags=re.IGNORECASE))
            for pat in ACTIVE_LISTENING_PATTERNS
        )

    @classmethod
    def analyze_dialogue(cls, utterances: List[Utterance], target_speaker: str = "USER") -> CommunicationMetrics:
        """Computes baseline [0-100] communication metrics across dialogue."""
        user_utterances = [u for u in utterances if u.speaker.upper() == target_speaker.upper()]
        counterpart_utterances = [u for u in utterances if u.speaker.upper() != target_speaker.upper()]

        user_text = " ".join(u.transcript for u in user_utterances)
        counterpart_text = " ".join(u.transcript for u in counterpart_utterances)
        total_words = len(user_text.split()) or 1

        # 1. Filler words
        fillers = cls.detect_fillers(user_text)
        total_fillers = sum(f.count for f in fillers)
        filler_rate_per_100_words = (total_fillers / total_words) * 100

        # 2. Assertiveness
        hedging_count, assertive_count = cls.calculate_hedging_vs_assertion(user_text)
        # Base score 75. Penalize hedging (-8 each), reward assertions (+6 each)
        raw_assertiveness = 75 - (hedging_count * 8) + (assertive_count * 6)
        assertiveness_score = max(10, min(100, int(raw_assertiveness)))

        # 3. Active Listening
        listening_signals = cls.calculate_active_listening_signals(user_text, counterpart_text)
        # Check turn alternation and question asking
        question_count = user_text.count("?")
        raw_listening = 60 + (listening_signals * 10) + (question_count * 5)
        active_listening_score = max(10, min(100, int(raw_listening)))

        # 4. Presence Score (Combines brevity, low filler rate, and assertiveness)
        filler_penalty = min(35, int(filler_rate_per_100_words * 5))
        raw_presence = (assertiveness_score * 0.5) + (active_listening_score * 0.3) + 20 - filler_penalty
        presence_score = max(10, min(100, int(raw_presence)))

        return CommunicationMetrics(
            presence_score=presence_score,
            assertiveness_score=assertiveness_score,
            active_listening_score=active_listening_score,
            filler_words_detected=fillers
        )
