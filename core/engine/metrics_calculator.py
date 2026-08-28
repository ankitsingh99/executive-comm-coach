"""
Deterministic Metrics Calculator for Executive Communication Analysis.
Calculates presence, assertiveness, active listening, and filler word statistics.
"""

import re
from typing import List, Dict, Tuple
from .schema import Utterance, FillerWordMetric, CommunicationMetrics


# Multi-word phrase fillers (English + Hinglish)
PHRASE_FILLER_PATTERNS = [
    (r"\byou know\b", "you know"),
    (r"\bi mean\b", "i mean"),
    (r"\bsort of\b", "sort of"),
    (r"\bkind of\b", "kind of"),
    (r"\btheek hai\b", "theek hai"),
    (r"\bsamajh gaya\b", "samajh gaya"),
    (r"\bmatlab ki\b", "matlab ki"),
    (r"\baisa hai ki\b", "aisa hai ki"),
    (r"\bdekha jaye toh\b", "dekha jaye toh"),
]

# Single-token word & phonetic hesitation patterns
TOKEN_FILLER_PATTERNS = [
    # Phonetic hesitation sounds
    r"\bu+m+\b",          # um, umm, ummm...
    r"\bu+h+m*\b",        # uh, uhh, uhhh...
    r"\be+r+m*\b",        # er, err, erm...
    r"\be+r+\b",          # er, err...
    r"\bh+m+\b",          # hm, hmm, hmmm...
    r"\bm+h+m*\b",        # mhm, mmhmm...
    r"\ba+h+\b",          # ah, ahh, ahhh...
    r"\ba{2,}\b",         # aa, aaa...
    r"\ba+a+h*\b",        # aah, aaah...
    r"\be+h+\b",          # eh, ehh...
    r"\bo+h+\b",          # oh, ohh...
    r"\bo{2,}h*\b",       # ooh, oohh...
    
    # English lexical fillers
    r"\bbasically\b",
    r"\bactually\b",
    r"\bliterally\b",
    r"\blike\b",
    r"\bright\b",
    
    # Hinglish & South Asian discourse fillers
    r"\bmatlab\b",
    r"\byaani\b",
    r"\barre\b",
    r"\bhaina\b",
    r"\bhaan\b",
    r"\bacha\b",
    r"\btoh\b",
    r"\byaar\b",
    r"\bbhai\b",
    r"\bwaise\b",
]

# Consolidated filler patterns
FILLER_PATTERNS = list(TOKEN_FILLER_PATTERNS) + [p for p, _ in PHRASE_FILLER_PATTERNS]

# Self-diminishing / hedging qualifiers (English + Hinglish)
HEDGING_PATTERNS = [
    # English
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
    r"\bmaybe\b",
    
    # Hinglish
    r"\bmujhe\s+(?:bhi\s+)?lagta\s+hai\b",
    r"\blag\s+raha\s+hai(?:\s+ki)?\b",
    r"\bshayad\b",
    r"\bagar\s+possible\s+ho\s+toh\b",
    r"\bagar\s+ho\s+sake\s+toh\b",
    r"\bthoda\s+(?:sa\s+)?doubt\s+hai\b",
    r"\bthoda\s+confusion\s+hai\b",
    r"\bmain\s+sure\s+nahi\s+hu\b",
    r"\bpata\s+nahi\s+but\b",
    r"\bmere\s+khayal\s+se\b",
    r"\baisa\s+lag\s+raha\s+tha\b",
    r"\bthoda\s+time\s+lag\s+sakta\s+hai\b",
]

# Strong definitive assertion markers (English + Hinglish)
ASSERTIVE_PATTERNS = [
    # English
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
    r"\bwe will ship\b",
    r"\bwe will deploy\b",
    
    # Hinglish
    r"\bhumne\s+decide\s+kiya\s+hai\b",
    r"\bpakka\s+(?:hum\s+)?kar\s+denge\b",
    r"\bmera\s+recommendation\s+hai\b",
    r"\bhumara\s+recommendation\s+hai\b",
    r"\bdata\s+dikhata\s+hai\b",
    r"\bnumbers\s+clear\s+hai\b",
    r"\bfinal\s+decision\s+ye\s+hai\b",
    r"\bhum\s+ship\s+karenge\b",
    r"\bhum\s+deploy\s+karenge\b",
    r"\bpriority\s+ye\s+honi\s+chahiye\b",
    r"\bblocker\s+ye\s+hai\b",
    r"\bhum\s+achieve\s+karenge\b",
]

# Active listening & validation markers (English + Hinglish)
ACTIVE_LISTENING_PATTERNS = [
    # English
    r"\bbuilding on what you said\b",
    r"\bto confirm\b",
    r"\bif i understand correctly\b",
    r"\bundertood\b",
    r"\bunderstood\b",
    r"\bthat makes sense\b",
    r"\bgood point\b",
    r"\bi see your point\b",
    r"\bwhat do you think about\b",
    r"\bhow do you see this\b",
    r"\bwhat are your thoughts\b",
    
    # Hinglish
    r"\bsahi\s+point\s+hai\b",
    r"\bsahi\s+baat\s+hai\b",
    r"\baapka\s+point\s+samajh\s+aaya\b",
    r"\baapka\s+point\s+clear\s+hai\b",
    r"\bbilkul\s+sahi\b",
    r"\btheek\s+baat\s+hai\b",
    r"\baapka\s+kya\s+kehna\s+hai\b",
    r"\baapko\s+kya\s+lagta\s+hai\b",
    r"\baap\s+bataiye\b",
    r"\bkya\s+lagta\s+hai\s+aapko\b",
]


class MetricsCalculator:
    """Calculates quantitative communication benchmarks from dialogue transcripts."""

    @classmethod
    def detect_fillers(cls, text: str) -> List[FillerWordMetric]:
        """Detects and tallies verbal and phonetic filler words."""
        counts: Dict[str, int] = {}
        working_text = text

        # 1. Match multi-word phrases first
        for pattern, label in PHRASE_FILLER_PATTERNS:
            matches = re.findall(pattern, working_text, flags=re.IGNORECASE)
            if matches:
                counts[label] = len(matches)
                # Replace with placeholder to prevent double matching
                working_text = re.sub(pattern, " ", working_text, flags=re.IGNORECASE)

        # 2. Match single tokens and phonetic vocalizations
        for pattern in TOKEN_FILLER_PATTERNS:
            matches = re.findall(pattern, working_text, flags=re.IGNORECASE)
            for m in matches:
                token_clean = m.lower().strip()
                counts[token_clean] = counts.get(token_clean, 0) + 1
            if matches:
                working_text = re.sub(pattern, " ", working_text, flags=re.IGNORECASE)

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
