"""
Deterministic Metrics Calculator for Executive Communication Analysis.
Calculates presence, assertiveness, active listening, and filler word statistics.
Supports bilingual English, Hindi, and code-mixed Hinglish.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from .schema import Utterance, FillerWordMetric, CommunicationMetrics

try:
    from ..asr_diarization.indic_normalizer import IndicNormalizer
except (ImportError, ValueError):
    from asr_diarization.indic_normalizer import IndicNormalizer


# Multi-word phrase fillers (English + Hinglish + Devanagari)
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
    (r"\bkya bolte ho\b", "kya bolte ho"),
    (r"\barre yaar\b", "arre yaar"),
    (r"\bactually matlab\b", "actually matlab"),
    (r"\bbasically yaar\b", "basically yaar"),
    (r"\b(matlab\s+ki\s+dekho)\b", "matlab ki dekho"),
]

# Single-token word, phonetic hesitation, & non-verbal sound patterns (English + Hinglish + Devanagari)
TOKEN_FILLER_PATTERNS = [
    # Non-verbal vocal sounds, tongue clicks, & tut-tuts
    r"\b(?:tch|tsk|tck|tskk)(?:[- ](?:tch|tsk|tck|tskk))*\b",  # tch, tsk, tch-tch, tsk-tsk...
    r"\b(?:uff|oof|ugh|argh|ahem|pfft|pshh|shh)\b",  # sigh, exhalation, throat clearing
    r"\b(?:huh|hunh)\b",  # vocal confusion / query sound
    # Phonetic hesitation sounds & vocal elongations
    r"\bu+m+\b",  # um, umm, ummm...
    r"\bu+h+m*\b",  # uh, uhh, uhhh, uhm...
    r"\be+r+m*\b",  # er, err, erm...
    r"\be+r+\b",  # er, err...
    r"\bh+m+\b",  # hm, hmm, hmmm...
    r"\bm+h+m*\b",  # mhm, mmhmm...
    r"\ba+h+\b",  # ah, ahh, ahhh...
    r"\ba{2,}\b",  # aa, aaa, aaaa...
    r"\ba+a+h*\b",  # aah, aaah...
    r"\be+h+\b",  # eh, ehh...
    r"\bo+h+\b",  # oh, ohh...
    r"\bo{2,}h*\b",  # ooh, oohh...
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
    r"\baccha\b",
    r"\btoh\b",
    r"\byaar\b",
    r"\bbhai\b",
    r"\bwaise\b",
    r"\bdekho\b",
    r"\bsuno\b",
    r"\bna\b",
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
    r"\bmujhe\s+aisa\s+lagta\s+hai\b",
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
    r"\bgalat\s+ho\s+sakta\s+hu\b",
    r"\bshyd\b",
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
    r"\bdata\s+saaf\s+dikhata\s+hai\b",
    r"\bnumbers\s+clear\s+hai\b",
    r"\bfinal\s+decision\s+ye\s+hai\b",
    r"\bhum\s+ship\s+karenge\b",
    r"\bhum\s+deploy\s+karenge\b",
    r"\bpriority\s+ye\s+honi\s+chahiye\b",
    r"\bblocker\s+ye\s+hai\b",
    r"\bhum\s+achieve\s+karenge\b",
    r"\bhume\s+karna\s+hi\s+hoga\b",
    r"\byeh\s+zaroori\s+hai\b",
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
    r"\bsahi\s+bol\s+rahe\s+ho\b",
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
        working_text = IndicNormalizer.normalize_text(text)

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
        norm_text = IndicNormalizer.normalize_text(text)
        hedging_count = sum(len(re.findall(pat, norm_text, flags=re.IGNORECASE)) for pat in HEDGING_PATTERNS)
        assertive_count = sum(len(re.findall(pat, norm_text, flags=re.IGNORECASE)) for pat in ASSERTIVE_PATTERNS)
        return hedging_count, assertive_count

    @classmethod
    def calculate_active_listening_signals(cls, user_text: str, counterpart_text: str) -> int:
        """Counts instances of validation, inquiry, and acknowledgment."""
        norm_user = IndicNormalizer.normalize_text(user_text)
        return sum(len(re.findall(pat, norm_user, flags=re.IGNORECASE)) for pat in ACTIVE_LISTENING_PATTERNS)

    @classmethod
    def analyze_dialogue(
        cls,
        utterances: List[Utterance],
        target_speaker: str = "USER",
        acoustic_fillers: Optional[List[Any]] = None,
    ) -> CommunicationMetrics:
        """Computes dynamic [0-100] communication metrics across dialogue."""
        if not utterances:
            return CommunicationMetrics(
                presence_score=75, assertiveness_score=75, active_listening_score=70, filler_words_detected=[]
            )

        # Identify user utterances with support for voiceprint/intro names
        target_up = (target_speaker or "USER").strip().upper()
        user_synonyms = {target_up, "USER", "SELF", "YOU", "ASHISH"}

        user_utterances = [u for u in utterances if u.speaker.strip().upper() in user_synonyms]
        counterpart_utterances = [u for u in utterances if u.speaker.strip().upper() not in user_synonyms]

        unique_spks = list(dict.fromkeys(u.speaker for u in utterances))
        is_solo = len(unique_spks) <= 1

        # If no utterances matched user_synonyms, map based on dialogue structure
        if not user_utterances and utterances:
            if is_solo:
                user_utterances = list(utterances)
                counterpart_utterances = []
            else:
                user_utterances = [u for u in utterances if u.speaker == unique_spks[0]]
                counterpart_utterances = [u for u in utterances if u.speaker != unique_spks[0]]

        user_text = IndicNormalizer.normalize_text(" ".join(u.transcript for u in user_utterances))
        counterpart_text = IndicNormalizer.normalize_text(" ".join(u.transcript for u in counterpart_utterances))
        total_words = len(user_text.split()) or 1

        # 1. Filler words & phonetic hesitation
        fillers = cls.detect_fillers(user_text)

        # Merge acoustic non-phonetic fillers if supplied
        if acoustic_fillers:
            filler_dict = {f.token: f.count for f in fillers}
            for af in acoustic_fillers:
                spk = getattr(af, "speaker", "USER").upper()
                if spk in user_synonyms or is_solo:
                    tok = getattr(af, "token", "umm").lower()
                    # Add acoustic filler count if not already reflected in text
                    filler_dict[tok] = filler_dict.get(tok, 0) + 1
            sorted_f = sorted(filler_dict.items(), key=lambda x: x[1], reverse=True)
            fillers = [FillerWordMetric(token=k, count=v) for k, v in sorted_f]

        total_fillers = sum(f.count for f in fillers)
        filler_rate_per_100_words = (total_fillers / total_words) * 100

        # 2. Assertiveness (conviction vs hedging)
        hedging_count, assertive_count = cls.calculate_hedging_vs_assertion(user_text)
        if assertive_count > 0 or hedging_count > 0:
            raw_assertiveness = 75 - (hedging_count * 8) + (assertive_count * 6)
        else:
            # Baseline certainty for clean declarative speech without hedging
            raw_assertiveness = 80 if total_words > 8 else 75
        assertiveness_score = max(15, min(98, int(raw_assertiveness)))

        # 3. Active Listening & Inquiry / Turn-taking dynamics
        listening_signals = cls.calculate_active_listening_signals(user_text, counterpart_text)
        question_count = user_text.count("?")

        user_interruptions = sum(1 for u in user_utterances if getattr(u, "interrupted_speaker", None))
        total_overlaps = sum(1 for u in utterances if getattr(u, "is_overlapping", False))
        interruption_penalty = min(28, user_interruptions * 7)

        if is_solo:
            # In Solo Practice: measures structured inquiry, self-prompting questions, and clear pacing
            if question_count > 0:
                raw_listening = 80 + min(15, question_count * 5)
            else:
                raw_listening = 72
        else:
            # In Multi-Speaker: measures validation signals, question asking, and avoidance of interruptions
            raw_listening = 65 + (listening_signals * 10) + (question_count * 5) - interruption_penalty

        active_listening_score = max(15, min(98, int(raw_listening)))

        # 4. Presence Score (Combines brevity, filler freedom, assertiveness, and delivery cadence)
        filler_penalty = min(35, int(filler_rate_per_100_words * 6))
        raw_presence = (assertiveness_score * 0.5) + (active_listening_score * 0.3) + 20 - filler_penalty
        presence_score = max(15, min(98, int(raw_presence)))

        return CommunicationMetrics(
            presence_score=presence_score,
            assertiveness_score=assertiveness_score,
            active_listening_score=active_listening_score,
            filler_words_detected=fillers,
            interruption_count=user_interruptions,
            overlap_count=total_overlaps,
        )
