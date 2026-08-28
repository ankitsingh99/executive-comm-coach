"""
Speaker Diarization, Verbal Name Recognition, and Transcript Alignment Engine (Stage 3).
Segments audio stream into aligned user vs counterpart utterance intervals,
recognizes verbal self-introductions (e.g. 'Hey I am Rahul and today...'),
and assigns rich speaker tags for both dialogue and solo speeches.
"""

import re
from typing import List, Dict, Tuple, Optional

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


STOP_WORDS = {
    'and', 'from', 'today', 'with', 'for', 'at', 'in', 'the', 'to', 'here',
    'speaking', 'of', 'lead', 'who', 'will', 'leading', 'as', 'is', 'on', 'by',
    'fine', 'good', 'thinking', 'trying', 'going', 'ready', 'sure', 'happy',
    'excited', 'working', 'sorry', 'back', 'done', 'there', 'just', 'really',
    'not', 'great', 'cool', 'looking', 'wondering', 'hoping', 'a', 'an', 'also',
    'so', 'now', 'we', 'i', 'our', 'all', 'welcome', 'thanks', 'thank'
}

SELF_INTRO_PATTERNS = [
    re.compile(r"\b(?:hey|hi|hello|namaste|good\s+morning|good\s+afternoon|good\s+evening)?\s*(?:,\s*)?(?:i\s*am|i['’]m|this\s+is|my\s+name\s+is|myself|it['’]s)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:here|this\s+side|speaking)\b", re.IGNORECASE),
    re.compile(r"\b(?:hey|hi|hello)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+here\b", re.IGNORECASE),
]


class DiarizationEngine:
    """Aligns speaker clusters into USER and COUNTERPART / Name turns and detects verbal self-introductions."""

    @classmethod
    def extract_speaker_name_from_text(cls, text: str) -> Optional[str]:
        """
        Detects self-introductions like 'Hey I am Rahul and today...', 'Vikram here',
        'This is Priya from...', and extracts the spoken name.
        """
        for pattern in SELF_INTRO_PATTERNS:
            match = pattern.search(text)
            if match:
                raw_candidate = match.group(1).strip()
                words = raw_candidate.split()
                valid_words = []
                for w in words:
                    if w.lower() in STOP_WORDS:
                        break
                    valid_words.append(w.title())
                if valid_words and len(valid_words[0]) >= 2:
                    return " ".join(valid_words)
        return None

    @classmethod
    def detect_and_apply_verbal_introductions(
        cls,
        utterances: List[Utterance],
        user_speaker_id: str = "USER"
    ) -> Tuple[List[Utterance], Optional[str], Optional[str]]:
        """
        Scans utterances for verbal introductions (e.g. 'Hey I am Rahul and today...').
        Detects counterpart introductions as well as solo user introductions.
        Returns (updated_utterances, detected_counterpart_name, detected_user_name).
        """
        detected_counterpart: Optional[str] = None
        detected_user: Optional[str] = None
        target_counterpart_tag: Optional[str] = None

        user_synonyms = {user_speaker_id.upper(), "USER", "SPEAKER_01", "SPEAKER_0", "SPEAKER 1", "SPEAKER_1", "SELF"}

        for u in utterances:
            spk_up = u.speaker.strip().upper()
            name = cls.extract_speaker_name_from_text(u.transcript)
            if name:
                if spk_up in user_synonyms or len(utterances) == 1:
                    if not detected_user:
                        detected_user = name
                else:
                    if not detected_counterpart:
                        detected_counterpart = name
                        target_counterpart_tag = u.speaker

        # Re-tag the dialogue turns if any names were found
        updated: List[Utterance] = []
        for u in utterances:
            spk_up = u.speaker.strip().upper()
            if spk_up in user_synonyms and detected_user:
                new_spk = detected_user
            elif detected_counterpart and (u.speaker == target_counterpart_tag or spk_up in {"COUNTERPART", "SPEAKER_02", "SPEAKER_2", "OTHER"}):
                new_spk = detected_counterpart
            else:
                new_spk = u.speaker

            updated.append(
                Utterance(
                    speaker=new_spk,
                    start_time=u.start_time,
                    end_time=u.end_time,
                    transcript=u.transcript
                )
            )

        return updated, detected_counterpart, detected_user

    @classmethod
    def assign_roles(
        cls,
        raw_utterances: List[Utterance],
        user_speaker_id: str = "USER",
        recognized_counterpart_name: Optional[str] = None,
        recognized_user_name: Optional[str] = None
    ) -> List[Utterance]:
        """
        Maps acoustic speaker clusters (e.g. SPEAKER_01, speaker_0, USER) to USER/Name and COUNTERPART/Name.
        """
        user_synonyms = {user_speaker_id.upper(), "USER", "SPEAKER_01", "SPEAKER_0", "SPEAKER 1", "SPEAKER_1", "SELF"}
        if recognized_user_name:
            user_synonyms.add(recognized_user_name.upper())

        counterpart_synonyms = {"COUNTERPART", "SPEAKER_02", "SPEAKER_1", "SPEAKER 2", "SPEAKER_2", "OTHER"}
        user_label = recognized_user_name.strip() if recognized_user_name else "USER"
        counterpart_label = recognized_counterpart_name.strip() if recognized_counterpart_name else "COUNTERPART"

        normalized: List[Utterance] = []
        for u in raw_utterances:
            spk_upper = u.speaker.strip().upper()
            if spk_upper in user_synonyms:
                speaker_label = user_label
            elif spk_upper in counterpart_synonyms:
                speaker_label = counterpart_label
            else:
                speaker_label = u.speaker.strip() or user_label

            normalized.append(
                Utterance(
                    speaker=speaker_label,
                    start_time=round(u.start_time, 2),
                    end_time=round(u.end_time, 2),
                    transcript=u.transcript.strip()
                )
            )
        return normalized

    @classmethod
    def format_dialogue_cli(
        cls,
        utterances: List[Utterance],
        user_name: Optional[str] = None,
        counterpart_name: Optional[str] = None
    ) -> str:
        """
        Formats dialogue turns into visually aligned CLI output with speaker tags and timestamps.
        """
        lines = []
        unique_spks = {u.speaker for u in utterances}
        is_solo = len(unique_spks) <= 1

        for u in utterances:
            start_m, start_s = divmod(int(u.start_time), 60)
            end_m, end_s = divmod(int(u.end_time), 60)
            time_tag = f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]"

            if u.speaker in ["USER", "Self"] or (user_name and u.speaker == user_name):
                if user_name:
                    tag = f"[{user_name.upper()} (Solo)]" if is_solo else f"[{user_name.upper()} / YOU]"
                else:
                    tag = "[USER (Solo)]" if is_solo else "[USER / YOU]"
            elif u.speaker == "COUNTERPART" and counterpart_name:
                tag = f"[{counterpart_name.upper()}]"
            else:
                tag = f"[{u.speaker}]"

            lines.append(f"    • {tag:<24} {time_tag}: \"{u.transcript}\"")
        return "\n".join(lines)

    @classmethod
    def format_dialogue_markdown(cls, utterances: List[Utterance]) -> str:
        """Formats the dialogue into clean Markdown turns with timestamps."""
        lines = []
        for u in utterances:
            start_m, start_s = divmod(int(u.start_time), 60)
            end_m, end_s = divmod(int(u.end_time), 60)
            time_tag = f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]"
            lines.append(f"**{u.speaker}** {time_tag}: \"{u.transcript}\"")
        return "\n\n".join(lines)
