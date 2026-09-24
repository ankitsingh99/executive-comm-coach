"""
Speaker Diarization, Verbal Name Recognition, and Transcript Alignment Engine (Stage 3).
Segments audio stream into aligned user vs counterpart utterance intervals,
recognizes verbal self-introductions (e.g. 'Hey I am Rahul and today...'),
and assigns rich speaker tags for both dialogue and solo speeches.
"""

import re
from typing import List, Tuple, Optional

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


STOP_WORDS = {
    "and",
    "from",
    "today",
    "with",
    "for",
    "at",
    "in",
    "the",
    "to",
    "here",
    "speaking",
    "of",
    "lead",
    "who",
    "will",
    "leading",
    "as",
    "is",
    "on",
    "by",
    "fine",
    "good",
    "thinking",
    "trying",
    "going",
    "ready",
    "sure",
    "happy",
    "excited",
    "working",
    "sorry",
    "back",
    "done",
    "there",
    "just",
    "really",
    "not",
    "great",
    "cool",
    "looking",
    "wondering",
    "hoping",
    "a",
    "an",
    "also",
    "so",
    "now",
    "we",
    "i",
    "our",
    "all",
    "welcome",
    "thanks",
    "thank",
    "thing",
    "things",
    "point",
    "points",
    "issue",
    "issues",
    "problem",
    "problems",
    "case",
    "step",
    "way",
    "more",
    "most",
    "important",
    "crucial",
    "main",
    "key",
    "clear",
    "right",
    "over",
    "out",
    "up",
    "down",
    "first",
    "second",
    "time",
    "people",
    "someone",
    "everyone",
    "anyone",
    "something",
    "anything",
    "nothing",
    "fact",
    "reason",
    "part",
    "side",
}

SELF_INTRO_PATTERNS = [
    re.compile(
        r"\b(?:hey|hi|hello|namaste|good\s+morning|good\s+afternoon|good\s+evening)?\s*(?:,\s*)?(?:i\s*am|i['’]m|this\s+is|my\s+name\s+is|myself|it['’]s)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.?!;]\s*|\b(?:hey|hi|hello|namaste)\s+)([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:here|this\s+side|speaking)\b",
        re.IGNORECASE,
    ),
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
        cls, utterances: List[Utterance], user_speaker_id: str = "USER"
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
            elif detected_counterpart and (
                u.speaker == target_counterpart_tag or spk_up in {"COUNTERPART", "SPEAKER_02", "SPEAKER_2", "OTHER"}
            ):
                new_spk = detected_counterpart
            else:
                new_spk = u.speaker

            updated.append(
                Utterance(speaker=new_spk, start_time=u.start_time, end_time=u.end_time, transcript=u.transcript)
            )

        return updated, detected_counterpart, detected_user

    @classmethod
    def compute_overlapping_speech(cls, utterances: List[Utterance]) -> Tuple[List[Utterance], int, float, int]:
        """
        Detects temporal overlaps (simultaneous speech / cross-talk) between distinct speakers.
        Identifies who interrupted whom when a speaker starts talking before the prior speaker stops.
        Returns (updated_utterances, total_overlap_events, total_overlap_duration_sec, total_interruptions).
        """
        if len(utterances) < 2:
            return utterances, 0, 0.0, 0

        # Sort utterances by start_time
        sorted_utts = sorted(utterances, key=lambda x: (x.start_time, x.end_time))
        total_overlap_events = 0
        total_overlap_duration_sec = 0.0
        total_interruptions = 0

        # Compare pairs of utterances for temporal overlap
        for i in range(len(sorted_utts)):
            u_i = sorted_utts[i]
            for j in range(i + 1, len(sorted_utts)):
                u_j = sorted_utts[j]

                # Stop searching ahead if the next utterance starts after current one ends
                if u_j.start_time >= u_i.end_time:
                    break

                # Overlap only counts if spoken by different individuals
                if u_i.speaker.strip().upper() != u_j.speaker.strip().upper():
                    overlap_start = max(u_i.start_time, u_j.start_time)
                    overlap_end = min(u_i.end_time, u_j.end_time)
                    overlap_dur = max(0.0, overlap_end - overlap_start)

                    # Threshold: 150ms minimum simultaneous speech to count as meaningful overlap
                    if overlap_dur >= 0.15:
                        total_overlap_events += 1
                        total_overlap_duration_sec += overlap_dur

                        u_i.is_overlapping = True
                        u_j.is_overlapping = True
                        u_i.overlap_duration_sec = max(u_i.overlap_duration_sec, round(overlap_dur, 2))
                        u_j.overlap_duration_sec = max(u_j.overlap_duration_sec, round(overlap_dur, 2))

                        # Determine interruption: if u_j started while u_i was still speaking
                        if u_j.start_time > u_i.start_time + 0.1:
                            u_j.interrupted_speaker = u_i.speaker
                            total_interruptions += 1
                        elif u_i.start_time > u_j.start_time + 0.1:
                            u_i.interrupted_speaker = u_j.speaker
                            total_interruptions += 1

        return sorted_utts, total_overlap_events, round(total_overlap_duration_sec, 2), total_interruptions

    @classmethod
    def assign_roles(
        cls,
        raw_utterances: List[Utterance],
        user_speaker_id: str = "USER",
        recognized_counterpart_name: Optional[str] = None,
        recognized_user_name: Optional[str] = None,
    ) -> List[Utterance]:
        """
        Maps acoustic speaker clusters (e.g. SPEAKER_01, speaker_0, USER) to USER/Name and COUNTERPART/Name,
        and computes overlapping speech intervals.
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
                    transcript=u.transcript.strip(),
                    is_overlapping=getattr(u, "is_overlapping", False),
                    overlap_duration_sec=getattr(u, "overlap_duration_sec", 0.0),
                    interrupted_speaker=getattr(u, "interrupted_speaker", None),
                )
            )

        # Run overlap analysis
        processed_utterances, _, _, _ = cls.compute_overlapping_speech(normalized)
        return processed_utterances

    @classmethod
    def format_dialogue_cli(
        cls, utterances: List[Utterance], user_name: Optional[str] = None, counterpart_name: Optional[str] = None
    ) -> str:
        """
        Formats dialogue turns into visually aligned CLI output with speaker tags, timestamps, and overlap indicators.
        """
        lines = []
        unique_spks = {u.speaker for u in utterances}
        is_solo = len(unique_spks) <= 1

        for u in utterances:
            st = u.start_time
            et = u.end_time
            if et <= st:
                et = st + max(1.0, round(len(u.transcript.split()) / 2.5, 1))
            start_m, start_s = divmod(int(round(st)), 60)
            end_m, end_s = divmod(int(round(et)), 60)
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

            # Overlap / Interruption visual indicators
            overlap_annotation = ""
            if getattr(u, "is_overlapping", False):
                if getattr(u, "interrupted_speaker", None):
                    overlap_annotation = f" \033[93m⚡ [INTERRUPTED {u.interrupted_speaker}]\033[0m"
                else:
                    overlap_annotation = f" \033[93m⚡ [OVERLAP {u.overlap_duration_sec:.1f}s]\033[0m"

            lines.append(f'    • {tag:<24} {time_tag}{overlap_annotation}: "{u.transcript}"')
        return "\n".join(lines)

    @classmethod
    def format_dialogue_markdown(cls, utterances: List[Utterance]) -> str:
        """Formats the dialogue into clean Markdown turns with timestamps and overlap badges."""
        lines = []
        for u in utterances:
            start_m, start_s = divmod(int(u.start_time), 60)
            end_m, end_s = divmod(int(u.end_time), 60)
            time_tag = f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]"
            overlap_badge = ""
            if getattr(u, "is_overlapping", False):
                if getattr(u, "interrupted_speaker", None):
                    overlap_badge = f" *(⚡ Interrupted {u.interrupted_speaker})*"
                else:
                    overlap_badge = f" *(⚡ Overlapping Speech {u.overlap_duration_sec:.1f}s)*"
            lines.append(f'**{u.speaker}** {time_tag}{overlap_badge}: "{u.transcript}"')
        return "\n\n".join(lines)
