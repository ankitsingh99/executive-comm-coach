"""
Speaker Diarization and Transcript Alignment Engine (Stage 3).
Segments audio stream into aligned user vs counterpart utterance intervals.
"""

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


class DiarizationEngine:
    """Aligns speaker clusters into USER and COUNTERPART turns."""

    @classmethod
    def assign_roles(
        cls,
        raw_utterances: List[Utterance],
        user_speaker_id: str = "SPEAKER_01"
    ) -> List[Utterance]:
        """Maps acoustic speaker clusters (e.g. SPEAKER_01, SPEAKER_02) to USER and COUNTERPART."""
        normalized: List[Utterance] = []
        for u in raw_utterances:
            speaker_label = "USER" if u.speaker in [user_speaker_id, "USER"] else "COUNTERPART"
            normalized.append(
                Utterance(
                    speaker=speaker_label,
                    start_time=u.start_time,
                    end_time=u.end_time,
                    transcript=u.transcript.strip()
                )
            )
        return normalized

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
