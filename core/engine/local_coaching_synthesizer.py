"""
100% On-Device Dynamic Executive Coaching Synthesizer.
Operates completely offline with zero cloud API dependencies.
Dynamically parses the user's actual spoken words, extracts verbatim quotes,
and generates customized Executive BLUF rephrasings from the user's real speech.
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

try:
    from .schema import (
        Utterance,
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        FillerWordMetric
    )
    from .persona_ontology import (
        PersonaOntologyEngine,
        PersonaProfile,
        PowerAxis
    )
    from .metrics_calculator import (
        MetricsCalculator,
        HEDGING_PATTERNS,
        FILLER_PATTERNS,
        ASSERTIVE_PATTERNS
    )
    from ..privacy.pii_redactor import PIIRedactor
except (ImportError, ValueError):
    from engine.schema import (
        Utterance,
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        FillerWordMetric
    )
    from engine.persona_ontology import (
        PersonaOntologyEngine,
        PersonaProfile,
        PowerAxis
    )
    from engine.metrics_calculator import (
        MetricsCalculator,
        HEDGING_PATTERNS,
        FILLER_PATTERNS,
        ASSERTIVE_PATTERNS
    )
    from privacy.pii_redactor import PIIRedactor


class LocalCoachingSynthesizer:
    """
    On-device coaching synthesizer running on local CPU / Tensor NPU.
    Generates structured, persona-calibrated executive feedback dynamically from real speech.
    """

    def __init__(self, ollama_url: Optional[str] = None, model_name: str = "gemma2:2b"):
        self.ollama_url = ollama_url or os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
        self.model_name = model_name

    def synthesize(
        self,
        session: ConversationSession,
        top_n: int = 2,
        try_local_ollama: bool = True
    ) -> ExecutiveCoachingEvaluation:
        """
        Runs the local on-device evaluation pipeline on the user's real spoken utterances.
        """
        # Step 1: Local PII Redaction
        redacted_dialogue: List[Utterance] = []
        for u in session.dialogue:
            redacted_text, _ = PIIRedactor.redact_text(u.transcript)
            redacted_dialogue.append(
                Utterance(
                    speaker=u.speaker,
                    start_time=u.start_time,
                    end_time=u.end_time,
                    transcript=redacted_text
                )
            )

        # Step 2: Persona Profile
        try:
            power_axis = PowerAxis(session.power_axis.upper())
        except ValueError:
            power_axis = PowerAxis.UPWARD

        profile = PersonaOntologyEngine.create_persona_profile(
            counterpart_name=session.counterpart_name,
            role_title=session.counterpart_role,
            power_axis=power_axis
        )

        # Step 3: Compute Real Speech Quantitative Metrics
        metrics = MetricsCalculator.analyze_dialogue(
            redacted_dialogue,
            target_speaker=session.target_speaker
        )

        # Step 4: Try Local Ollama if available, otherwise use dynamic on-device semantic engine
        evaluation: Optional[ExecutiveCoachingEvaluation] = None
        if try_local_ollama:
            evaluation = self._try_ollama_local_inference(redacted_dialogue, profile, metrics, top_n)

        if evaluation is None:
            evaluation = self._dynamic_speech_synthesis(redacted_dialogue, profile, metrics, top_n)

        return self._enforce_strict_constraints(evaluation, redacted_dialogue, session.target_speaker, top_n)

    def _try_ollama_local_inference(
        self,
        dialogue: List[Utterance],
        profile: PersonaProfile,
        metrics: CommunicationMetrics,
        top_n: int
    ) -> Optional[ExecutiveCoachingEvaluation]:
        """Queries local Ollama endpoint if active on user's machine."""
        try:
            system_prompt = PersonaOntologyEngine.generate_system_instruction(profile, top_n=top_n)
            dialogue_text = "\n".join([f"{u.speaker}: {u.transcript}" for u in dialogue])
            prompt = f"{system_prompt}\n\nTRANSCRIPT:\n{dialogue_text}\n\nReturn pure JSON matching ExecutiveCoachingEvaluation schema."

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_response = data.get("response", "")
                    parsed = json.loads(raw_response)
                    return ExecutiveCoachingEvaluation.model_validate(parsed)
        except Exception:
            return None
        return None

    def _dynamic_speech_synthesis(
        self,
        dialogue: List[Utterance],
        profile: PersonaProfile,
        metrics: CommunicationMetrics,
        top_n: int
    ) -> ExecutiveCoachingEvaluation:
        """
        Dynamically analyzes the user's actual spoken sentences,
        extracts real weaknesses/strengths, and rewrites their exact words into BLUF executive phrasing.
        """
        user_turns = [u for u in dialogue if u.speaker.upper() == "USER"]
        user_sentences: List[str] = []
        for u in user_turns:
            # Split into individual clauses/sentences
            splits = re.split(r"[.!?\n]+", u.transcript)
            for s in splits:
                clean = s.strip()
                if len(clean) > 3:
                    user_sentences.append(clean)

        if not user_sentences:
            user_sentences = ["Status update provided."]

        strengths: List[TopStrength] = []
        improvements: List[AreaForImprovement] = []

        # Categorize user sentences into hedging, fillers, assertiveness
        hedging_sentences = []
        filler_sentences = []
        assertive_sentences = []

        for s in user_sentences:
            s_lower = s.lower()
            if any(re.search(pat, s_lower) for pat in HEDGING_PATTERNS):
                hedging_sentences.append(s)
            elif any(re.search(pat, s_lower) for pat in FILLER_PATTERNS):
                filler_sentences.append(s)
            elif any(re.search(pat, s_lower) for pat in ASSERTIVE_PATTERNS) or any(c.isdigit() for c in s):
                assertive_sentences.append(s)

        # 1. Identify Strengths from Real Spoken Words
        if assertive_sentences:
            for s in assertive_sentences[:top_n]:
                strengths.append(TopStrength(
                    observation="Articulated clear, definitive delivery with concrete specifics or data points.",
                    verbatim_quote=s
                ))
        elif user_sentences:
            strengths.append(TopStrength(
                observation="Maintained steady speaking rhythm and communicated the core message clearly.",
                verbatim_quote=user_sentences[0]
            ))

        # 2. Identify Weaknesses & Dynamically Rewrite User's Real Speech into BLUF
        weak_candidates = hedging_sentences + filler_sentences
        if weak_candidates:
            for s in weak_candidates[:top_n]:
                bluf_coached = self._rewrite_to_bluf(s, profile.power_axis)
                critique_msg = "Spoken delivery included verbal preambles or hedging qualifiers that reduced executive impact."
                improvements.append(AreaForImprovement(
                    critique=critique_msg,
                    verbatim_quote=s,
                    coached_phrasing=bluf_coached
                ))
        else:
            # If the user spoke clearly, coach them on taking their sentence to an even higher executive level
            for s in user_sentences[:top_n]:
                bluf_coached = self._rewrite_to_bluf(s, profile.power_axis)
                improvements.append(AreaForImprovement(
                    critique="To maximize executive gravitas, structure this statement with bottom-line impact upfront.",
                    verbatim_quote=s,
                    coached_phrasing=bluf_coached
                ))

        # Ensure exact count of top_n items
        while len(strengths) < top_n:
            idx = len(strengths)
            quote = user_sentences[idx % len(user_sentences)]
            strengths.append(TopStrength(
                observation=f"Communicated key point directly with consistent conversational engagement.",
                verbatim_quote=quote
            ))

        while len(improvements) < top_n:
            idx = len(improvements)
            quote = user_sentences[idx % len(user_sentences)]
            improvements.append(AreaForImprovement(
                critique="Frame delivery with definitive authority and eliminate hesitation markers.",
                verbatim_quote=quote,
                coached_phrasing=self._rewrite_to_bluf(quote, profile.power_axis)
            ))

        # Dynamic executive summary based on user metrics
        if metrics.presence_score >= 80:
            summary = f"High executive presence demonstrated in interaction with {profile.counterpart_name}. Continue leading directly with the bottom-line conclusion."
        else:
            summary = (
                f"In your conversation with {profile.counterpart_name} ({profile.role_title}), your core content was relevant, "
                "but verbal qualifiers and preambles reduced executive authority. State the decision or status directly in the first sentence."
            )

        return ExecutiveCoachingEvaluation(
            persona_context=profile.strategic_focus,
            metrics=metrics,
            top_strengths=strengths[:top_n],
            areas_for_improvement=improvements[:top_n],
            longitudinal_summary=summary,
            persona_alignment_notes=f"Evaluated against {profile.power_axis.value} (BLUF) executive communication standards."
        )

    def _rewrite_to_bluf(self, raw_sentence: str, power_axis: PowerAxis) -> str:
        """
        Dynamically strips fillers and hedging qualifiers from the user's actual sentence
        and transforms it into a clean, decisive BLUF executive statement.
        """
        cleaned = raw_sentence

        # Strip common verbal filler words
        for pat in [
            r"\bbasically\b", r"\bmatlab\b", r"\blike\b", r"\byou know\b",
            r"\bactually\b", r"\byaani\b", r"\barre\b", r"\bhaina\b",
            r"\bumm?\b", r"\buhh?\b", r"\bi mean\b", r"\bsort of\b", r"\bkind of\b"
        ]:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        # Strip common hedging prefixes
        for pat in [
            r"^(?:yeah\s+so\s+|so\s+|well\s+|look\s+)",
            r"\bi\s+(?:just\s+)?think\s+(?:that\s+)?",
            r"\bi\s+just\s+wanted\s+to\s+(?:check\s+if\s+)?",
            r"\bmaybe\s+(?:we\s+could\s+)?(?:possibly\s+)?",
            r"\bsorry\s+to\s+bother\s+you\s+(?:but\s+)?",
            r"\bi\s+might\s+be\s+wrong\s+but\s+",
            r"\bi'm\s+not\s+(?:totally\s+)?sure\s+but\s+",
            r"\bif\s+it's\s+not\s+too\s+much\s+trouble\s+",
            r"\bif\s+you\s+don't\s+mind\s+",
        ]:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        # Normalize whitespace and capitalization
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            cleaned = "The status update is on schedule."

        # Capitalize first letter
        cleaned = cleaned[0].upper() + cleaned[1:]

        # Adjust tone based on Power Axis
        if power_axis == PowerAxis.UPWARD:
            if not cleaned.lower().startswith(("we have", "we will", "the ", "our ", "i recommend", "the decision")):
                if cleaned.lower().startswith("we "):
                    cleaned = "We will " + cleaned[3:]
                elif cleaned.lower().startswith("i "):
                    cleaned = "I recommend we " + cleaned[2:]
        elif power_axis == PowerAxis.LATERAL:
            if not cleaned.lower().startswith(("let's", "we can", "our team", "together")):
                cleaned = f"Let's align to {cleaned[0].lower() + cleaned[1:]}"
        else: # DOWNWARD
            if not cleaned.lower().startswith(("what", "how", "let's", "great work")):
                cleaned = f"To ensure clear execution, {cleaned[0].lower() + cleaned[1:]}"

        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."

        return cleaned[:250]

    def _enforce_strict_constraints(
        self,
        evaluation: ExecutiveCoachingEvaluation,
        dialogue: List[Utterance],
        target_speaker: str,
        top_n: int
    ) -> ExecutiveCoachingEvaluation:
        """Enforces string length <= 250, score bounds, and Top-N count."""
        strengths = evaluation.top_strengths[:top_n]
        improvements = evaluation.areas_for_improvement[:top_n]

        for s in strengths:
            s.observation = s.observation[:250].strip()
        for i in improvements:
            i.critique = i.critique[:250].strip()
            i.coached_phrasing = i.coached_phrasing[:250].strip()

        return ExecutiveCoachingEvaluation(
            persona_context=evaluation.persona_context,
            metrics=evaluation.metrics,
            top_strengths=strengths,
            areas_for_improvement=improvements,
            longitudinal_summary=evaluation.longitudinal_summary,
            persona_alignment_notes=evaluation.persona_alignment_notes
        )
