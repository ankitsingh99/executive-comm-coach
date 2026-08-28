"""
Gemini Intelligence Coaching Synthesis Engine.
Uses Google Gemini for deep semantic coaching, register-tailored rephrasings,
and multi-dimensional communication scoring.
"""

import os
import json
from typing import Optional, List, Dict, Any

try:
    from .schema import (
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        FillerWordMetric,
        Utterance
    )
    from .persona_ontology import PersonaOntologyEngine, PowerAxis, PersonaProfile
    from .metrics_calculator import MetricsCalculator
    from ..privacy.pii_redactor import PIIRedactor
    from ..config import get_gemini_api_key, GEMINI_MODEL
except (ImportError, ValueError):
    from engine.schema import (
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        FillerWordMetric,
        Utterance
    )
    from engine.persona_ontology import PersonaOntologyEngine, PowerAxis, PersonaProfile
    from engine.metrics_calculator import MetricsCalculator
    from privacy.pii_redactor import PIIRedactor
    from config import get_gemini_api_key, GEMINI_MODEL


class GeminiCoachingSynthesizer:
    """
    Coaching synthesizer powered by Google Gemini.
    Generates tailored, highly actionable coaching across all communication modes.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and self._get_client() is not None)

    def synthesize(
        self,
        session: ConversationSession,
        top_n: Optional[int] = None
    ) -> Optional[ExecutiveCoachingEvaluation]:
        """
        Synthesizes structured coaching evaluation using Gemini.
        """
        if not self.is_available():
            return None

        client = self._get_client()
        if not client:
            return None

        redacted_dialogue: List[Utterance] = []
        for u in session.dialogue:
            red_text, _ = PIIRedactor.redact_text(u.transcript)
            redacted_dialogue.append(
                Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text)
            )

        try:
            power_axis = PowerAxis(session.power_axis.upper())
        except ValueError:
            power_axis = PowerAxis.SOLO

        profile = PersonaOntologyEngine.create_persona_profile(
            counterpart_name=session.counterpart_name or "",
            role_title=session.counterpart_role or "",
            power_axis=power_axis
        )

        metrics = MetricsCalculator.analyze_dialogue(
            redacted_dialogue,
            target_speaker=session.target_speaker
        )

        system_instruction = PersonaOntologyEngine.generate_system_instruction(profile)
        dialogue_text = "\n".join([f"{u.speaker}: {u.transcript}" for u in redacted_dialogue])

        prompt = f"""{system_instruction}

Analyze the following transcribed dialogue:
TRANSCRIPT:
{dialogue_text}

Detected Fillers: {', '.join([f'{f.token} ({f.count}x)' for f in metrics.filler_words_detected]) or 'None'}

Return pure JSON matching this exact structure:
{{
  "persona_context": "{profile.strategic_focus}",
  "metrics": {{
    "presence_score": {metrics.presence_score},
    "assertiveness_score": {metrics.assertiveness_score},
    "active_listening_score": {metrics.active_listening_score},
    "filler_words_detected": [
      {', '.join([f'{{"token": "{f.token}", "count": {f.count}}}' for f in metrics.filler_words_detected])}
    ]
  }},
  "top_strengths": [
    {{
      "observation": "Concise delivery strength (maximum 200 chars)",
      "verbatim_quote": "Exact quote from transcript"
    }}
  ],
  "areas_for_improvement": [
    {{
      "critique": "[Friction point observed]. Action: [Concrete prescriptive rule] (maximum 220 chars)",
      "verbatim_quote": "Exact quote from transcript",
      "coached_phrasing": "Natural, high-impact rephrasing tailored to {power_axis.value} mode"
    }}
  ],
  "longitudinal_summary": "Two punchy sentences summarizing overall takeaway and core action item.",
  "persona_alignment_notes": "Evaluated against {power_axis.value} communication rubric."
}}

Guidelines:
1. Deliver genuine strengths and genuine improvement areas without artificial padding.
2. Every critique must include a direct 'Action:' directive.
3. Every coached_phrasing must be natural, polished, and directly rephrase what was actually said.
4. Return ONLY valid JSON without markdown wrapping.
"""

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw_text)

            strengths = [
                TopStrength(
                    observation=s.get("observation", "")[:250],
                    verbatim_quote=s.get("verbatim_quote", "")
                )
                for s in data.get("top_strengths", [])
            ]

            improvements = [
                AreaForImprovement(
                    critique=a.get("critique", "")[:250],
                    verbatim_quote=a.get("verbatim_quote", ""),
                    coached_phrasing=a.get("coached_phrasing", "")[:250]
                )
                for a in data.get("areas_for_improvement", [])
            ]

            final_strengths = strengths[:top_n] if top_n and top_n > 0 else strengths
            final_improvements = improvements[:top_n] if top_n and top_n > 0 else improvements

            return ExecutiveCoachingEvaluation(
                persona_context=data.get("persona_context", profile.strategic_focus),
                metrics=metrics,
                top_strengths=final_strengths,
                areas_for_improvement=final_improvements,
                longitudinal_summary=data.get("longitudinal_summary", ""),
                persona_alignment_notes=data.get("persona_alignment_notes", f"Evaluated against {power_axis.value} communication rubric.")
            )

        except Exception as e:
            return None
