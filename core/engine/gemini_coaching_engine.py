"""
Gemini Intelligence Coaching Synthesis Engine.
Uses Google Gemini for deep semantic coaching, register-tailored rephrasings,
and multi-dimensional communication scoring.
"""

import os
import json
import logging
import warnings
from typing import Optional, List, Dict, Any

# Suppress GenAI automatic function calling warning
logging.getLogger("google.genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

try:
    from .schema import (
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        ActionItem,
        FillerWordMetric,
        Utterance
    )
    from .persona_ontology import PersonaOntologyEngine, PowerAxis, PersonaProfile
    from .metrics_calculator import MetricsCalculator
    from .action_item_extractor import ActionItemExtractor
    from ..privacy.pii_redactor import PIIRedactor
    from ..config import get_gemini_api_key, GEMINI_MODEL
except (ImportError, ValueError):
    from engine.schema import (
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        ActionItem,
        FillerWordMetric,
        Utterance
    )
    from engine.persona_ontology import PersonaOntologyEngine, PowerAxis, PersonaProfile
    from engine.metrics_calculator import MetricsCalculator
    from engine.action_item_extractor import ActionItemExtractor
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
  "action_items": [
    {{
      "owner": "Speaker name (e.g. Rahul or USER)",
      "task": "Concrete summary of committed task or meeting follow-up",
      "due_time_or_date": "Extracted date/time anchor (e.g. 31 Aug at 10 AM, Tomorrow EOD, Friday) or null",
      "verbatim_quote": "Exact spoken sentence containing the commitment or scheduling promise",
      "category": "Follow-up Call / Meeting | Deliverable / Commitment | Review / Investigation | Assigned Request",
      "urgency": "High | Medium | Normal"
    }}
  ],
  "longitudinal_summary": "Two punchy sentences summarizing overall takeaway and core action item.",
  "persona_alignment_notes": "Evaluated against {power_axis.value} communication rubric."
}}

Guidelines:
1. Multilingual & Hinglish Fluency: The transcript may contain English, Hindi, Hinglish (code-mixed Hindi-English), or South Asian corporate idioms (e.g. 'matlab hume ye kal ship karna hai', 'mujhe lagta hai ki latency badh sakti hai', 'aap please update bhej dena', 'theek hai'). You MUST fluently comprehend Hinglish dialogue turns, identify real communication friction points, extract all commitments/action items, and provide polished executive coached phrasing with high conviction.
2. Deliver genuine strengths and genuine improvement areas without artificial padding.
3. Detect ALL commitments, scheduling promises, follow-up calls (e.g. 'I will call you on 31 aug at 10 am', 'main kal 10 baje call karunga'), deliverables, and assigned tasks into 'action_items'.
4. Every critique must include a direct 'Action:' directive.
5. Every coached_phrasing must be natural, polished, and directly rephrase what was actually said.
6. Return ONLY valid JSON without markdown wrapping.
"""

        try:
            from google.genai import types

            config_kwargs = {"response_mime_type": "application/json"}
            try:
                config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
            except Exception:
                pass

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs)
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

            action_items = [
                ActionItem(
                    owner=ai.get("owner", "USER"),
                    task=ai.get("task", ""),
                    due_time_or_date=ai.get("due_time_or_date"),
                    verbatim_quote=ai.get("verbatim_quote", ""),
                    category=ai.get("category", "Follow-up"),
                    urgency=ai.get("urgency", "Normal")
                )
                for ai in data.get("action_items", [])
            ]

            # Fallback to deterministic NLP extractor if LLM missed items
            if not action_items:
                action_items = ActionItemExtractor.extract_from_dialogue(redacted_dialogue)

            final_strengths = strengths[:top_n] if top_n and top_n > 0 else strengths
            final_improvements = improvements[:top_n] if top_n and top_n > 0 else improvements

            return ExecutiveCoachingEvaluation(
                persona_context=data.get("persona_context", profile.strategic_focus),
                metrics=metrics,
                top_strengths=final_strengths,
                areas_for_improvement=final_improvements,
                action_items=action_items,
                longitudinal_summary=data.get("longitudinal_summary", ""),
                persona_alignment_notes=data.get("persona_alignment_notes", f"Evaluated against {power_axis.value} communication rubric.")
            )

        except Exception as e:
            return None
