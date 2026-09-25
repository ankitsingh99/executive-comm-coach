"""
Executive Coaching Synthesis Engine.
Defaults to 100% on-device local execution for development and edge deployment.
"""

from typing import Optional

from .gemini_coaching_engine import GeminiCoachingSynthesizer
from .local_coaching_synthesizer import LocalCoachingSynthesizer
from .schema import ConversationSession, ExecutiveCoachingEvaluation


class ExecutiveCoachingEngine:
    """
    Primary interface for Executive Coaching evaluation.
    Leverages Gemini for SOTA contextual coaching with automated local fallback.
    """

    def __init__(self, use_local_only: bool = False):
        self.use_local_only = use_local_only
        self.local_synthesizer = LocalCoachingSynthesizer()
        self.gemini_synthesizer = GeminiCoachingSynthesizer()

    def evaluate_session(
        self, session: ConversationSession, top_n: Optional[int] = None, use_llm: bool = True
    ) -> ExecutiveCoachingEvaluation:
        """
        Executes coaching evaluation via Gemini when available, falling back to on-device NLP.
        """
        if not self.use_local_only and use_llm and self.gemini_synthesizer.is_available():
            gemini_eval = self.gemini_synthesizer.synthesize(session=session, top_n=top_n)
            if gemini_eval is not None:
                return gemini_eval

        return self.local_synthesizer.synthesize(session=session, top_n=top_n, try_local_ollama=use_llm)
