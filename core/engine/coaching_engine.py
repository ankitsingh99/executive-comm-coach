"""
Executive Coaching Synthesis Engine.
Defaults to 100% on-device local execution for development and edge deployment.
"""

import os
from typing import Optional
from .schema import ConversationSession, ExecutiveCoachingEvaluation
from .local_coaching_synthesizer import LocalCoachingSynthesizer


class ExecutiveCoachingEngine:
    """
    Primary interface for Executive Coaching evaluation.
    Operates locally on-device by default with zero cloud dependencies.
    """

    def __init__(self, use_local_only: bool = True):
        self.use_local_only = use_local_only
        self.local_synthesizer = LocalCoachingSynthesizer()

    def evaluate_session(
        self,
        session: ConversationSession,
        top_n: int = 3,
        use_llm: bool = False
    ) -> ExecutiveCoachingEvaluation:
        """
        Executes on-device coaching evaluation.
        """
        return self.local_synthesizer.synthesize(
            session=session,
            top_n=top_n,
            try_local_ollama=use_llm
        )
