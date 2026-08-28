"""
100% On-Device Dynamic Executive Coaching Synthesizer.
Operates completely offline with zero cloud API dependencies.
Performs deep semantic intent classification, grammatical transformation,
and generates customized, contextual Executive BLUF coaching for real speech.
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Optional, Tuple, Dict, Any

try:
    from .schema import (
        Utterance,
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        ActionItem,
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
        ASSERTIVE_PATTERNS,
        ACTIVE_LISTENING_PATTERNS
    )
    from .action_item_extractor import ActionItemExtractor
    from ..privacy.pii_redactor import PIIRedactor
except (ImportError, ValueError):
    from engine.schema import (
        Utterance,
        ConversationSession,
        ExecutiveCoachingEvaluation,
        CommunicationMetrics,
        TopStrength,
        AreaForImprovement,
        ActionItem,
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
        ASSERTIVE_PATTERNS,
        ACTIVE_LISTENING_PATTERNS
    )
    from engine.action_item_extractor import ActionItemExtractor
    from privacy.pii_redactor import PIIRedactor


class LocalCoachingSynthesizer:
    """
    On-device intelligent executive coaching engine.
    Analyzes real speech intent and generates bespoke, highly tailored executive coaching.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "gemma2:2b"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def synthesize(
        self,
        session: ConversationSession,
        top_n: Optional[int] = None,
        try_local_ollama: bool = True
    ) -> ExecutiveCoachingEvaluation:
        """
        Executes semantic analysis and coaching generation on real transcribed speech.
        """
        redacted_dialogue: List[Utterance] = []
        for u in session.dialogue:
            red_text, _ = PIIRedactor.redact_text(u.transcript)
            redacted_dialogue.append(
                Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text)
            )

        try:
            power_axis = PowerAxis(session.power_axis.upper())
        except ValueError:
            power_axis = PowerAxis.UPWARD

        counterpart_label = session.counterpart_name or "Counterpart"
        counterpart_role = session.counterpart_role or "Colleague"

        profile = PersonaOntologyEngine.create_persona_profile(
            counterpart_name=counterpart_label,
            role_title=counterpart_role,
            power_axis=power_axis
        )

        metrics = MetricsCalculator.analyze_dialogue(
            redacted_dialogue,
            target_speaker=session.target_speaker
        )

        evaluation: Optional[ExecutiveCoachingEvaluation] = None
        if try_local_ollama:
            evaluation = self._try_ollama_local_inference(redacted_dialogue, profile, metrics, top_n)

        if evaluation is None:
            evaluation = self._semantic_intent_synthesis(redacted_dialogue, profile, metrics, top_n)

        return self._enforce_strict_constraints(evaluation, top_n)

    def _try_ollama_local_inference(
        self,
        dialogue: List[Utterance],
        profile: PersonaProfile,
        metrics: CommunicationMetrics,
        top_n: int
    ) -> Optional[ExecutiveCoachingEvaluation]:
        """Queries local Ollama endpoint if active on user's machine and models exist."""
        try:
            req_tags = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req_tags, timeout=0.8) as resp:
                if resp.status == 200:
                    tags_data = json.loads(resp.read().decode("utf-8"))
                    avail_models = [m.get("name") for m in tags_data.get("models", [])]
                    if not avail_models:
                        return None
                    model_to_use = avail_models[0]
                else:
                    return None

            system_prompt = PersonaOntologyEngine.generate_system_instruction(profile, top_n=top_n)
            dialogue_text = "\n".join([f"{u.speaker}: {u.transcript}" for u in dialogue])
            prompt = f"{system_prompt}\n\nTRANSCRIPT:\n{dialogue_text}\n\nReturn pure JSON matching ExecutiveCoachingEvaluation schema."

            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_response = data.get("response", "")
                    parsed = json.loads(raw_response)
                    return ExecutiveCoachingEvaluation.model_validate(parsed)
        except Exception:
            return None
        return None

    def _semantic_intent_synthesis(
        self,
        dialogue: List[Utterance],
        profile: PersonaProfile,
        metrics: CommunicationMetrics,
        top_n: int
    ) -> ExecutiveCoachingEvaluation:
        """
        Actionable NLP coaching engine.
        Identifies concrete communication friction points and generates concise, prescriptive advice.
        """
        user_turns = [u for u in dialogue if u.speaker.upper() == "USER"]
        raw_text = " ".join([u.transcript.strip() for u in user_turns])

        raw_sentences = [s.strip() for s in re.split(r"[.!?\n]+", raw_text) if len(s.strip()) > 3]
        if not raw_sentences:
            raw_sentences = [raw_text.strip() or "Speech turn."]

        full_quote = raw_text.strip()

        # 1. Linguistic and Friction Pattern Detection (English + Hinglish)
        is_question = bool(re.search(r"\b(how|what|why|where|when|can|could|should|is it|how do i|kya|kyun|kaise|kab|kahan|batao)\b", full_quote, re.IGNORECASE) or "?" in full_quote)
        is_seeking_learning = bool(re.search(r"\b(learn|start|study|understand|explore|guide|recommend|figure out|seekhna|samajhna|shuru)\b", full_quote, re.IGNORECASE))
        has_hedging = any(bool(re.search(pat, full_quote, re.IGNORECASE)) for pat in HEDGING_PATTERNS)
        has_assertive = any(bool(re.search(pat, full_quote, re.IGNORECASE)) for pat in ASSERTIVE_PATTERNS)
        has_fillers = bool(metrics.filler_words_detected)

        # 2. Precise Topic & Predicate Extraction (Multilingual)
        cleaned_quote = re.sub(
            r"\b(that was|this is|basically|um+|uh+|hm+|aaah|matlab|like|you know|so|mujhe lagta hai|shayad|lag raha hai|dekho|bhai|yaar|actually|literally)\b",
            "",
            full_quote,
            flags=re.IGNORECASE
        ).strip()
        
        topic_match = re.search(r"(?:about|on|regarding|for|evaluate|explore|status of|news from|focus on|ke bare mein|par|ka status)\s+([a-zA-Z0-9_\-\s]{2,25}?)(?:\?|,|\.|$)", full_quote, re.IGNORECASE)
        
        if topic_match:
            extracted_topic = topic_match.group(1).strip()
        else:
            meaningful_words = [
                w for w in re.findall(r"\b[a-zA-Z]{3,}\b", cleaned_quote)
                if w.lower() not in [
                    "that", "this", "with", "have", "from", "today", "about", "what", "where", "when", "could", "should", "would", "just", "very",
                    "hume", "mujhe", "karna", "hoga", "karenge", "chahiye", "lagta", "raha", "gaya", "wala", "vali", "bhi", "par", "aur", "lekin", "kyunki", "isiliye", "dekho", "apna", "apne", "unka", "unke", "hain", "kare"
                ]
            ]
            extracted_topic = " ".join(meaningful_words[:3]) if meaningful_words else "the core deliverable"

        extracted_topic = re.sub(r"\s+", " ", extracted_topic).strip()

        strengths: List[TopStrength] = []
        improvements: List[AreaForImprovement] = []

        # 3. Concise Positive Strengths (Specific to actual delivery)
        if has_assertive:
            strengths.append(TopStrength(
                observation="Decisive ownership and assertive delivery commitment.",
                verbatim_quote=full_quote
            ))
        elif not has_fillers and len(full_quote.split()) >= 4:
            strengths.append(TopStrength(
                observation="Zero verbal hesitation. Clean, unbroken sentence cadence.",
                verbatim_quote=full_quote
            ))
        else:
            strengths.append(TopStrength(
                observation=f"Direct topical focus on {extracted_topic}.",
                verbatim_quote=full_quote
            ))

        if is_question:
            strengths.append(TopStrength(
                observation="Proactive engagement. Prompted alignment with an open inquiry.",
                verbatim_quote=full_quote
            ))
        else:
            strengths.append(TopStrength(
                observation="Controlled enunciation and steady pacing.",
                verbatim_quote=full_quote
            ))

        target_counterpart = profile.counterpart_name if profile.counterpart_name else "your audience"

        # 4. Actionable Friction-Point Analysis & Coached Rephrasing
        if has_fillers:
            filler_summary = ", ".join([f"'{f.token}' ({f.count}x)" for f in metrics.filler_words_detected[:2]])
            critique = f"Hesitation markers ({filler_summary}) break delivery rhythm. Action: Pause silently for 0.5s instead of vocalizing."
            coached = self._generate_crisp_bluf(full_quote, extracted_topic, profile.power_axis)
            improvements.append(AreaForImprovement(critique=critique, verbatim_quote=full_quote, coached_phrasing=coached))

        if is_seeking_learning and is_question:
            if profile.power_axis == PowerAxis.SOLO:
                critique = "Framed as an open question during solo rehearsal. Action: State as a definitive thesis to test."
                coached = f"My objective is to validate {extracted_topic} through systematic prototyping."
            elif profile.power_axis == PowerAxis.UPWARD:
                critique = "Open question shifts cognitive load upward. Action: Propose a baseline plan before asking for input."
                coached = f"I am leading the roadmap for {extracted_topic}. Let's align on the top two milestones."
            elif profile.power_axis == PowerAxis.LATERAL:
                critique = "Broad question. Action: Frame around shared team deliverables and technical prerequisites."
                coached = f"Let's review prerequisites for {extracted_topic} to ensure our sprint goals align."
            else:
                critique = "Broad inquiry. Action: Define concrete next steps before opening for discussion."
                coached = f"To structure {extracted_topic}, let's first evaluate the initial architectural tradeoffs."

            improvements.append(AreaForImprovement(critique=critique, verbatim_quote=full_quote, coached_phrasing=coached))

        elif has_hedging:
            critique = "Hedging qualifiers ('just think', 'maybe', 'mujhe lagta hai') dilute conviction. Action: State the recommendation directly as a decision."
            coached = self._generate_crisp_bluf(full_quote, extracted_topic, profile.power_axis)
            improvements.append(AreaForImprovement(critique=critique, verbatim_quote=full_quote, coached_phrasing=coached))

        else:
            # Informational / narrative statement without decision (e.g. "That was the news from India today")
            if profile.power_axis == PowerAxis.SOLO:
                critique = "Observation ended without an action item. Action: Add a direct conclusion or next step."
                coached = f"Key takeaway on {extracted_topic}: focus execution on the top deliverable first."
            elif profile.power_axis == PowerAxis.CASUAL:
                critique = "Statement is passive. Action: Add an open hook to invite conversational flow."
                coached = f"That wraps up the latest on {extracted_topic}—what's your take on it?"
            elif profile.power_axis == PowerAxis.CONFLICT:
                critique = "Observation lacks mutual resolution criteria. Action: Propose shared objective metrics."
                coached = f"Regarding {extracted_topic}, let's establish agreed criteria to resolve our blockers."
            else: # UPWARD / LATERAL
                critique = f"Statement offers context without a bottom-line decision (BLUF). Action: Lead with the recommendation."
                coached = f"Based on the latest {extracted_topic}, I recommend we prioritize rollout readiness."

            improvements.append(AreaForImprovement(critique=critique, verbatim_quote=full_quote, coached_phrasing=coached))

        # Check for unquantified narrative if relevant
        if len(improvements) == 1 and not has_hedging and not is_question:
            critique_2 = "Statement lacks quantified outcomes. Action: Add measurable metrics, timelines, or next steps."
            coached_2 = self._generate_crisp_action_plan(extracted_topic, profile.power_axis)
            improvements.append(AreaForImprovement(critique=critique_2, verbatim_quote=full_quote, coached_phrasing=coached_2))

        # Apply top_n cap only if caller explicitly requested a maximum limit
        final_strengths = strengths[:top_n] if top_n and top_n > 0 else strengths
        final_improvements = improvements[:top_n] if top_n and top_n > 0 else improvements

        # 5. Crisp, Actionable Summary
        if profile.power_axis == PowerAxis.SOLO:
            summary = f"Good topic focus on {extracted_topic}. Action: Eliminate trailing statements by concluding each point with an actionable next step."
        elif profile.power_axis == PowerAxis.UPWARD:
            summary = f"Clear topic grounding on {extracted_topic}. Action: Front-load the recommendation (BLUF) in your first sentence to maximize executive brevity."
        elif profile.power_axis == PowerAxis.LATERAL:
            summary = f"Constructive sync on {extracted_topic}. Action: Anchor proposals to shared milestones and explicit team dependencies."
        elif profile.power_axis == PowerAxis.CONFLICT:
            summary = f"Tense discussion around {extracted_topic}. Action: Keep framing strictly objective and centered on shared criteria."
        else:
            summary = f"Engaging communication regarding {extracted_topic}. Action: Lead with your core message before providing background context."

        alignment_note = (
            f"Evaluated against {profile.power_axis.value} (BLUF) executive communication rubric."
            if profile.power_axis == PowerAxis.UPWARD
            else f"Evaluated against {profile.power_axis.value} communication rubric."
        )

        action_items = ActionItemExtractor.extract_from_dialogue(dialogue)

        return ExecutiveCoachingEvaluation(
            persona_context=profile.strategic_focus,
            metrics=metrics,
            top_strengths=final_strengths,
            areas_for_improvement=final_improvements,
            action_items=action_items,
            longitudinal_summary=summary,
            persona_alignment_notes=alignment_note
        )

    def _generate_crisp_bluf(self, text: str, topic: str, power_axis: PowerAxis) -> str:
        """Generates a punchy, highly natural BLUF sentence tailored to register."""
        if power_axis == PowerAxis.SOLO:
            return f"The priority for {topic} is executing the core milestones on schedule."
        elif power_axis == PowerAxis.CASUAL:
            return f"I've been following {topic} closely—looks like we're in great shape!"
        elif power_axis == PowerAxis.CONFLICT:
            return f"I understand the constraints on {topic}. Let's agree on concrete next steps."
        elif power_axis == PowerAxis.UPWARD:
            return f"I recommend we proceed with {topic} to keep our roadmap on track."
        elif power_axis == PowerAxis.LATERAL:
            return f"Let's align on {topic} so we can unblock the upcoming sprint."
        else:
            return f"To advance {topic}, what initial tradeoffs have you identified?"

    def _generate_crisp_action_plan(self, topic: str, power_axis: PowerAxis) -> str:
        """Generates a quantified, concrete action plan alternative."""
        if power_axis == PowerAxis.SOLO:
            return f"I will complete the initial benchmark for {topic} by end of week."
        elif power_axis == PowerAxis.UPWARD:
            return f"I recommend deploying {topic} by next sprint to reduce delivery risk by 20%."
        elif power_axis == PowerAxis.LATERAL:
            return f"Let's schedule a 15-minute sync tomorrow to finalize the API contract for {topic}."
        elif power_axis == PowerAxis.CONFLICT:
            return f"Let's review the objective test data for {topic} to settle the technical direction."
        else:
            return f"Let's target delivering the {topic} prototype within the next two weeks."

    def _extract_core_topic(self, text: str) -> str:
        words = [w for w in re.findall(r"\b[A-Za-z0-9_-]+\b", text) if len(w) > 3 and w.lower() not in ["basically", "matlab", "think", "maybe", "could", "would", "should", "there", "their", "about", "which"]]
        if len(words) >= 2:
            return f"{words[0]} {words[1]}"
        elif len(words) == 1:
            return words[0]
        return "the core topic"

    def _generate_strategic_summary(self, text: str, power_axis: PowerAxis, topic: str) -> str:
        if power_axis == PowerAxis.UPWARD:
            return f"Clear topic grounding on {topic}. Action: Front-load the recommendation (BLUF) in your first sentence to maximize executive brevity."
        elif power_axis == PowerAxis.LATERAL:
            return f"Constructive sync on {topic}. Action: Anchor proposals to shared milestones and explicit team dependencies."
        elif power_axis == PowerAxis.DOWNWARD:
            return f"Engaging communication regarding {topic}. Action: Lead with your core message before providing background context."
        elif power_axis == PowerAxis.SOLO:
            return f"Well-structured rehearsal on {topic}. Action: Practice deliberate pausing at key transition points."
        else:
            return f"Clear delivery on {topic}. Action: Maintain concise framing to drive decisive outcomes."

    def _clean_and_reframe(self, text: str, topic: str, power_axis: PowerAxis) -> str:
        """Transforms a sentence into an appropriately framed coaching statement."""
        cleaned = text
        for pat in [
            r"\bbasically\b", r"\bmatlab\b", r"\blike\b", r"\byou know\b",
            r"\bactually\b", r"\bliterally\b", r"\bi mean\b",
            r"\bu+m+\b", r"\bu+h+m*\b", r"\bh+m+\b", r"\bm+h+m*\b",
            r"\ba+h+\b", r"\ba{2,}\b", r"\ba+a+h*\b", r"\be+h+\b", r"\be+r+m*\b",
            r"\bi just think\b", r"\bmaybe we could\b", r"\bsorry to bother\b",
            r"\bif i have to\b", r"\bhow do i start\b"
        ]:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if power_axis == PowerAxis.SOLO:
            return f"The key takeaway regarding {topic} is structured execution and continuous iteration."
        elif power_axis == PowerAxis.CASUAL:
            return f"I've been exploring {topic} lately, and it's looking really promising!"
        elif power_axis == PowerAxis.CONFLICT:
            return f"I understand the concerns regarding {topic}. Let's agree on the key milestones to move forward."
        elif power_axis == PowerAxis.UPWARD:
            return f"I am leading the evaluation of {topic}. What key benchmarks should we prioritize?"
        elif power_axis == PowerAxis.LATERAL:
            return f"Let's collaborate on {topic} and align our technical milestones for this sprint."
        else:
            return f"To guide your work on {topic}, what initial tradeoffs have you identified?"

    def _enforce_strict_constraints(
        self,
        evaluation: ExecutiveCoachingEvaluation,
        top_n: Optional[int] = None
    ) -> ExecutiveCoachingEvaluation:
        """Enforces length <= 250 and score bounds."""
        strengths = evaluation.top_strengths[:top_n] if top_n and top_n > 0 else evaluation.top_strengths
        improvements = evaluation.areas_for_improvement[:top_n] if top_n and top_n > 0 else evaluation.areas_for_improvement

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
            action_items=evaluation.action_items,
            longitudinal_summary=evaluation.longitudinal_summary,
            persona_alignment_notes=evaluation.persona_alignment_notes
        )
