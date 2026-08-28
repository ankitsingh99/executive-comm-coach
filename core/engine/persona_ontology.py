"""
Persona Conditioning Context & Relational Ontology for Executive Communication Coaching.
Defines dynamic evaluation strategies, power axis rubrics, and relational prompt conditioning.
"""

from enum import Enum
from typing import Dict, List, Any, Optional

try:
    from .schema import BaseModel
except (ImportError, ValueError):
    from engine.schema import BaseModel


class PowerAxis(str, Enum):
    SOLO = "SOLO"          # Self-Practice / Monologue / Thought rehearsal / Speech
    CASUAL = "CASUAL"      # Informal / Social banter / Coffee chat / Friends
    LATERAL = "LATERAL"    # Peer / Cross-functional Stakeholder / PM / Tech Lead
    UPWARD = "UPWARD"      # Manager / Director / VP / CXO
    DOWNWARD = "DOWNWARD"  # Direct Report / Intern / Mentee
    CONFLICT = "CONFLICT"  # Difficult Conversation / Negotiation / Dispute Resolution


class EvaluationRubricDimension(BaseModel):
    name: str = ""
    description: str = ""
    target_behavior: str = ""
    anti_pattern: str = ""
    weight: float = 1.0

    def __init__(
        self,
        name: str = "",
        description: str = "",
        target_behavior: str = "",
        anti_pattern: str = "",
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            target_behavior=target_behavior,
            anti_pattern=anti_pattern,
            weight=weight,
            **kwargs
        )
        self.name = name
        self.description = description
        self.target_behavior = target_behavior
        self.anti_pattern = anti_pattern
        self.weight = weight


class PersonaProfile(BaseModel):
    power_axis: PowerAxis = PowerAxis.SOLO
    role_title: str = ""
    counterpart_name: str = ""
    strategic_focus: str = ""
    rubric_dimensions: List[EvaluationRubricDimension] = []
    custom_guidelines: List[str] = []

    def __init__(
        self,
        power_axis: PowerAxis = PowerAxis.SOLO,
        role_title: str = "",
        counterpart_name: str = "",
        strategic_focus: str = "",
        rubric_dimensions: Optional[List[EvaluationRubricDimension]] = None,
        custom_guidelines: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            power_axis=power_axis,
            role_title=role_title,
            counterpart_name=counterpart_name,
            strategic_focus=strategic_focus,
            rubric_dimensions=rubric_dimensions or [],
            custom_guidelines=custom_guidelines or [],
            **kwargs
        )
        self.power_axis = power_axis
        self.role_title = role_title
        self.counterpart_name = counterpart_name
        self.strategic_focus = strategic_focus
        self.rubric_dimensions = rubric_dimensions or []
        self.custom_guidelines = custom_guidelines or []


SOLO_RUBRIC = [
    EvaluationRubricDimension(
        name="Clarity of Thought & Cohesive Structure",
        description="Structured self-expression with clear main thesis and logical flow.",
        target_behavior="State the core point cleanly, followed by coherent reasoning or progression.",
        anti_pattern="Disorganized stream of consciousness, loose wandering thoughts.",
        weight=1.5
    ),
    EvaluationRubricDimension(
        name="Crisp Delivery & Minimal Filler Words",
        description="Clean, deliberate speech rhythm without excessive verbal fillers.",
        target_behavior="Use deliberate pauses instead of verbal crutches ('um', 'like', 'matlab').",
        anti_pattern="Frequent filler words, repetitive stammering, verbal crutches.",
        weight=1.4
    ),
    EvaluationRubricDimension(
        name="Pacing, Cadence & Confident Vocal Projection",
        description="Steady, measured speaking rate with natural vocal inflection and confidence.",
        target_behavior="Maintain a balanced tempo (130-160 WPM) with confident, steady tone.",
        anti_pattern="Rushing through thoughts, trailing off at sentence endings.",
        weight=1.2
    ),
    EvaluationRubricDimension(
        name="Impactful Vocabulary & Polish",
        description="Precise and expressive language choice suitable for personal practice and rehearsals.",
        target_behavior="Use clear, vivid, and precise vocabulary.",
        anti_pattern="Vague placeholders ('things and stuff', 'you know what I mean').",
        weight=1.0
    )
]

CASUAL_RUBRIC = [
    EvaluationRubricDimension(
        name="Conversational Warmth & Relatability",
        description="Approachable, friendly tone that fosters interpersonal connection and ease.",
        target_behavior="Show enthusiasm, conversational warmth, and relatable expressions.",
        anti_pattern="Excessively stiff, robotic, or overly transactional language.",
        weight=1.5
    ),
    EvaluationRubricDimension(
        name="Natural Flow & Conversational Ease",
        description="Smooth, effortless dialogue flow without awkward pauses or rigid phrasing.",
        target_behavior="Speak naturally, adapting seamlessly to the conversational vibe.",
        anti_pattern="Stilted corporate speak in an informal setting.",
        weight=1.3
    ),
    EvaluationRubricDimension(
        name="Bilingual Fluency & Code-Switching Polish",
        description="Effortless, natural multilingual/Hinglish flow without awkward crutches.",
        target_behavior="Organic code-switching and clear enunciation.",
        anti_pattern="Heavy reliance on repetitive verbal crutches ('matlab', 'basically').",
        weight=1.0
    ),
    EvaluationRubricDimension(
        name="Active Engagement & Reciprocity",
        description="Engaging the listener with reciprocal hooks and open, friendly presence.",
        target_behavior="Include conversational invitations and active relational warmth.",
        anti_pattern="One-sided monologue, cold indifference.",
        weight=1.0
    )
]

UPWARD_RUBRIC = [
    EvaluationRubricDimension(
        name="Executive Brevity & BLUF",
        description="Bottom-Line-Up-Front delivery stating core decisions, impact, or blockers first.",
        target_behavior="State the decision, recommendation, or status in the first sentence, followed by quantified metrics.",
        anti_pattern="Rambling preambles, chronological story-telling, or burying the ask/blocker.",
        weight=1.5
    ),
    EvaluationRubricDimension(
        name="Authority & Definitive Assertion",
        description="Confidence and decisiveness without self-diminishing qualifiers.",
        target_behavior="Use definitive framing ('Our data indicates', 'I recommend', 'We need to').",
        anti_pattern="Excessive hedging ('I just think', 'Maybe we could possibly', 'Sorry to bother you', 'I might be wrong but').",
        weight=1.2
    ),
    EvaluationRubricDimension(
        name="Quantified Strategic Impact",
        description="Framing discussions around ROI, business metrics, timelines, and risks.",
        target_behavior="Quantify metrics, latency, revenue, costs, and timeline implications clearly.",
        anti_pattern="Vague descriptions ('it works faster', 'we did a lot of things').",
        weight=1.0
    ),
    EvaluationRubricDimension(
        name="Diplomatic Candor & Code-Switching Elegance",
        description="Professional Hinglish elegance and respectful deference without subservience.",
        target_behavior="Clean code-switching with natural fluency; appropriate honorifics paired with clear business clarity.",
        anti_pattern="Excessive fillers ('matlab', 'basically', 'like') or apologetic subservience.",
        weight=1.0
    )
]

LATERAL_RUBRIC = [
    EvaluationRubricDimension(
        name="Collaborative Framing & Mutual Benefit",
        description="Structuring alignment around shared goals and win-win team objectives.",
        target_behavior="Highlight shared milestones, acknowledge peer dependencies, and propose collaborative solutions.",
        anti_pattern="Siloed demands, defensive posture, or passing blame.",
        weight=1.4
    ),
    EvaluationRubricDimension(
        name="Strategic Inquiry & Active Listening",
        description="Asking open questions to unblock dependencies and actively validating counterpart points.",
        target_behavior="Mirror/summarize peer perspectives before proposing changes; ask clarifying open-ended questions.",
        anti_pattern="Steamrolling over peer inputs, interrupting, or dismissing technical constraints.",
        weight=1.3
    ),
    EvaluationRubricDimension(
        name="Diplomatic Firmness",
        description="Holding ground on core priorities while remaining constructive and flexible.",
        target_behavior="Clear assertion of constraints while offering alternative pathways.",
        anti_pattern="Passive-aggressive compliance or immediate unstructured capitulation.",
        weight=1.0
    ),
    EvaluationRubricDimension(
        name="Bilingual Polish & Clarity",
        description="Crisp bilingual articulation in cross-functional syncs without confusing slang.",
        target_behavior="Precise terminology, minimal filler words, high clarity.",
        anti_pattern="Muddled explanations loaded with verbal ticks ('like, you know, matlab').",
        weight=0.8
    )
]

DOWNWARD_RUBRIC = [
    EvaluationRubricDimension(
        name="Clear Direction & Expectation Setting",
        description="Providing crisp, unambiguous context, priority definition, and next steps.",
        target_behavior="Clearly define the goal, definition of done, and key milestones.",
        anti_pattern="Ambiguous or conflicting instructions, unstructured brain-dumps.",
        weight=1.4
    ),
    EvaluationRubricDimension(
        name="Psychological Safety & Empathetic Coaching",
        description="Fostering an open environment where questions and errors are constructively addressed.",
        target_behavior="Validate effort, encourage questions, and coach through guidance rather than reprimand.",
        anti_pattern="Dismissive tone, micro-management, or shutting down questions.",
        weight=1.5
    ),
    EvaluationRubricDimension(
        name="Socratic Questioning",
        description="Guiding the mentee/report to solutions through structured open-ended prompts.",
        target_behavior="Ask questions that prompt problem-solving ('What tradeoffs do you see with approach X?').",
        anti_pattern="Dictating all answers without building autonomy.",
        weight=1.2
    ),
    EvaluationRubricDimension(
        name="Active Listening & Validation",
        description="Allowing the mentee to articulate their thoughts completely before intervening.",
        target_behavior="Acknowledge blockers, mirror difficulties, and provide affirmative guidance.",
        anti_pattern="Premature interruption, invalidation of difficulties.",
        weight=1.0
    )
]

CONFLICT_RUBRIC = [
    EvaluationRubricDimension(
        name="Emotional De-escalation & Neutral Objectivity",
        description="Maintaining calm poise and addressing facts rather than escalating emotional tension.",
        target_behavior="Use neutral, objective framing and calm, steady tone.",
        anti_pattern="Accusatory language ('you always', 'you failed to'), defensiveness, emotional escalation.",
        weight=1.5
    ),
    EvaluationRubricDimension(
        name="Empathetic Perspective-Taking",
        description="Demonstrating genuine understanding of the counterpart's constraints before offering counterpoints.",
        target_behavior="Acknowledge counterpart's position ('I understand your concern regarding X').",
        anti_pattern="Dismissing counterpart's concerns, immediate invalidation.",
        weight=1.4
    ),
    EvaluationRubricDimension(
        name="Principled Solution-Oriented Framing",
        description="Focusing on mutual interests and actionable compromises rather than fixed positions.",
        target_behavior="Propose constructive solutions and shared criteria for resolution.",
        anti_pattern="Ultimatums, rigid stonewalling, zero-sum mindset.",
        weight=1.2
    ),
    EvaluationRubricDimension(
        name="Assertive & Respectful Boundaries",
        description="Holding necessary boundaries with respect without hostility.",
        target_behavior="Clearly articulate limits and constraints calmly and firmly.",
        anti_pattern="Passive aggression, sarcasm, or appeasement that harms long-term goals.",
        weight=1.0
    )
]


class PersonaOntologyEngine:
    """Manages persona creation, power axis selection, and dynamic prompt conditioning."""

    @staticmethod
    def get_rubric_for_power_axis(power_axis: PowerAxis) -> List[EvaluationRubricDimension]:
        if power_axis == PowerAxis.SOLO:
            return SOLO_RUBRIC
        elif power_axis == PowerAxis.CASUAL:
            return CASUAL_RUBRIC
        elif power_axis == PowerAxis.UPWARD:
            return UPWARD_RUBRIC
        elif power_axis == PowerAxis.LATERAL:
            return LATERAL_RUBRIC
        elif power_axis == PowerAxis.DOWNWARD:
            return DOWNWARD_RUBRIC
        elif power_axis == PowerAxis.CONFLICT:
            return CONFLICT_RUBRIC
        return SOLO_RUBRIC

    @classmethod
    def create_persona_profile(
        cls,
        counterpart_name: str = "",
        role_title: str = "",
        power_axis: PowerAxis = PowerAxis.SOLO,
        custom_notes: Optional[List[str]] = None
    ) -> PersonaProfile:
        rubric = cls.get_rubric_for_power_axis(power_axis)
        
        if power_axis == PowerAxis.SOLO:
            strategic_focus = (
                "SOLO SELF-PRACTICE / MONOLOGUE: Speaking independently to structure thoughts or rehearse speech. "
                "Prioritize Thought Structure, Crisp Delivery, Elimination of Filler Words, and Confident Cadence."
            )
        elif power_axis == PowerAxis.CASUAL:
            target = f"with {counterpart_name} ({role_title})" if counterpart_name else "in an informal setting"
            strategic_focus = (
                f"CASUAL / SOCIAL STRATEGY: Conversing {target}. "
                "Prioritize Conversational Warmth, Natural Flow, Multilingual Polish, and Engaging Reciprocity."
            )
        elif power_axis == PowerAxis.UPWARD:
            target = f"with {counterpart_name} ({role_title})" if counterpart_name else "with Senior Leadership"
            strategic_focus = (
                f"UPWARD STRATEGY: Interacting {target}. "
                "Prioritize Executive Brevity (BLUF), Quantified Impact, and High Assertiveness. "
                "Strictly eliminate rambling intros and self-diminishing qualifiers."
            )
        elif power_axis == PowerAxis.LATERAL:
            target = f"with peer {counterpart_name} ({role_title})" if counterpart_name else "with cross-functional peers"
            strategic_focus = (
                f"LATERAL STRATEGY: Collaborating {target}. "
                "Prioritize Mutual Benefit framing, Strategic Inquiry, Active Listening, and Diplomatic Alignment."
            )
        elif power_axis == PowerAxis.DOWNWARD:
            target = f"with {counterpart_name} ({role_title})" if counterpart_name else "with direct reports / mentees"
            strategic_focus = (
                f"DOWNWARD STRATEGY: Mentoring / Leading {target}. "
                "Prioritize Psychological Safety, Clear Direction, Socratic Questioning, and Empathetic Feedback."
            )
        elif power_axis == PowerAxis.CONFLICT:
            target = f"with {counterpart_name} ({role_title})" if counterpart_name else "in a tense discussion"
            strategic_focus = (
                f"CONFLICT / NEGOTIATION STRATEGY: Navigating difficult conversation {target}. "
                "Prioritize Emotional De-escalation, Empathetic Perspective-Taking, and Principled Solution-Oriented Framing."
            )
        else:
            strategic_focus = "GENERAL COMMUNICATION: Prioritize Clarity, Active Listening, and Articulation."

        return PersonaProfile(
            power_axis=power_axis,
            role_title=role_title,
            counterpart_name=counterpart_name,
            strategic_focus=strategic_focus,
            rubric_dimensions=rubric,
            custom_guidelines=custom_notes or []
        )

    @classmethod
    def generate_system_instruction(cls, profile: PersonaProfile, top_n: int = 3) -> str:
        """Generates a specialized system prompt injected with the persona ontology."""
        dimensions_text = "\n".join([
            f"- **{d.name}** (Weight {d.weight}x): Target: {d.target_behavior} | Anti-Pattern to flag: {d.anti_pattern}"
            for d in profile.rubric_dimensions
        ])

        target_info = f"{profile.counterpart_name} ({profile.role_title})" if profile.counterpart_name else "Self-Practice / General Context"

        return f"""You are the Universal Conversational Intelligence & Communication Coach (Google Pixel Tensor AI Engine).
You evaluate speech dialogue transcripts across all communication registers (formal, informal, casual, solo practice, executive, peer, and conflict resolution) to provide high-impact, actionable coaching.

### Context & Communication Register:
- Counterpart / Audience: {target_info}
- Communication Mode / Register: {profile.power_axis.value}
- Strategic Focus: {profile.strategic_focus}

### Relational Evaluation Dimensions:
{dimensions_text}

### Deterministic Structured Output Instructions:
1. Deliver EXACTLY {top_n} Top Strengths and EXACTLY {top_n} Areas for Improvement.
2. Every `verbatim_quote` MUST be an exact verbatim substring from the USER's turns in the transcript.
3. Every `observation`, `critique`, and `coached_phrasing` MUST be concise (maximum 250 characters).
4. For every Area for Improvement, provide a high-impact, polished `coached_phrasing` calibrated specifically for the {profile.power_axis.value} mode.
5. Accurately score metrics [0, 100]:
   - `presence_score`: Overall structure, clarity, and register-appropriate delivery
   - `assertiveness_score`: Confidence and clarity vs self-diminishing qualifiers
   - `active_listening_score`: Conversational reciprocity, mirroring, and engagement
   - `filler_words_detected`: Itemized counts of verbal fillers ('matlab', 'basically', 'like', 'you know', 'um', etc.).
"""
