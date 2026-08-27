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
    UPWARD = "UPWARD"      # Manager, Director, VP, CXO
    LATERAL = "LATERAL"    # Peer, Cross-functional Stakeholder, PM, Tech Lead
    DOWNWARD = "DOWNWARD"  # Direct Report, Intern, Mentee


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
    power_axis: PowerAxis = PowerAxis.LATERAL
    role_title: str = ""
    counterpart_name: str = ""
    strategic_focus: str = ""
    rubric_dimensions: List[EvaluationRubricDimension] = []
    custom_guidelines: List[str] = []

    def __init__(
        self,
        power_axis: PowerAxis = PowerAxis.LATERAL,
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


# Detailed Relational Strategy Matrix based on the Executive Framework
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
        target_behavior="Clean code-switching with natural fluency; appropriate honorifics (Aap) paired with clear business clarity.",
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


class PersonaOntologyEngine:
    """Manages persona creation, power axis selection, and dynamic prompt conditioning."""

    @staticmethod
    def get_rubric_for_power_axis(power_axis: PowerAxis) -> List[EvaluationRubricDimension]:
        if power_axis == PowerAxis.UPWARD:
            return UPWARD_RUBRIC
        elif power_axis == PowerAxis.LATERAL:
            return LATERAL_RUBRIC
        elif power_axis == PowerAxis.DOWNWARD:
            return DOWNWARD_RUBRIC
        return LATERAL_RUBRIC

    @classmethod
    def create_persona_profile(
        cls,
        counterpart_name: str,
        role_title: str,
        power_axis: PowerAxis,
        custom_notes: Optional[List[str]] = None
    ) -> PersonaProfile:
        rubric = cls.get_rubric_for_power_axis(power_axis)
        
        if power_axis == PowerAxis.UPWARD:
            strategic_focus = (
                f"UPWARD STRATEGY: Interacting with {counterpart_name} ({role_title}). "
                "Prioritize Executive Brevity (BLUF), Quantified Impact, and High Assertiveness. "
                "Strictly eliminate rambling intros and self-diminishing qualifiers."
            )
        elif power_axis == PowerAxis.LATERAL:
            strategic_focus = (
                f"LATERAL STRATEGY: Collaborating with peer {counterpart_name} ({role_title}). "
                "Prioritize Mutual Benefit framing, Strategic Inquiry, Active Listening, and Diplomatic Alignment."
            )
        else:
            strategic_focus = (
                f"DOWNWARD STRATEGY: Mentoring / Leading {counterpart_name} ({role_title}). "
                "Prioritize Psychological Safety, Clear Direction, Socratic Questioning, and Empathetic Feedback."
            )

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

        return f"""You are the Executive Conversational Intelligence & Communication Coach (Google Pixel Tensor AI Engine).
You evaluate workplace dialogue transcripts to provide high-impact, actionable executive coaching.

### Counterpart & Power Axis Context:
- Counterpart: {profile.counterpart_name} ({profile.role_title})
- Power Axis: {profile.power_axis.value}
- Strategic Focus: {profile.strategic_focus}

### Relational Evaluation Dimensions:
{dimensions_text}

### Deterministic Structured Output Instructions:
1. Deliver EXACTLY {top_n} Top Strengths and EXACTLY {top_n} Areas for Improvement.
2. Every `verbatim_quote` MUST be an exact verbatim substring from the USER's turns in the transcript.
3. Every `observation`, `critique`, and `coached_phrasing` MUST be concise (maximum 250 characters).
4. For every Area for Improvement, provide a high-impact, polished `coached_phrasing` that demonstrates executive presence, BLUF, or assertive code-switching.
5. Accurately score metrics [0, 100]:
   - `presence_score`: Overall executive structure & BLUF delivery
   - `assertiveness_score`: Assertions vs self-diminishing qualifiers ('I just think', 'Maybe we could')
   - `active_listening_score`: Acknowledgments, mirroring, inquiry before advancing agenda
   - `filler_words_detected`: Itemized counts of verbal fillers ('matlab', 'basically', 'like', 'you know', etc.).
"""
