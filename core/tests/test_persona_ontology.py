"""
Automated unit tests for Persona Ontology and Relational Rubric calibration.
"""

from engine.persona_ontology import (
    PowerAxis,
    PersonaOntologyEngine,
    UPWARD_RUBRIC,
    LATERAL_RUBRIC,
    DOWNWARD_RUBRIC
)


def test_persona_profiles_creation():
    upward_profile = PersonaOntologyEngine.create_persona_profile(
        counterpart_name="Engineering Director",
        role_title="Senior Director",
        power_axis=PowerAxis.UPWARD
    )
    assert upward_profile.power_axis == PowerAxis.UPWARD
    assert "Executive Brevity & BLUF" in [d.name for d in upward_profile.rubric_dimensions]
    assert "UPWARD STRATEGY" in upward_profile.strategic_focus

    lateral_profile = PersonaOntologyEngine.create_persona_profile(
        counterpart_name="Product Manager",
        role_title="Staff PM",
        power_axis=PowerAxis.LATERAL
    )
    assert lateral_profile.power_axis == PowerAxis.LATERAL
    assert "Collaborative Framing & Mutual Benefit" in [d.name for d in lateral_profile.rubric_dimensions]

    downward_profile = PersonaOntologyEngine.create_persona_profile(
        counterpart_name="Software Engineer",
        role_title="Associate Engineer",
        power_axis=PowerAxis.DOWNWARD
    )
    assert downward_profile.power_axis == PowerAxis.DOWNWARD
    assert "Psychological Safety & Empathetic Coaching" in [d.name for d in downward_profile.rubric_dimensions]


def test_system_instruction_generation():
    profile = PersonaOntologyEngine.create_persona_profile(
        counterpart_name="VP of Engineering",
        role_title="VP of Engineering",
        power_axis=PowerAxis.UPWARD
    )
    prompt = PersonaOntologyEngine.generate_system_instruction(profile, top_n=3)
    assert "VP of Engineering" in prompt
    assert "EXACTLY 3 Top Strengths" in prompt
    assert "presence_score" in prompt
