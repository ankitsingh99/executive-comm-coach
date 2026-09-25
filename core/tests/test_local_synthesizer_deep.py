"""
Deep test suite for LocalCoachingSynthesizer covering all power axes,
NLP heuristics, Hinglish synthesis, BLUF generation, and Ollama fallbacks.
"""

import json
from unittest.mock import patch, MagicMock
from engine.schema import (
    Utterance,
    ConversationSession,
    CommunicationMetrics,
    FillerWordMetric,
)
from engine.persona_ontology import PowerAxis, PersonaProfile
from engine.local_coaching_synthesizer import LocalCoachingSynthesizer


def test_local_synthesizer_all_power_axes_english():
    """Verify coaching synthesis across all 6 power axes in English."""
    synthesizer = LocalCoachingSynthesizer()

    axes = [
        PowerAxis.SOLO,
        PowerAxis.UPWARD,
        PowerAxis.LATERAL,
        PowerAxis.CONFLICT,
        PowerAxis.CASUAL,
        PowerAxis.DOWNWARD,
    ]

    for axis in axes:
        session = ConversationSession(
            session_id=f"test_{axis.value}",
            power_axis=axis.value,
            target_speaker="USER",
            dialogue=[
                Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="I basically think we should deploy the feature tomorrow.")
            ]
        )
        eval_res = synthesizer.synthesize(session, try_local_ollama=False)
        assert eval_res is not None
        assert eval_res.top_strengths
        assert eval_res.areas_for_improvement
        assert len(eval_res.longitudinal_summary) > 5
        assert axis.value in eval_res.persona_alignment_notes or "BLUF" in eval_res.persona_alignment_notes


def test_local_synthesizer_all_power_axes_hinglish():
    """Verify coaching synthesis across all 6 power axes in Hinglish."""
    synthesizer = LocalCoachingSynthesizer()

    axes = [
        PowerAxis.SOLO,
        PowerAxis.UPWARD,
        PowerAxis.LATERAL,
        PowerAxis.CONFLICT,
        PowerAxis.CASUAL,
        PowerAxis.DOWNWARD,
    ]

    for axis in axes:
        session = ConversationSession(
            session_id=f"test_hinglish_{axis.value}",
            power_axis=axis.value,
            target_speaker="USER",
            dialogue=[
                Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="Dekho basically mujhe lagta hai hume caching deploy karni chahiye.")
            ]
        )
        eval_res = synthesizer.synthesize(session, try_local_ollama=False)
        assert eval_res is not None
        assert len(eval_res.top_strengths) >= 1
        assert len(eval_res.areas_for_improvement) >= 1
        # Check that coached phrasing has natural Hinglish/English content
        coached_text = eval_res.areas_for_improvement[0].coached_phrasing
        assert len(coached_text) > 5


def test_local_synthesizer_inquiry_and_learning_modes():
    """Test question-based statements and mentorship learning queries."""
    synthesizer = LocalCoachingSynthesizer()

    # Question inquiry
    session_q = ConversationSession(
        session_id="test_q",
        power_axis="UPWARD",
        target_speaker="USER",
        dialogue=[
            Utterance(speaker="USER", start_time=0.0, end_time=3.5, transcript="How do I start the benchmark evaluation?")
        ]
    )
    eval_q = synthesizer.synthesize(session_q, try_local_ollama=False)
    assert eval_q is not None
    assert len(eval_q.areas_for_improvement) >= 1

    # Learning mode
    session_learn = ConversationSession(
        session_id="test_learn",
        power_axis="DOWNWARD",
        target_speaker="USER",
        dialogue=[
            Utterance(speaker="USER", start_time=0.0, end_time=3.5, transcript="I want to understand and explore the system architecture.")
        ]
    )
    eval_learn = synthesizer.synthesize(session_learn, try_local_ollama=False)
    assert eval_learn is not None
    assert len(eval_learn.top_strengths) >= 1


def test_local_synthesizer_helpers_coverage():
    """Test helper functions directly for complete branch coverage."""
    synthesizer = LocalCoachingSynthesizer()

    # _generate_crisp_bluf
    for axis in [PowerAxis.SOLO, PowerAxis.CASUAL, PowerAxis.CONFLICT, PowerAxis.UPWARD, PowerAxis.LATERAL, PowerAxis.DOWNWARD]:
        bluf_en = synthesizer._generate_crisp_bluf("I think caching is good", "caching latency", axis, is_hinglish=False)
        assert len(bluf_en) > 5
        bluf_hi = synthesizer._generate_crisp_bluf("caching achha hai", "caching latency", axis, is_hinglish=True)
        assert len(bluf_hi) > 5

    # _generate_crisp_action_plan
    for axis in [PowerAxis.SOLO, PowerAxis.CASUAL, PowerAxis.CONFLICT, PowerAxis.UPWARD, PowerAxis.LATERAL, PowerAxis.DOWNWARD]:
        act_en = synthesizer._generate_crisp_action_plan("caching benchmark", axis, is_hinglish=False)
        assert len(act_en) > 5
        act_hi = synthesizer._generate_crisp_action_plan("caching benchmark", axis, is_hinglish=True)
        assert len(act_hi) > 5

    # _generate_strategic_summary
    for axis in [PowerAxis.SOLO, PowerAxis.UPWARD, PowerAxis.LATERAL, PowerAxis.DOWNWARD, PowerAxis.CONFLICT]:
        strat = synthesizer._generate_strategic_summary("some text", axis, "the core topic")
        assert len(strat) > 5

    # _clean_and_reframe
    for axis in [PowerAxis.SOLO, PowerAxis.CASUAL, PowerAxis.CONFLICT, PowerAxis.UPWARD, PowerAxis.LATERAL, PowerAxis.DOWNWARD]:
        reframed = synthesizer._clean_and_reframe("basically matlab you know i just think we could improve", "system throughput", axis)
        assert "basically" not in reframed
        assert "matlab" not in reframed

    # _extract_core_topic
    assert synthesizer._extract_core_topic("Let's review database migrations and latency") == "review database"
    assert synthesizer._extract_core_topic("deploy") == "deploy"
    assert synthesizer._extract_core_topic("ok") == "the core topic"


def test_local_synthesizer_ollama_mock():
    """Test Ollama LLM queries with successful response and error fallback."""
    synthesizer = LocalCoachingSynthesizer()
    dialogue = [Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="We will ship on Thursday.")]
    profile = PersonaProfile(persona_name="Executive", strategic_focus="Brevity", power_axis=PowerAxis.UPWARD)
    metrics = CommunicationMetrics(presence_score=80)

    # 1. Mock successful Ollama response
    mock_tags_resp = MagicMock()
    mock_tags_resp.status = 200
    mock_tags_resp.__enter__.return_value = mock_tags_resp
    mock_tags_resp.read.return_value = json.dumps({"models": [{"name": "gemma2:2b"}]}).encode("utf-8")

    mock_generate_resp = MagicMock()
    mock_generate_resp.status = 200
    mock_generate_resp.__enter__.return_value = mock_generate_resp
    sample_eval = {
        "persona_context": "Brevity",
        "metrics": {"presence_score": 85},
        "top_strengths": [{"observation": "Clear focus", "verbatim_quote": "We will ship."}],
        "areas_for_improvement": [{"critique": "Lead with BLUF", "verbatim_quote": "We will ship.", "coached_phrasing": "Ship Thursday."}],
        "action_items": [],
        "key_highlights": [],
        "longitudinal_summary": "Good concise delivery.",
        "persona_alignment_notes": "Aligned with UPWARD"
    }
    mock_generate_resp.read.return_value = json.dumps({"response": json.dumps(sample_eval)}).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [mock_tags_resp, mock_generate_resp]
        result = synthesizer._try_ollama_local_inference(dialogue, profile, metrics, top_n=2)
        assert result is not None
        assert result.persona_context == "Brevity"

    # 2. Mock Ollama tags empty / fail
    mock_empty_tags = MagicMock()
    mock_empty_tags.status = 200
    mock_empty_tags.__enter__.return_value = mock_empty_tags
    mock_empty_tags.read.return_value = json.dumps({"models": []}).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = mock_empty_tags
        assert synthesizer._try_ollama_local_inference(dialogue, profile, metrics, top_n=2) is None

    # 3. Mock Ollama network error
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert synthesizer._try_ollama_local_inference(dialogue, profile, metrics, top_n=2) is None
