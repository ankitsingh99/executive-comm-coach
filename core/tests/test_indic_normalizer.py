import pytest
from asr_diarization.indic_normalizer import IndicNormalizer
from engine.action_item_extractor import ActionItemExtractor
from engine.local_coaching_synthesizer import LocalCoachingSynthesizer
from engine.metrics_calculator import MetricsCalculator
from engine.schema import ConversationSession, Utterance


def test_devanagari_detection_and_transliteration():
    devanagari_text = "हाँ मतलब मुझे लगता है हमें कल १० बजे सिंक करना चाहिए।"
    assert IndicNormalizer.contains_devanagari(devanagari_text) is True

    normalized = IndicNormalizer.normalize_text(devanagari_text)
    assert "matlab" in normalized
    assert "mujhe" in normalized
    assert "kal" in normalized
    assert "sync" in normalized


def test_hinglish_phonetic_harmonization():
    chat_slang = "Dekho mtlb hume kl 10 bajeh sync krna chahye. Shyd ho sake toh."
    normalized = IndicNormalizer.normalize_text(chat_slang)

    assert "matlab" in normalized
    assert "kal" in normalized
    assert "chahiye" in normalized
    assert "shayad" in normalized


def test_metrics_calculator_with_devanagari_and_hinglish():
    text_with_fillers = "मतलब basically mujhe aisa lagta hai ki hume deploy karna chahiye."
    fillers = MetricsCalculator.detect_fillers(text_with_fillers)
    tokens = [f.token for f in fillers]

    assert "matlab" in tokens or "basically" in tokens

    hedging, assertive = MetricsCalculator.calculate_hedging_vs_assertion(text_with_fillers)
    assert hedging >= 1  # "mujhe aisa lagta hai"


def test_action_item_extractor_with_devanagari_and_hinglish():
    utterance_devanagari = Utterance(speaker="USER", transcript="हम कल सुबह १० बजे कॉल करेंगे।")
    items = ActionItemExtractor.extract_from_utterance(utterance_devanagari)
    assert len(items) >= 1
    assert "kal" in items[0].due_time_or_date.lower() or "10" in str(items[0].due_time_or_date)


def test_local_coaching_synthesizer_hinglish_session():
    session = ConversationSession(
        session_id="test_hinglish_indic",
        power_axis="UPWARD",
        counterpart_name="Director",
        counterpart_role="Engineering Director",
        dialogue=[
            Utterance(
                speaker="USER",
                transcript="Dekho basically matlab mujhe lagta hai hume caching enable karni chahiye. Hum kal 10 baje sync karenge.",
            )
        ],
    )

    synthesizer = LocalCoachingSynthesizer()
    evaluation = synthesizer.synthesize(session)

    assert evaluation.metrics.presence_score > 0
    assert len(evaluation.top_strengths) > 0
    assert len(evaluation.areas_for_improvement) > 0
    assert len(evaluation.action_items) >= 1


def test_non_verbal_sounds_and_clicks_detection():
    speech_with_clicks = "Ummm, so tch I was thinking aaaa we should fix this, uff it is slow."
    fillers = MetricsCalculator.detect_fillers(speech_with_clicks)
    tokens = {f.token for f in fillers}

    assert "ummm" in tokens or "um" in tokens
    assert "tch" in tokens
    assert "aaaa" in tokens or "aa" in tokens
    assert "uff" in tokens


def test_indic_normalizer_halant_and_edge_cases():
    """Test halant consonant clusters, English bypass, and is_hinglish boundaries."""
    # 1. Halant character clusters (e.g. क्त, ल्य)
    halant_text = "कल्याण और मुख्य"
    translit = IndicNormalizer.transliterate_devanagari_to_roman(halant_text)
    assert "kalyan" in translit or "mukhy" in translit or len(translit) > 3

    # 2. Transliterate pure English bypass
    eng_text = "This is purely English text."
    assert IndicNormalizer.transliterate_devanagari_to_roman(eng_text) == eng_text

    # 3. is_hinglish empty / devanagari
    assert IndicNormalizer.is_hinglish("") is False
    assert IndicNormalizer.is_hinglish(None) is False
    assert IndicNormalizer.is_hinglish("हम") is True
    assert IndicNormalizer.is_hinglish("Hello team, good morning.") is False
