"""
Interactive Local On-Device Executive Communication Coach CLI.
Enables instant evaluation of conversations completely offline.
"""

import sys
import os
from datetime import datetime, timezone

# Ensure path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.schema import ConversationSession, Utterance
from engine.persona_ontology import PowerAxis
from engine.coaching_engine import ExecutiveCoachingEngine
from asr_diarization.local_stt_engine import LocalSTTEngine
from privacy.pii_redactor import PIIRedactor
from privacy.dpdp_compliance import DPDPComplianceManager
from config import DATA_DIR


def run_local_cli():
    print("=" * 80)
    print(" ON-DEVICE EXECUTIVE COMMUNICATION COACH (LOCAL MODE)")
    print("=" * 80)
    print(" Mode: 100% Local On-Device Execution (Zero Cloud API / Zero Latency)\n")

    sample_dialogue = """
COUNTERPART: Hey, let's review the quarterly infrastructure costs. Are we still on track?
USER: Yeah so basically, matlab we were looking at the logs and I just think maybe we could reduce AWS spend by 15%, but there were some team blockers.
COUNTERPART: Can you give me the exact numbers and the timeline for completion?
USER: Understood. Our data demonstrates that caching reduced database load by 35%. We have decided to deploy the cost-saving policy on Thursday morning.
COUNTERPART: Perfect, I will call you on 31 aug at 10 am to review the cost savings.
"""

    stt_engine = LocalSTTEngine()
    utterances = stt_engine.process_local_transcript(sample_dialogue)

    from asr_diarization.diarizer import DiarizationEngine

    print("--- Transcribed Dialogue & Speaker Attribution ---")
    print(DiarizationEngine.format_dialogue_cli(utterances))
    print("-" * 80)

    counterparts = [
        ("Sandeep Sharma", "VP of Engineering", PowerAxis.UPWARD.value),
        ("Pooja Nair", "Principal Product Manager", PowerAxis.LATERAL.value),
        ("Kunal Verma", "Junior Software Engineer", PowerAxis.DOWNWARD.value),
    ]

    engine = ExecutiveCoachingEngine(use_local_only=True)
    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)

    for name, role, axis in counterparts:
        session_id = f"local_{int(datetime.now(timezone.utc).timestamp())}_{axis.lower()}"
        session = ConversationSession(
            session_id=session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_speaker="USER",
            counterpart_name=name,
            counterpart_role=role,
            power_axis=axis,
            dialogue=utterances
        )

        evaluation = engine.evaluate_session(session, top_n=2)

        print(f"\n[LOCAL EVALUATION] Counterpart: {name} ({role}) | Power Axis: {axis}")
        print(f"   Presence: {evaluation.metrics.presence_score}/100 | Assertiveness: {evaluation.metrics.assertiveness_score}/100 | Active Listening: {evaluation.metrics.active_listening_score}/100")
        print(f"   Fillers: {[f'{f.token}: {f.count}' for f in evaluation.metrics.filler_words_detected]}")
        print(f"   Focus:    {evaluation.persona_context}")
        print(f"   Advice:   {evaluation.longitudinal_summary}")

        print("   Top Strengths:")
        for idx, s in enumerate(evaluation.top_strengths, 1):
            print(f"      {idx}. {s.observation} (Quote: \"{s.verbatim_quote}\")")

        print("   Areas for Improvement & Coached Alternatives:")
        for idx, a in enumerate(evaluation.areas_for_improvement, 1):
            print(f"      {idx}. Critique: {a.critique}")
            print(f"         Coached:  \"{a.coached_phrasing}\"")

        if evaluation.action_items:
            print("   Detected Action Items & Commitments:")
            for idx, ai in enumerate(evaluation.action_items, 1):
                due_info = f" | Due: {ai.due_time_or_date}" if ai.due_time_or_date else ""
                print(f"      📌 {idx}. [{ai.owner}] {ai.category}{due_info}: {ai.task}")
                print(f"         Quote: \"{ai.verbatim_quote}\"")

    print("\n" + "=" * 80)
    print(" Local processing and parsing completed successfully on your device.")
    print("=" * 80)


if __name__ == "__main__":
    run_local_cli()
