"""
Automated unit tests for DPDP Act Compliance, Audible Chime, and Right to Erasure.
"""

import os
import tempfile
from privacy.dpdp_compliance import DPDPComplianceManager


def test_dpdp_consent_and_erasure():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DPDPComplianceManager(storage_root=temp_dir)
        session_id = "test_dpdp_session_999"

        # Test consent logging
        consent = manager.log_session_consent(session_id, counterpart_notified=True)
        assert consent.session_id == session_id
        assert consent.counterpart_notified is True
        assert consent.audible_chime_played is True

        # Test chime trigger
        chime_msg = manager.trigger_audible_chime()
        assert "DPDP_CHIME_TRIGGERED" in chime_msg

        # Create mock session files to test erasure
        session_path = os.path.join(temp_dir, session_id)
        os.makedirs(session_path, exist_ok=True)
        sample_audio = os.path.join(session_path, "audio.enc")
        with open(sample_audio, "wb") as f:
            f.write(b"ENCRYPTED_AUDIO_DATA_AES256")

        assert os.path.exists(sample_audio)

        # Execute statutory right to erasure
        erase_result = manager.execute_statutory_erasure(session_id)
        assert erase_result["status"] == "ERASED"
        assert erase_result["session_id"] == session_id
        assert not os.path.exists(session_path)
        assert session_id not in manager.consent_logs
