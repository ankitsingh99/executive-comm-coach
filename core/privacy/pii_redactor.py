"""
Local Privacy-Preserving PII and Sensitive Data Redaction Engine.
Implements DPDP Act (2023) privacy-by-design safeguards before text is stored or processed.
"""

import re
from typing import Tuple, List, Dict


class PIIRedactor:
    """Local regex and entity redaction scanner for sensitive workplace dialogue."""

    # Patterns for sensitive Indian identifiers, financials, and credentials
    PHONE_PATTERN = r"\b(?:\+?91[-.\s]?)?[6789]\d{9}\b"
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    PAN_PATTERN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    AADHAAR_PATTERN = r"\b\d{4}\s\d{4}\s\d{4}\b"
    FINANCIAL_PATTERN = r"(?:(?:Rs\.?|₹|INR|\$|USD)\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:lakh|crore|k|million|bn))?|\b\d+\s*(?:lakh|crore|k|million|billion)\s*(?:rupees|inr|dollars)?)"
    SECRET_KEY_PATTERN = r"(?:api_key|password|secret|bearer|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,})['\"]?"
    INTERNAL_URL_PATTERN = r"https?://(?:internal\.|corp\.|staging\.|[a-z0-9-]+\.internal)[^\s]*"

    @classmethod
    def redact_text(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redacts sensitive PII and secrets, returning (redacted_text, redaction_counts).
        """
        redacted = text
        counts: Dict[str, int] = {
            "PHONE": 0,
            "EMAIL": 0,
            "PAN": 0,
            "AADHAAR": 0,
            "FINANCIAL": 0,
            "CREDENTIAL": 0,
            "INTERNAL_URL": 0,
        }

        # Internal URLs
        urls = re.findall(cls.INTERNAL_URL_PATTERN, redacted, flags=re.IGNORECASE)
        counts["INTERNAL_URL"] = len(urls)
        redacted = re.sub(cls.INTERNAL_URL_PATTERN, "[REDACTED_INTERNAL_URL]", redacted, flags=re.IGNORECASE)

        # Email
        emails = re.findall(cls.EMAIL_PATTERN, redacted, flags=re.IGNORECASE)
        counts["EMAIL"] = len(emails)
        redacted = re.sub(cls.EMAIL_PATTERN, "[REDACTED_EMAIL]", redacted, flags=re.IGNORECASE)

        # Phone
        phones = re.findall(cls.PHONE_PATTERN, redacted)
        counts["PHONE"] = len(phones)
        redacted = re.sub(cls.PHONE_PATTERN, "[REDACTED_PHONE]", redacted)

        # PAN
        pans = re.findall(cls.PAN_PATTERN, redacted)
        counts["PAN"] = len(pans)
        redacted = re.sub(cls.PAN_PATTERN, "[REDACTED_PAN]", redacted)

        # Aadhaar
        aadhaar = re.findall(cls.AADHAAR_PATTERN, redacted)
        counts["AADHAAR"] = len(aadhaar)
        redacted = re.sub(cls.AADHAAR_PATTERN, "[REDACTED_AADHAAR]", redacted)

        # Financial values (salary, deals, budgets)
        financials = re.findall(cls.FINANCIAL_PATTERN, redacted, flags=re.IGNORECASE)
        counts["FINANCIAL"] = len(financials)
        redacted = re.sub(cls.FINANCIAL_PATTERN, "[REDACTED_FINANCIAL]", redacted, flags=re.IGNORECASE)

        # Credentials & API secrets
        secrets = re.findall(cls.SECRET_KEY_PATTERN, redacted, flags=re.IGNORECASE)
        counts["CREDENTIAL"] = len(secrets)
        redacted = re.sub(cls.SECRET_KEY_PATTERN, "secret: [REDACTED_SECRET]", redacted, flags=re.IGNORECASE)

        return redacted, counts
