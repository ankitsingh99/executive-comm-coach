"""Privacy and DPDP Compliance module."""

from .dpdp_compliance import ConsentRecord, DPDPComplianceManager
from .pii_redactor import PIIRedactor

__all__ = ["PIIRedactor", "DPDPComplianceManager", "ConsentRecord"]
