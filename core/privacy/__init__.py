"""Privacy and DPDP Compliance module."""
from .pii_redactor import PIIRedactor
from .dpdp_compliance import DPDPComplianceManager, ConsentRecord

__all__ = ["PIIRedactor", "DPDPComplianceManager", "ConsentRecord"]
