"""
Enumerations for Schema Metamodel and Lifecycle (SCH-001, SCH-004).
Strictly adheres to PRD v1.1 §6.1, §6.2, and §10.1.
"""

from enum import Enum


class SchemaStatus(str, Enum):
    """
    Schema lifecycle status matching PRD SCH-004:
    - draft: initial authoring state
    - testing: under evaluation in test harness
    - approved: approved for production extraction
    - retired: archived / superseded version
    """
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    RETIRED = "retired"


class FieldType(str, Enum):
    """
    Supported field types matching PRD §6.1 SCH-001:
    string, integer, decimal, boolean, date, enum, object, array.
    """
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    OBJECT = "object"
    ARRAY = "array"


class PIIClass(str, Enum):
    """
    PII classifications matching PRD §6.1, §6.2, §10.1 (SEC-007, SCH-014):
    - direct_identifier: Direct personal identifiers (e.g. Full Name, SSN, PAN)
    - government_id: Government identity numbers (e.g. PAN, Aadhaar, Passport)
    - financial_identifier: Bank account numbers, credit card numbers
    - contact_information: Phone numbers, email addresses, residential address
    - health_information: Protected Health Information (PHI), diagnoses, prescriptions
    - biometric: Biometric records, photographs, signatures
    - sensitive_personal: Sensitive categories like religion, sexual orientation, caste
    """
    DIRECT_IDENTIFIER = "direct_identifier"
    GOVERNMENT_ID = "government_id"
    FINANCIAL_IDENTIFIER = "financial_identifier"
    CONTACT_INFORMATION = "contact_information"
    HEALTH_INFORMATION = "health_information"
    BIOMETRIC = "biometric"
    SENSITIVE_PERSONAL = "sensitive_personal"


class RuleOnFail(str, Enum):
    """
    Outcome when a validation or business rule fails (PRD §6.2, FR-VAL-006):
    - review: Routes document to human review queue (SCR-04)
    - reject: Flags document for rejection
    - flag: Attaches an advisory flag to envelope without forcing review
    - fail: Hard failure status
    """
    REVIEW = "review"
    REJECT = "reject"
    FLAG = "flag"
    FAIL = "fail"

