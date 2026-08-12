"""Dataset auditing and leakage-safe window indexing."""

from gridimpair.data.audit import AuditSummary, audit_source_dataset
from gridimpair.data.windows import make_grouped_windows

__all__ = ["AuditSummary", "audit_source_dataset", "make_grouped_windows"]
