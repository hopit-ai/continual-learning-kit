"""Protect exported pytest failure reports from environment-value disclosure."""
from env_redaction import (pytest_make_collect_report, pytest_runtest_makereport,
                           pytest_warning_recorded)
