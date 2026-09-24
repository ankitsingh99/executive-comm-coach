"""
Unit tests for Smart Temporal & AM/PM Next-Occurrence Resolver Engine.
"""

from datetime import datetime
from engine.temporal_resolver import TemporalResolver


def test_next_occurrence_ampm_at_8am():
    # At 8:15 AM, "call xyz at 9" should resolve to 9:00 AM today (inferred AM)
    ref_dt = datetime(2026, 9, 12, 8, 15, 0)
    res = TemporalResolver.resolve_time_expression("call xyz at 9", ref_dt=ref_dt)

    assert res is not None
    assert res.inferred_ampm == "AM"
    assert "Today at 9:00 AM" in res.formatted_label
    assert res.resolved_datetime == "2026-09-12T09:00:00"


def test_next_occurrence_ampm_at_11am():
    # At 11:30 AM, "call xyz at 9" should resolve to 9:00 PM today (inferred PM)
    ref_dt = datetime(2026, 9, 12, 11, 30, 0)
    res = TemporalResolver.resolve_time_expression("call xyz at 9", ref_dt=ref_dt)

    assert res is not None
    assert res.inferred_ampm == "PM"
    assert "Today at 9:00 PM" in res.formatted_label
    assert res.resolved_datetime == "2026-09-12T21:00:00"


def test_next_occurrence_ampm_at_10pm():
    # At 10:00 PM (22:00), "call xyz at 9" should resolve to 9:00 AM tomorrow (inferred AM)
    ref_dt = datetime(2026, 9, 12, 22, 0, 0)
    res = TemporalResolver.resolve_time_expression("call xyz at 9", ref_dt=ref_dt)

    assert res is not None
    assert res.inferred_ampm == "AM"
    assert "Tomorrow at 9:00 AM" in res.formatted_label
    assert res.resolved_datetime == "2026-09-13T09:00:00"


def test_next_occurrence_afternoon_meeting():
    # At 9:00 AM, "meet at 4" should resolve to 4:00 PM today (inferred PM)
    ref_dt = datetime(2026, 9, 12, 9, 0, 0)
    res = TemporalResolver.resolve_time_expression("meet at 4", ref_dt=ref_dt)

    assert res is not None
    assert res.inferred_ampm == "PM"
    assert "Today at 4:00 PM" in res.formatted_label
    assert res.resolved_datetime == "2026-09-12T16:00:00"


def test_explicit_ampm_preservation():
    # Explicit "9 AM" and "9 PM" should be honored
    ref_dt = datetime(2026, 9, 12, 11, 30, 0)
    res_am = TemporalResolver.resolve_time_expression("sync at 9 AM", ref_dt=ref_dt)
    assert res_am is not None
    assert res_am.inferred_ampm == "AM"
    assert "9:00 AM" in res_am.formatted_label

    res_pm = TemporalResolver.resolve_time_expression("sync at 9 PM", ref_dt=ref_dt)
    assert res_pm is not None
    assert res_pm.inferred_ampm == "PM"
    assert "9:00 PM" in res_pm.formatted_label


def test_hinglish_temporal_expressions():
    ref_dt = datetime(2026, 9, 12, 10, 0, 0)  # Saturday

    # "kal shaam 9 baje" -> Tomorrow at 9:00 PM
    res1 = TemporalResolver.resolve_time_expression("main kal shaam 9 baje update dunga", ref_dt=ref_dt)
    assert res1 is not None
    assert res1.inferred_ampm == "PM"
    assert "Tomorrow at 9:00 PM" in res1.formatted_label
    assert res1.resolved_datetime == "2026-09-13T21:00:00"

    # "subah 10 baje"
    res2 = TemporalResolver.resolve_time_expression("subah 10 baje sync karenge", ref_dt=ref_dt)
    assert res2 is not None
    assert res2.inferred_ampm == "AM"
    assert "10:00 AM" in res2.formatted_label


def test_relative_deadlines_eod_and_asap():
    ref_dt = datetime(2026, 9, 12, 14, 0, 0)

    res_eod = TemporalResolver.resolve_time_expression("please finish by EOD", ref_dt=ref_dt)
    assert res_eod is not None
    assert "EOD" in res_eod.formatted_label
    assert res_eod.urgency == "High"

    res_asap = TemporalResolver.resolve_time_expression("deploy the fix asap", ref_dt=ref_dt)
    assert res_asap is not None
    assert "ASAP" in res_asap.formatted_label
    assert res_asap.urgency == "High"
