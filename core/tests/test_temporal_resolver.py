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


def test_24h_format_and_qualifiers():
    ref_dt = datetime(2026, 9, 12, 10, 0, 0)
    # 1. 24-hour format
    res_24 = TemporalResolver.resolve_time_expression("deploy at 15:30", ref_dt=ref_dt)
    assert res_24 is not None
    assert "3:30 PM" in res_24.formatted_label

    # 2. Hinglish Subah / Shaam / Dopahar
    res_subah = TemporalResolver.resolve_time_expression("kal subah 8 baje sync karenge", ref_dt=ref_dt)
    assert res_subah is not None
    assert res_subah.inferred_ampm == "AM"

    res_shaam = TemporalResolver.resolve_time_expression("shaam ko 7 baje milte hain", ref_dt=ref_dt)
    assert res_shaam is not None
    assert res_shaam.inferred_ampm == "PM"

    res_dopahar = TemporalResolver.resolve_time_expression("dopahar 2 baje call hai", ref_dt=ref_dt)
    assert res_dopahar is not None
    assert res_dopahar.inferred_ampm == "PM"


def test_relative_and_day_of_week_deadlines():
    ref_dt = datetime(2026, 9, 12, 10, 0, 0)  # Saturday

    # 1. Tomorrow at ambiguous 10 (should be AM) and 4 (should be PM)
    res_tom_10 = TemporalResolver.resolve_time_expression("sync tomorrow at 10", ref_dt=ref_dt)
    assert res_tom_10 is not None
    assert res_tom_10.inferred_ampm == "AM"
    assert "Tomorrow at 10:00 AM" in res_tom_10.formatted_label

    res_tom_4 = TemporalResolver.resolve_time_expression("sync tomorrow at 4", ref_dt=ref_dt)
    assert res_tom_4 is not None
    assert res_tom_4.inferred_ampm == "PM"
    assert "Tomorrow at 4:00 PM" in res_tom_4.formatted_label

    # 2. Next Friday / EOD
    res_fri = TemporalResolver.resolve_time_expression("deliver by Friday", ref_dt=ref_dt)
    assert res_fri is not None
    assert "Friday" in res_fri.formatted_label

    # 3. Parso 4 baje
    res_parso = TemporalResolver.resolve_time_expression("parso 4 baje demo denge", ref_dt=ref_dt)
    assert res_parso is not None
    assert res_parso.inferred_ampm == "PM"

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
    res_asap = TemporalResolver.resolve_time_expression("deploy the fix asap", ref_dt=ref_dt)
    assert res_asap is not None
    assert "ASAP" in res_asap.formatted_label


def test_temporal_resolver_extended_deadlines_and_calendar_dates():
    """Test EOW, next week, tonight, today time resolution, and calendar date boundaries."""
    ref_dt = datetime(2026, 9, 12, 19, 30, 0)  # Saturday 7:30 PM

    # 1. EOD when current time is after 18:00 (rolls forward to tomorrow 6 PM)
    res_late_eod = TemporalResolver.resolve_time_expression("finish by eod", ref_dt=ref_dt)
    assert res_late_eod is not None
    assert "2026-09-13T18:00:00" in res_late_eod.resolved_datetime

    # 2. End of week / EOW
    ref_wed = datetime(2026, 9, 9, 10, 0, 0)  # Wednesday
    res_eow = TemporalResolver.resolve_time_expression("deliver by end of week", ref_dt=ref_wed)
    assert res_eow is not None
    assert "Friday by EOD" in res_eow.formatted_label

    # 3. Next week / agle hafte
    res_nw = TemporalResolver.resolve_time_expression("milte hain agle hafte", ref_dt=ref_wed)
    assert res_nw is not None
    assert "Next Week" in res_nw.formatted_label

    # 4. Tonight
    res_tonight = TemporalResolver.resolve_time_expression("call you tonight at 8", ref_dt=datetime(2026, 9, 12, 10, 0, 0))
    assert res_tonight is not None
    assert res_tonight.inferred_ampm == "PM"

    # 5. Today at 3 (ambiguous hour on today: picks 3 PM if morning)
    res_today = TemporalResolver.resolve_time_expression("meet today at 3", ref_dt=datetime(2026, 9, 12, 10, 0, 0))
    assert res_today is not None
    assert res_today.inferred_ampm == "PM"

    # 6. 'Parson' (Hindi alternative spelling of parso)
    res_parson = TemporalResolver.resolve_time_expression("parson subah 10 baje", ref_dt=ref_wed)
    assert res_parson is not None
    assert res_parson.inferred_ampm == "AM"

    # 7. Invalid date in text (e.g. 31 Feb)
    assert TemporalResolver.resolve_time_expression("deadline is 31 Feb 2026", ref_dt=ref_wed) is None

    # 8. Past date rollforward (> 180 days ago rolls to next year)
    res_roll = TemporalResolver.resolve_time_expression("due on 15 Jan", ref_dt=datetime(2026, 9, 12, 10, 0, 0))
    assert res_roll is not None
    assert "2027" in res_roll.resolved_datetime
