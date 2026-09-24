"""
Smart Temporal & AM/PM Next-Occurrence Resolver Engine.
Intelligently detects time expressions (English, Hindi, Hinglish),
infers AM/PM for ambiguous clock times based on the earliest future occurrence
relative to a reference timestamp, and normalizes dates and deadlines into ISO formats.
"""

import re
from datetime import datetime, timedelta, date, time
from typing import Optional
from dataclasses import dataclass


@dataclass
class TemporalResolution:
    """Structured result from temporal resolution."""

    raw_match: str
    formatted_label: str  # e.g. "Today at 9:00 PM" or "Tomorrow at 9:00 AM"
    resolved_datetime: Optional[str]  # ISO 8601 string: "2026-09-12T21:00:00"
    inferred_ampm: Optional[str]  # "AM" | "PM" | None
    urgency: str  # "High" | "Normal"
    confidence: float  # 0.0 to 1.0


class TemporalResolver:
    """
    Intelligent time and date parser with next-occurrence AM/PM inference.
    """

    # Regex for days of the week
    WEEKDAYS = {
        "monday": 0,
        "somwar": 0,
        "tuesday": 1,
        "mangalwar": 1,
        "wednesday": 2,
        "budhwar": 2,
        "thursday": 3,
        "guruwar": 3,
        "veervar": 3,
        "friday": 4,
        "shukrawar": 4,
        "saturday": 5,
        "shanivar": 5,
        "sunday": 6,
        "ravivar": 6,
        "itwar": 6,
    }

    MONTHS = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    @classmethod
    def resolve_time_expression(cls, text: str, ref_dt: Optional[datetime] = None) -> Optional[TemporalResolution]:
        """
        Parses temporal expressions from text and resolves them relative to ref_dt.
        If AM/PM is omitted (e.g. 'call at 9'), infers the closest future occurrence.
        """
        if not text or not text.strip():
            return None

        if ref_dt is None:
            ref_dt = datetime.now()

        cleaned_text = text.strip()

        # 1. Check for specific date patterns e.g. "31 Aug at 10 AM", "Aug 31st at 10:30 PM", "31 aug ko 10 baje"
        specific_date_res = cls._parse_specific_date(cleaned_text, ref_dt)
        if specific_date_res:
            return specific_date_res

        # 2. Check for day names / relative days + optional time e.g. "Friday at 4", "tomorrow at 9", "kal shaam 9 baje"
        relative_day_res = cls._parse_relative_day_and_time(cleaned_text, ref_dt)
        if relative_day_res:
            return relative_day_res

        # 3. Check for standalone time expressions e.g. "at 9", "call xyz at 9", "meet at 4:30", "by 5 pm", "10 baje"
        standalone_time_res = cls._parse_standalone_time(cleaned_text, ref_dt)
        if standalone_time_res:
            return standalone_time_res

        # 4. Check for general relative deadlines e.g. "EOD", "end of week", "asap", "next week", "shaam tak"
        general_deadline_res = cls._parse_general_deadline(cleaned_text, ref_dt)
        if general_deadline_res:
            return general_deadline_res

        return None

    @classmethod
    def _parse_standalone_time(cls, text: str, ref_dt: datetime) -> Optional[TemporalResolution]:
        """
        Detects standalone clock times: 'at 9', 'at 9:30', 'by 5', 'call xyz at 9', 'shaam 5 baje', '9 baje', etc.
        """
        # Pattern for clock times with or without explicit AM/PM
        # Examples: "at 9", "at 9:30", "by 5 PM", "around 11 am", "9 baje", "shaam 9 baje", "subah 8 baje"
        pattern = re.compile(
            r"\b(?:(?:at|by|around|around\s+about|ko|tak|for)\s+)?"
            r"(?:(subah|shaam|dopahar|raat|morning|evening|afternoon|night)\s+)?"
            r"(\d{1,2})(?::(\d{2}))?\s*"
            r"(?:(am|pm|baje|o['’]?clock))?"
            r"(?:\s+(subah|shaam|dopahar|raat|morning|evening|afternoon|night))?\b",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            prefix_qualifier = match.group(1)
            hour_str = match.group(2)
            min_str = match.group(3)
            suffix_marker = match.group(4)
            suffix_qualifier = match.group(5)

            if not hour_str:
                continue

            hour = int(hour_str)
            minute = int(min_str) if min_str else 0

            # Guard against invalid hour/minute numbers
            if hour < 1 or hour > 24 or minute < 0 or minute > 59:
                continue

            raw_matched = match.group(0).strip()
            # If there's no pre/post preposition and no am/pm/baje/o'clock/qualifier, check if preceded by 'at'/'by'/'around' or an action verb
            has_trigger = bool(
                re.search(
                    r"\b(at|by|around|ko|tak|baje|am|pm|o['’]?clock|subah|shaam|dopahar|raat|morning|evening|night)\b",
                    raw_matched,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(call|meet|sync|ring|connect|talk|ping|start|ship|deliver|send|deploy|finish|complete)\b.*?"
                    + re.escape(raw_matched),
                    text,
                    re.IGNORECASE,
                )
            )

            if not has_trigger:
                continue

            qualifier = (prefix_qualifier or suffix_qualifier or "").lower()
            marker = (suffix_marker or "").lower()

            # Determine if explicit AM or PM is specified
            explicit_ampm: Optional[str] = None
            if marker in ["am", "pm"]:
                explicit_ampm = marker.upper()
            elif qualifier in ["subah", "morning"]:
                explicit_ampm = "AM"
            elif qualifier in ["shaam", "raat", "evening", "night"]:
                explicit_ampm = "PM"
            elif qualifier in ["dopahar", "afternoon"]:
                explicit_ampm = "PM" if hour != 12 else "PM"

            if hour > 12:
                # 24-hour format e.g. 14:00
                resolved_dt = ref_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if resolved_dt < ref_dt:
                    resolved_dt += timedelta(days=1)
                inferred = "PM"
                formatted = f"{resolved_dt.strftime('%b %d')} at {resolved_dt.strftime('%I:%M %p').lstrip('0')}"
                return TemporalResolution(
                    raw_match=raw_matched,
                    formatted_label=formatted,
                    resolved_datetime=resolved_dt.isoformat(),
                    inferred_ampm=inferred,
                    urgency="High" if (resolved_dt - ref_dt).total_seconds() < 86400 else "Normal",
                    confidence=0.95,
                )

            if explicit_ampm:
                # Exact AM/PM is known
                target_hour = hour % 12 if explicit_ampm == "AM" else (hour % 12) + 12
                candidate_dt = ref_dt.replace(hour=target_hour, minute=minute, second=0, microsecond=0)
                if candidate_dt < ref_dt:
                    candidate_dt += timedelta(days=1)

                day_label = "Today" if candidate_dt.date() == ref_dt.date() else "Tomorrow"
                time_label = candidate_dt.strftime("%I:%M %p").lstrip("0")
                formatted = f"{day_label} at {time_label}"

                return TemporalResolution(
                    raw_match=raw_matched,
                    formatted_label=formatted,
                    resolved_datetime=candidate_dt.isoformat(),
                    inferred_ampm=explicit_ampm,
                    urgency="High" if (candidate_dt - ref_dt).total_seconds() < 86400 else "Normal",
                    confidence=0.98,
                )

            # --- AMBIGUOUS TIME: Calculate Next Occurrence of the clock time ---
            # Generate Candidate 1 (AM Today), Candidate 2 (PM Today), Candidate 3 (AM Tomorrow)
            am_hour = hour % 12
            pm_hour = (hour % 12) + 12

            candidates = [
                ref_dt.replace(hour=am_hour, minute=minute, second=0, microsecond=0),
                ref_dt.replace(hour=pm_hour, minute=minute, second=0, microsecond=0),
                (ref_dt + timedelta(days=1)).replace(hour=am_hour, minute=minute, second=0, microsecond=0),
                (ref_dt + timedelta(days=1)).replace(hour=pm_hour, minute=minute, second=0, microsecond=0),
            ]

            # Choose the earliest candidate that is strictly in the future (or within a 1-minute margin)
            valid_candidates = [c for c in candidates if c >= ref_dt - timedelta(minutes=1)]
            valid_candidates.sort()
            best_dt = valid_candidates[0] if valid_candidates else candidates[2]

            inferred_ampm = "AM" if best_dt.hour < 12 else "PM"
            day_label = "Today" if best_dt.date() == ref_dt.date() else "Tomorrow"
            time_label = best_dt.strftime("%I:%M %p").lstrip("0")
            formatted = f"{day_label} at {time_label}"

            return TemporalResolution(
                raw_match=raw_matched,
                formatted_label=formatted,
                resolved_datetime=best_dt.isoformat(),
                inferred_ampm=inferred_ampm,
                urgency="High" if (best_dt - ref_dt).total_seconds() < 86400 else "Normal",
                confidence=0.92,
            )

        return None

    @classmethod
    def _parse_relative_day_and_time(cls, text: str, ref_dt: datetime) -> Optional[TemporalResolution]:
        """
        Parses relative day phrases e.g. 'tomorrow at 9', 'kal 10 baje', 'this friday by 5 PM', 'tonight at 8'.
        """
        pattern = re.compile(
            r"\b(today|tomorrow|tonight|kal|aaj|parson|this\s+(?:morning|afternoon|evening)|"
            r"(?:next|this|on|is)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|somwar|mangalwar|budhwar|guruwar|shukrawar|shanivar|ravivar))"
            r"(?:\s+(morning|afternoon|evening|night|subah|shaam|dopahar|raat|ko|tak))?"
            r"(?:\s+(?:at|by|around|ko|mein)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje)?)?\b",
            re.IGNORECASE,
        )

        match = pattern.search(text)
        if not match:
            return None

        day_term = match.group(1).lower().strip()
        day_qualifier = (match.group(2) or "").lower().strip()
        hour_str = match.group(3)
        min_str = match.group(4)
        time_marker = (match.group(5) or "").lower().strip()

        target_date = ref_dt.date()
        day_offset = 0

        if "tomorrow" in day_term or "kal" in day_term:
            day_offset = 1
        elif "parson" in day_term:
            day_offset = 2
        elif "today" in day_term or "aaj" in day_term:
            day_offset = 0
        elif "tonight" in day_term:
            day_offset = 0
            if not day_qualifier:
                day_qualifier = "night"
        else:
            # Check weekday
            for day_name, target_weekday in cls.WEEKDAYS.items():
                if day_name in day_term:
                    current_weekday = ref_dt.weekday()
                    days_ahead = (target_weekday - current_weekday) % 7
                    if days_ahead == 0 and "next" in day_term:
                        days_ahead = 7
                    elif days_ahead == 0 and (ref_dt.hour >= 18 or "this" not in day_term):
                        days_ahead = 7
                    day_offset = days_ahead
                    break

        target_date = ref_dt.date() + timedelta(days=day_offset)

        # If hour is provided
        if hour_str:
            hour = int(hour_str)
            minute = int(min_str) if min_str else 0

            explicit_ampm = None
            if time_marker in ["am", "pm"]:
                explicit_ampm = time_marker.upper()
            elif day_qualifier in ["subah", "morning"]:
                explicit_ampm = "AM"
            elif day_qualifier in ["shaam", "raat", "evening", "night"]:
                explicit_ampm = "PM"
            elif day_qualifier in ["dopahar", "afternoon"]:
                explicit_ampm = "PM"

            if explicit_ampm:
                target_hour = hour % 12 if explicit_ampm == "AM" else (hour % 12) + 12
                resolved_dt = datetime.combine(target_date, time(hour=target_hour, minute=minute))
                inferred = explicit_ampm
            else:
                # If target day is today, pick next upcoming occurrence
                if day_offset == 0:
                    candidates = [
                        datetime.combine(target_date, time(hour=hour % 12, minute=minute)),
                        datetime.combine(target_date, time(hour=(hour % 12) + 12, minute=minute)),
                    ]
                    valid = [c for c in candidates if c >= ref_dt - timedelta(minutes=1)]
                    resolved_dt = valid[0] if valid else candidates[1]
                    inferred = "AM" if resolved_dt.hour < 12 else "PM"
                else:
                    # Future day with ambiguous hour (e.g. "tomorrow at 9")
                    # Standard work context default: 9-11 is AM, 1-6 is PM, or nearest business hour
                    if hour >= 8 and hour <= 11:
                        target_hour = hour
                        inferred = "AM"
                    elif hour >= 1 and hour <= 7:
                        target_hour = hour + 12
                        inferred = "PM"
                    else:
                        target_hour = hour
                        inferred = "AM" if hour < 12 else "PM"
                    resolved_dt = datetime.combine(target_date, time(hour=target_hour, minute=minute))

            day_display = (
                "Today"
                if target_date == ref_dt.date()
                else (
                    "Tomorrow"
                    if target_date == ref_dt.date() + timedelta(days=1)
                    else target_date.strftime("%A, %b %d")
                )
            )
            time_display = resolved_dt.strftime("%I:%M %p").lstrip("0")
            formatted = f"{day_display} at {time_display}"

            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label=formatted,
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm=inferred,
                urgency="High" if day_offset <= 1 else "Normal",
                confidence=0.95,
            )
        else:
            # Day only (e.g., "by Friday", "tomorrow morning", "kal shaam tak")
            default_hour = 17  # 5 PM default for EOD/day deadlines
            if day_qualifier in ["subah", "morning"]:
                default_hour = 10
            elif day_qualifier in ["dopahar", "afternoon"]:
                default_hour = 14
            elif day_qualifier in ["shaam", "raat", "evening", "night"]:
                default_hour = 20

            resolved_dt = datetime.combine(target_date, time(hour=default_hour, minute=0))
            day_display = (
                "Today"
                if target_date == ref_dt.date()
                else (
                    "Tomorrow"
                    if target_date == ref_dt.date() + timedelta(days=1)
                    else target_date.strftime("%A, %b %d")
                )
            )
            time_qual_str = f" ({day_qualifier.capitalize()})" if day_qualifier else ""
            formatted = f"{day_display}{time_qual_str}"

            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label=formatted,
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm="PM" if default_hour >= 12 else "AM",
                urgency="High" if day_offset <= 1 else "Normal",
                confidence=0.90,
            )

    @classmethod
    def _parse_specific_date(cls, text: str, ref_dt: datetime) -> Optional[TemporalResolution]:
        """
        Parses explicit dates like '31 Aug at 10 AM', 'Aug 31st at 10:30 PM', '15 September'.
        """
        pattern = re.compile(
            r"\b(?:(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]{3,9})|([a-zA-Z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?)"
            r"(?:\s+(?:at|by|around|ko)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje)?)?\b",
            re.IGNORECASE,
        )

        match = pattern.search(text)
        if not match:
            return None

        day_1 = match.group(1)
        month_1 = match.group(2)
        month_2 = match.group(3)
        day_2 = match.group(4)

        day_num = int(day_1 if day_1 else day_2)
        month_str = (month_1 if month_1 else month_2).lower()

        month_num = None
        for m_key, m_val in cls.MONTHS.items():
            if m_key == month_str or month_str.startswith(m_key):
                month_num = m_val
                break

        if not month_num:
            return None

        current_year = ref_dt.year
        try:
            target_date = date(current_year, month_num, day_num)
            # If target date in current year is > 6 months in the past, assume next year
            if (ref_dt.date() - target_date).days > 180:
                target_date = date(current_year + 1, month_num, day_num)
        except ValueError:
            return None

        hour_str = match.group(5)
        min_str = match.group(6)
        marker = (match.group(7) or "").lower()

        hour = int(hour_str) if hour_str else 10
        minute = int(min_str) if min_str else 0

        inferred_ampm = None
        if marker in ["am", "pm"]:
            inferred_ampm = marker.upper()
            hour = hour % 12 if inferred_ampm == "AM" else (hour % 12) + 12
        else:
            inferred_ampm = "AM" if hour < 12 else "PM"

        resolved_dt = datetime.combine(target_date, time(hour=hour, minute=minute))
        time_part = f" at {resolved_dt.strftime('%I:%M %p').lstrip('0')}" if hour_str else ""
        formatted = f"{target_date.strftime('%d %b')}{time_part}"

        return TemporalResolution(
            raw_match=match.group(0).strip(),
            formatted_label=formatted,
            resolved_datetime=resolved_dt.isoformat(),
            inferred_ampm=inferred_ampm,
            urgency="High" if (resolved_dt - ref_dt).total_seconds() < 86400 * 2 else "Normal",
            confidence=0.98,
        )

    @classmethod
    def _parse_general_deadline(cls, text: str, ref_dt: datetime) -> Optional[TemporalResolution]:
        """
        Parses general deadline markers like 'EOD', 'end of day', 'end of week', 'asap', 'next week'.
        """
        pattern = re.compile(
            r"\b(eod|end\s+of\s+day|end\s+of\s+week|eow|asap|next\s+week|agle\s+hafte|is\s+hafte|shaam\s+tak|aaj\s+shaam)\b",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None

        matched_phrase = match.group(1).lower()
        if (
            "eod" in matched_phrase
            or "end of day" in matched_phrase
            or "shaam tak" in matched_phrase
            or "aaj shaam" in matched_phrase
        ):
            resolved_dt = ref_dt.replace(hour=18, minute=0, second=0, microsecond=0)
            if resolved_dt < ref_dt:
                resolved_dt += timedelta(days=1)
            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label="Today by EOD (6:00 PM)",
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm="PM",
                urgency="High",
                confidence=0.90,
            )
        elif "asap" in matched_phrase:
            resolved_dt = ref_dt + timedelta(hours=2)
            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label="ASAP (Within 2 hours)",
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm="AM" if resolved_dt.hour < 12 else "PM",
                urgency="High",
                confidence=0.95,
            )
        elif "end of week" in matched_phrase or "eow" in matched_phrase:
            days_until_friday = (4 - ref_dt.weekday()) % 7
            if days_until_friday == 0 and ref_dt.hour >= 18:
                days_until_friday = 7
            target_date = ref_dt.date() + timedelta(days=days_until_friday)
            resolved_dt = datetime.combine(target_date, time(hour=18, minute=0))
            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label="Friday by EOD",
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm="PM",
                urgency="Normal",
                confidence=0.90,
            )
        elif "next week" in matched_phrase or "agle hafte" in matched_phrase:
            days_until_next_mon = (7 - ref_dt.weekday()) % 7 or 7
            target_date = ref_dt.date() + timedelta(days=days_until_next_mon)
            resolved_dt = datetime.combine(target_date, time(hour=10, minute=0))
            return TemporalResolution(
                raw_match=match.group(0).strip(),
                formatted_label=f"Next Week ({target_date.strftime('%b %d')})",
                resolved_datetime=resolved_dt.isoformat(),
                inferred_ampm="AM",
                urgency="Normal",
                confidence=0.88,
            )

        return None
