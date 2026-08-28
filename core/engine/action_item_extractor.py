"""
On-Device Action Item, Commitment & Follow-up Extraction Engine.
Extracts verbal commitments, scheduling promises, deliverables, and deadlines
from dialogue turns with ownership, temporal anchors, and urgency classification.
"""

import re
from typing import List, Optional, Tuple, Dict, Any

try:
    from .schema import Utterance, ActionItem
except (ImportError, ValueError):
    from engine.schema import Utterance, ActionItem


# Regex for temporal dates, days, times, and deadlines (English + Hinglish)
TIME_PATTERNS = [
    # "31 Aug at 10 AM", "Aug 31st at 10:30 PM", "31 aug ko 10 baje"
    re.compile(
        r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?)"
        r"(?:\s+(?:at|by|around|ko)?\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|baje)?)?\b",
        re.IGNORECASE
    ),
    # "tomorrow at 10 AM", "Friday by 5 PM", "kal 10 baje", "kal shaam tak", "shaam tak", "is friday ko"
    re.compile(
        r"\b(?:today|tomorrow|tonight|kal|aaj|parson|shaam|subah|dopahar|raat|this\s+(?:morning|afternoon|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"(?:next|on|by|is)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|somwar|mangalwar|budhwar|guruwar|shukrawar|shanivar|ravivar|next\s+week|next\s+month|end\s+of\s+day|end\s+of\s+week|eod|agle\s+hafte|is\s+hafte))"
        r"(?:\s+(?:morning|afternoon|evening|night|eod|subah|shaam|dopahar|raat|ko|tak))*"
        r"(?:\s+(?:at|by|around|ko|mein)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm|baje)?)?\b",
        re.IGNORECASE
    ),
    # "at 10 AM", "at 3:30 PM", "by 5 PM", "10 baje", "shaam 5 baje"
    re.compile(r"\b(?:at|by|around|shaam|subah|dopahar|raat)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm|baje)\b", re.IGNORECASE)
]

# Action & Commitment Intent Patterns (English + Hinglish)
INTENT_PATTERNS = [
    # Scheduling & Follow-up calls (English + Hinglish)
    (
        re.compile(r"\b(?:i\s*will|i['’]ll|we\s*will|we['’]ll|let['’]s|let\s*us)\s+(?:call|connect|sync|ring|meet|set\s+up\s+a\s+call|schedule\s+a\s+sync|schedule\s+a\s+follow[- ]?up|have\s+a\s+chat)\b|\b(?:main|hum)\b.*?\b(?:call\s+karunga|call\s+karenge|sync\s+karenge|connect\s+karenge|baat\s+karenge|baat\s+karunga)\b", re.IGNORECASE),
        "Follow-up Call / Meeting"
    ),
    # Deliverables & Shipments (English + Hinglish)
    (
        re.compile(r"\b(?:i\s*will|i['’]ll|we\s*will|we['’]ll|we\s+have\s+decided\s+to|i\s+can|i\s+commit\s+to)\s+(?:send|share|deploy|ship|release|email|forward|publish|deliver|prepare|provide|update|submit)\b|\b(?:main|hum)\b.*?\b(?:ship\s+kar\s+denge|deploy\s+kar\s+denge|bhej\s+dunga|bhej\s+denge|share\s+kar\s+dunga|share\s+karenge|complete\s+kar\s+lenge|release\s+karenge)\b", re.IGNORECASE),
        "Deliverable / Commitment"
    ),
    # Review & Investigation (English + Hinglish)
    (
        re.compile(r"\b(?:i\s*will|i['’]ll|we\s*will|we['’]ll)\s+(?:review|check|test|audit|investigate|look\s+into|follow\s+up\s+on|verify|debug|fix)\b|\b(?:main|hum)\b.*?\b(?:dekh\s+lunga|review\s+kar\s+lunga|check\s+kar\s+lunga|debug\s+karenge|test\s+karenge)\b", re.IGNORECASE),
        "Review / Investigation"
    ),
    # Delegated action requests (English + Hinglish)
    (
        re.compile(r"\b(?:can\s+you|could\s+you|please|make\s+sure\s+to|kindly)\s+(?:send|share|email|update|review|check|deploy|deliver|fix)\b|\b(?:aap|please|kripya)\b.*?\b(?:bhej\s+dena|share\s+kar\s+dena|update\s+kar\s+dena|dekh\s+lena|review\s+kar\s+lena)\b", re.IGNORECASE),
        "Assigned Request"
    )
]


class ActionItemExtractor:
    """
    On-device heuristic and semantic extractor for commitments and action items.
    """

    @classmethod
    def extract_temporal_anchor(cls, text: str) -> Optional[str]:
        """Extracts dates, days, times, and deadlines from text."""
        for pattern in TIME_PATTERNS:
            match = pattern.search(text)
            if match:
                candidate = match.group(0).strip()
                # Clean up leading 'at', 'on', 'by' if isolated
                cleaned = re.sub(r"^(?:at|on|by)\s+", "", candidate, flags=re.IGNORECASE).strip()
                if len(cleaned) >= 2:
                    return candidate.strip()
        return None

    @classmethod
    def extract_task_description(cls, text: str, matched_intent: str) -> str:
        """Extracts concise task summary from the spoken sentence."""
        cleaned = text.strip().rstrip(".?!")
        # Remove conversational hedging prefixes
        cleaned = re.sub(r"^(?:yeah|yes|okay|ok|sure|understood|alright|so|basically|matlab|actually|definitely)[,\s]+", "", cleaned, flags=re.IGNORECASE)
        # Cap length
        if len(cleaned) > 120:
            cleaned = cleaned[:117] + "..."
        return cleaned

    @classmethod
    def extract_from_utterance(cls, utterance: Utterance) -> List[ActionItem]:
        """Analyzes a single utterance and extracts action items if present."""
        action_items: List[ActionItem] = []
        text = utterance.transcript.strip()
        if not text or len(text) < 8:
            return []

        # Split on sentence boundaries (period, question mark, or semicolons)
        sentences = [s.strip() for s in re.split(r"[.?!;]\s*", text) if len(s.strip()) > 8]
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            detected_category = None
            for pattern, cat_label in INTENT_PATTERNS:
                if pattern.search(sentence):
                    detected_category = cat_label
                    break

            # If an explicit intent was found, or if a strong temporal anchor with action verb exists
            time_anchor = cls.extract_temporal_anchor(sentence)

            if detected_category or (time_anchor and re.search(r"\b(?:will|shall|can|commit|going\s+to|schedule|target)\b", sentence, re.IGNORECASE)):
                category = detected_category or "Task Commitment"
                task_summary = cls.extract_task_description(sentence, category)
                
                # Determine urgency
                urgency = "High" if (time_anchor and any(k in time_anchor.lower() for k in ["today", "tomorrow", "tonight", "asap", "morning"])) else "Normal"

                owner = utterance.speaker
                action_items.append(
                    ActionItem(
                        owner=owner,
                        task=task_summary,
                        due_time_or_date=time_anchor,
                        verbatim_quote=sentence,
                        category=category,
                        urgency=urgency
                    )
                )

        return action_items

    @classmethod
    def extract_from_dialogue(cls, utterances: List[Utterance]) -> List[ActionItem]:
        """Scans all dialogue turns and returns deduplicated action items."""
        all_items: List[ActionItem] = []
        seen_quotes = set()

        for u in utterances:
            items = cls.extract_from_utterance(u)
            for item in items:
                quote_key = item.verbatim_quote.lower().strip()
                if quote_key not in seen_quotes:
                    seen_quotes.add(quote_key)
                    all_items.append(item)

        return all_items
