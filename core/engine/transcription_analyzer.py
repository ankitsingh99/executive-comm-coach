"""
Transcription Analyzer Engine for Executive Communication Coach.
Analyzes transcribed dialogue turns or raw speech text to extract:
1. Key Highlights, Strategic Decisions, and Thematic Takeaways.
2. Potential Tasks, Commitments, and Action Items with Smart Next-Occurrence AM/PM Resolution.
3. Discussion Topics, Overall Summary, and Conversational Sentiment/Tone.
Supports on-device deterministic NLP and optional Gemini AI synthesis.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

try:
    from ..asr_diarization.indic_normalizer import IndicNormalizer
    from ..config import GEMINI_MODEL, get_gemini_api_key
    from .action_item_extractor import ActionItemExtractor
    from .schema import (
        ActionItem,
        ConversationSession,
        KeyHighlight,
        TranscriptionAnalysisResult,
        Utterance,
    )
except (ImportError, ValueError):
    from asr_diarization.indic_normalizer import IndicNormalizer
    from config import GEMINI_MODEL, get_gemini_api_key
    from engine.action_item_extractor import ActionItemExtractor
    from engine.schema import (
        ActionItem,
        ConversationSession,
        KeyHighlight,
        TranscriptionAnalysisResult,
        Utterance,
    )


# Strategic Highlight & Decision Heuristics (English + Hinglish)
HIGHLIGHT_PATTERNS = [
    # Decisions & Consensus
    (
        re.compile(
            r"\b(?:we\s+have\s+decided\s+to|we\s+agreed\s+(?:on|that)|the\s+decision\s+is|final\s+decision|aligned\s+on|consensus\s+is|conclusion\s+is)\b|"
            r"\b(?:faisla\s+ye\s+hai|decide\s+kiya\s+hai|agree\s+kiya\s+hai|final\s+ho\s+gaya)\b",
            re.IGNORECASE,
        ),
        "Decision",
        "High",
    ),
    # Strategic Direction & Priorities
    (
        re.compile(
            r"\b(?:the\s+key\s+priority\s+is|our\s+main\s+focus\s+is|strategic\s+goal|top\s+priority|bottom\s+line\s+is|core\s+objective)\b|"
            r"\b(?:main\s+focus|sabse\s+important\s+baat|priority\s+ye\s+hai|core\s+point)\b",
            re.IGNORECASE,
        ),
        "Strategy",
        "High",
    ),
    # Milestones & Delivery Status
    (
        re.compile(
            r"\b(?:milestone\s+reached|release\s+is\s+ready|completed\s+the|successfully\s+deployed|status\s+update|progress\s+is)\b|"
            r"\b(?:deploy\s+ho\s+gaya|complete\s+ho\s+chuka\s+hai|status\s+ye\s+hai)\b",
            re.IGNORECASE,
        ),
        "Milestone",
        "Normal",
    ),
    # Crucial Insights & Risks
    (
        re.compile(
            r"\b(?:the\s+main\s+risk\s+is|major\s+blocker|crucial\s+takeaway|key\s+learning|important\s+to\s+note|latency\s+issue|performance\s+impact)\b|"
            r"\b(?:sabse\s+bada\s+risk|blocker\s+hai|dhyan\s+rakhna\s+hoga|problem\s+ye\s+hai)\b",
            re.IGNORECASE,
        ),
        "Key Insight",
        "High",
    ),
]


class TranscriptionAnalyzer:
    """
    Intelligent transcription analyzer extracting highlights, tasks, and discussion summaries.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model
        self._gemini_client = None

    def _get_client(self):
        if self._gemini_client is None and self.api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._gemini_client = None
        return self._gemini_client

    def is_gemini_available(self) -> bool:
        return bool(self.api_key and self._get_client() is not None)

    def analyze(
        self,
        input_data: Union[str, List[Utterance], ConversationSession],
        ref_dt: Optional[datetime] = None,
        session_id: str = "transcription_analysis",
        use_gemini: bool = True,
    ) -> TranscriptionAnalysisResult:
        """
        Performs comprehensive transcription analysis to extract key highlights and potential tasks.
        """
        if ref_dt is None:
            ref_dt = datetime.now()

        # Normalize input to list of utterances
        utterances = self._normalize_to_utterances(input_data)
        if not utterances:
            return TranscriptionAnalysisResult(
                session_id=session_id,
                timestamp_utc=ref_dt.isoformat(),
                summary="No audible speech detected to analyze.",
                key_highlights=[],
                potential_tasks=[],
                topics_discussed=[],
                sentiment_tone="Neutral",
            )

        # Try Gemini LLM analysis if enabled and available
        if use_gemini and self.is_gemini_available():
            llm_result = self._analyze_with_gemini(utterances, ref_dt, session_id)
            if llm_result:
                return llm_result

        # Zero-latency on-device NLP heuristic analysis
        return self._analyze_on_device(utterances, ref_dt, session_id)

    def extract_key_highlights(self, utterances: List[Utterance]) -> List[KeyHighlight]:
        """
        Extracts key highlights, strategic takeaways, and decisions from dialogue.
        """
        highlights: List[KeyHighlight] = []
        seen_quotes = set()

        for u in utterances:
            text = IndicNormalizer.normalize_text(u.transcript.strip())
            if not text:
                continue

            sentences = [s.strip() for s in re.split(r"[.?!;]\s*", text) if len(s.strip()) > 10]
            if not sentences:
                sentences = [text]

            for sentence in sentences:
                quote_key = sentence.lower()
                if quote_key in seen_quotes:
                    continue

                matched_category = None
                matched_importance = "Normal"

                for pattern, category, importance in HIGHLIGHT_PATTERNS:
                    if pattern.search(sentence):
                        matched_category = category
                        matched_importance = importance
                        break

                if matched_category:
                    seen_quotes.add(quote_key)
                    headline = self._generate_highlight_headline(sentence, matched_category)
                    takeaway = self._generate_highlight_takeaway(sentence)

                    highlights.append(
                        KeyHighlight(
                            headline=headline,
                            takeaway=takeaway,
                            speaker=u.speaker,
                            verbatim_quote=sentence,
                            category=matched_category,
                            importance=matched_importance,
                        )
                    )

        # If no explicit decision/milestone markers fired, extract top substantive sentences
        if not highlights and utterances:
            substantive_sentences = []
            for u in utterances:
                for s in re.split(r"[.?!;]\s*", u.transcript):
                    s_clean = s.strip()
                    if len(s_clean.split()) >= 4 and not re.match(
                        r"^(?:yes|yeah|ok|okay|hi|hello|sure|theek hai)\b", s_clean, re.IGNORECASE
                    ):
                        substantive_sentences.append((s_clean, u.speaker))

            for s_text, spk in substantive_sentences[:2]:
                highlights.append(
                    KeyHighlight(
                        headline=f"Core Discussion: {s_text[:45]}...",
                        takeaway=s_text,
                        speaker=spk,
                        verbatim_quote=s_text,
                        category="Key Discussion Point",
                        importance="Normal",
                    )
                )

        return highlights

    def extract_potential_tasks(
        self, utterances: List[Utterance], ref_dt: Optional[datetime] = None
    ) -> List[ActionItem]:
        """
        Extracts potential tasks and action items with smart AM/PM next-occurrence temporal resolution.
        """
        return ActionItemExtractor.extract_from_dialogue(utterances, ref_dt=ref_dt)

    def _analyze_on_device(
        self, utterances: List[Utterance], ref_dt: datetime, session_id: str
    ) -> TranscriptionAnalysisResult:
        """
        Fast on-device heuristic analyzer.
        """
        highlights = self.extract_key_highlights(utterances)
        tasks = self.extract_potential_tasks(utterances, ref_dt=ref_dt)

        all_text = " ".join([u.transcript for u in utterances])
        topics = self._extract_topics(all_text)
        summary = self._generate_summary(utterances, highlights, tasks)
        tone = self._detect_sentiment_tone(all_text)

        return TranscriptionAnalysisResult(
            session_id=session_id,
            timestamp_utc=ref_dt.isoformat(),
            summary=summary,
            key_highlights=highlights,
            potential_tasks=tasks,
            topics_discussed=topics,
            sentiment_tone=tone,
        )

    def _analyze_with_gemini(
        self, utterances: List[Utterance], ref_dt: datetime, session_id: str
    ) -> Optional[TranscriptionAnalysisResult]:
        """
        Performs semantic analysis with Gemini LLM.
        """
        client = self._get_client()
        if not client:
            return None

        dialogue_text = "\n".join([f"{u.speaker}: {u.transcript}" for u in utterances])
        current_time_str = ref_dt.strftime("%A, %d %B %Y %I:%M %p")

        prompt = f"""You are an Executive Transcription Intelligence Analyzer.
Analyze the following transcript from a spoken conversation or voice turn.

REFERENCE DATE/TIME: {current_time_str}

TRANSCRIPT:
{dialogue_text}

Task:
1. Extract KEY HIGHLIGHTS: major decisions, critical discussion takeaways, strategic goals, or milestones.
2. Extract POTENTIAL TASKS / ACTION ITEMS: commitments, follow-ups, scheduled calls, deliverables with owner and deadlines.
   IMPORTANT for Time Resolution: If a time like 'at 9' is mentioned without AM/PM, resolve to the NEXT upcoming 9 o'clock relative to reference datetime ({current_time_str}).
3. Extract salient discussion topics/tags.
4. Provide a crisp 2-sentence executive summary and sentiment/tone.

Return pure JSON matching this exact structure:
{{
  "summary": "Crisp 2-sentence executive summary of the conversation.",
  "key_highlights": [
    {{
      "headline": "Punchy highlight title (max 80 chars)",
      "takeaway": "Key takeaway context and implication",
      "speaker": "Speaker name",
      "verbatim_quote": "Exact sentence from transcript",
      "category": "Decision | Strategy | Milestone | Key Insight | Key Discussion Point",
      "importance": "High | Normal"
    }}
  ],
  "potential_tasks": [
    {{
      "owner": "Speaker name (e.g. Rahul, USER)",
      "task": "Concrete summary of committed task",
      "due_time_or_date": "Formatted deadline (e.g. Today at 9:00 PM, Tomorrow at 9:00 AM, Friday EOD)",
      "resolved_datetime": "ISO 8601 timestamp string if resolvable, or null",
      "target_time_inferred_ampm": "AM | PM | null",
      "verbatim_quote": "Exact spoken sentence containing commitment",
      "category": "Follow-up Call / Meeting | Deliverable / Commitment | Review / Investigation | Assigned Request",
      "urgency": "High | Normal"
    }}
  ],
  "topics_discussed": ["Topic1", "Topic2"],
  "sentiment_tone": "Decisive & Collaborative | Calm & Measured | Urgent & Focused"
}}
"""
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)

            highlights = [
                KeyHighlight(
                    headline=h.get("headline", "")[:250],
                    takeaway=h.get("takeaway", "")[:300],
                    speaker=h.get("speaker", "SPEAKER"),
                    verbatim_quote=h.get("verbatim_quote", ""),
                    category=h.get("category", "Key Takeaway"),
                    importance=h.get("importance", "Normal"),
                )
                for h in data.get("key_highlights", [])
            ]

            tasks = [
                ActionItem(
                    owner=t.get("owner", "USER"),
                    task=t.get("task", ""),
                    due_time_or_date=t.get("due_time_or_date"),
                    resolved_datetime=t.get("resolved_datetime"),
                    target_time_inferred_ampm=t.get("target_time_inferred_ampm"),
                    verbatim_quote=t.get("verbatim_quote", ""),
                    category=t.get("category", "Follow-up"),
                    urgency=t.get("urgency", "Normal"),
                )
                for t in data.get("potential_tasks", [])
            ]

            # Merge with deterministic NLP tasks if LLM missed items
            if not tasks:
                tasks = self.extract_potential_tasks(utterances, ref_dt=ref_dt)

            return TranscriptionAnalysisResult(
                session_id=session_id,
                timestamp_utc=ref_dt.isoformat(),
                summary=data.get("summary", ""),
                key_highlights=highlights,
                potential_tasks=tasks,
                topics_discussed=data.get("topics_discussed", []),
                sentiment_tone=data.get("sentiment_tone", "Neutral & Constructive"),
            )
        except Exception:
            return None

    def _normalize_to_utterances(self, input_data: Union[str, List[Utterance], ConversationSession]) -> List[Utterance]:
        """Converts diverse input types into a standardized list of Utterance objects."""
        if isinstance(input_data, ConversationSession):
            return input_data.dialogue
        elif isinstance(input_data, list):
            return [u for u in input_data if isinstance(u, Utterance)]
        elif isinstance(input_data, str):
            text = input_data.strip()
            if not text:
                return []
            # Check for multi-line speaker tags like "Speaker 1: ... \n Speaker 2: ..."
            lines = text.split("\n")
            utterances: List[Utterance] = []
            cur_time = 0.0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^(SPEAKER_\d+|[A-Za-z]+)\s*:\s*(.*)$", line, re.IGNORECASE)
                if match:
                    spk = match.group(1).upper()
                    content = match.group(2).strip()
                    utterances.append(
                        Utterance(speaker=spk, start_time=cur_time, end_time=cur_time + 3.0, transcript=content)
                    )
                    cur_time += 3.5
                else:
                    utterances.append(
                        Utterance(speaker="USER", start_time=cur_time, end_time=cur_time + 3.0, transcript=line)
                    )
                    cur_time += 3.5
            return utterances
        return []

    def _generate_highlight_headline(self, text: str, category: str) -> str:
        cleaned = re.sub(r"^(?:we\s+have|we|the|our|sabse|main|priority)\s+", "", text, flags=re.IGNORECASE).strip()
        words = cleaned.split()[:8]
        short_text = " ".join(words)
        return f"{category}: {short_text}..." if len(cleaned.split()) > 8 else f"{category}: {short_text}"

    def _generate_highlight_takeaway(self, text: str) -> str:
        return text.strip().rstrip(".?!") + "."

    def _extract_topics(self, text: str) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9_-]{4,}\b", text.lower())
        stop_words = {
            "this",
            "that",
            "with",
            "have",
            "from",
            "today",
            "about",
            "what",
            "where",
            "when",
            "could",
            "should",
            "would",
            "just",
            "very",
            "hume",
            "mujhe",
            "karna",
            "hoga",
            "karenge",
            "chahiye",
            "lagta",
            "raha",
            "gaya",
            "wala",
            "vali",
            "bhi",
            "par",
            "aur",
            "lekin",
            "kyunki",
            "isiliye",
            "dekho",
            "there",
            "their",
            "will",
            "going",
            "please",
            "send",
            "call",
            "meet",
            "sync",
            "team",
            "okay",
            "yeah",
            "actually",
            "basically",
        }
        filtered = [w for w in words if w not in stop_words]
        counts: Dict[str, int] = {}
        for w in filtered:
            counts[w] = counts.get(w, 0) + 1
        sorted_topics = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0].capitalize() for t in sorted_topics[:4]]

    def _generate_summary(
        self, utterances: List[Utterance], highlights: List[KeyHighlight], tasks: List[ActionItem]
    ) -> str:
        dialogue_turns = len(utterances)

        if highlights:
            core_hl = highlights[0].takeaway
            task_str = f" with {len(tasks)} actionable follow-up commitments recorded." if tasks else "."
            return f"Dialogue focused on {core_hl.lower()}{task_str}"

        elif tasks:
            return f"Session established {len(tasks)} concrete action items across {dialogue_turns} dialogue turns."
        else:
            return f"Discussion covered key topical points across {dialogue_turns} speech turns."

    def _detect_sentiment_tone(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["blocker", "risk", "issue", "problem", "delay", "tention", "trouble"]):
            return "Urgent & Issue-Focused"
        elif any(
            w in text_lower for w in ["great", "aligned", "agree", "perfect", "good", "congratulations", "shandar"]
        ):
            return "Positive & Collaborative"
        elif any(w in text_lower for w in ["decide", "ship", "deliver", "deploy", "commit", "schedule"]):
            return "Decisive & Action-Oriented"
        return "Calm & Constructive"
