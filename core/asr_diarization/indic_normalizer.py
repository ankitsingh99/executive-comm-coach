"""
Indic and Hinglish Script Normalizer & Transliteration Utility.
Converts Devanagari script to standardized Romanized Hinglish, normalizes
phonetic variations, and ensures consistent downstream NLP/regex extraction.
"""

import re
from typing import Dict

# Comprehensive Devanagari to Romanized IAST / Popular Phonetic Mapping
DEVANAGARI_VOWELS: Dict[str, str] = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "अं": "an",
    "अः": "ah",
    "ऑ": "o",
    "ऍ": "e",
}

DEVANAGARI_MATRAS: Dict[str, str] = {
    "ा": "aa",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "oo",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ं": "n",
    "ँ": "n",
    "ः": "h",
    "्": "",
    "ॉ": "o",
    "ॅ": "e",
}

DEVANAGARI_CONSONANTS: Dict[str, str] = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "क़": "q",
    "ख़": "kh",
    "ग़": "gh",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
    "क्ष": "ksh",
    "त्र": "tr",
    "ज्ञ": "gya",
    "श्र": "shr",
}

# High-frequency Devanagari words to idiomatic Romanized Hinglish dictionary
DEVANAGARI_LEXICON: Dict[str, str] = {
    "नमस्ते": "namaste",
    "धन्यवाद": "dhanyavaad",
    "हाँ": "haan",
    "नहीं": "nahi",
    "मतलब": "matlab",
    "यानी": "yaani",
    "हैना": "haina",
    "अच्छा": "accha",
    "अरे": "arre",
    "भाई": "bhai",
    "यार": "yaar",
    "शायद": "shayad",
    "मुझे": "mujhe",
    "हमे": "hume",
    "हमें": "hume",
    "लगता": "lagta",
    "है": "hai",
    "कल": "kal",
    "आज": "aaj",
    "परसों": "parso",
    "सुबह": "subah",
    "शाम": "shaam",
    "दोपहर": "dopahar",
    "रात": "raat",
    "बजे": "baje",
    "करना": "karna",
    "करेंगे": "karenge",
    "करूंगा": "karunga",
    "करूँगा": "karunga",
    "चाहिए": "chahiye",
    "होगा": "hoga",
    "सकता": "sakta",
    "सकते": "sakte",
    "सकती": "sakti",
    "ठीक": "theek",
    "फैसला": "faisla",
    "पक्का": "pakka",
    "बिल्कुल": "bilkul",
    "बात": "baat",
    "बताओ": "batao",
    "सुनो": "suno",
    "देखो": "dekho",
    "सिंक": "sync",
    "कॉल": "call",
    "मीटिंग": "meeting",
    "अपडेट": "update",
    "शेयर": "share",
    "भेज": "bhej",
    "देंगे": "denge",
    "दूंगा": "dunga",
    "रिलीज़": "release",
    "डिप्लॉय": "deploy",
}

# Common phonetic variations in informal Romanized Hinglish chat/transcripts
HINGLISH_PHONETIC_HARMONIZER = [
    (r"\bmatlb\b", "matlab"),
    (r"\bmtlb\b", "matlab"),
    (r"\byaani\s+ki\b", "yaani ki"),
    (r"\byani\b", "yaani"),
    (r"\bhainaa?\b", "haina"),
    (r"\bhai\s+na\b", "haina"),
    (r"\bshyd\b", "shayad"),
    (r"\bchahye\b", "chahiye"),
    (r"\bchaahiye\b", "chahiye"),
    (r"\bkl\b", "kal"),
    (r"\bkall\b", "kal"),
    (r"\bparson\b", "parso"),
    (r"\btarson\b", "tarso"),
    (r"\bbajeh\b", "baje"),
    (r"\bthk\b", "theek"),
    (r"\bthik\b", "theek"),
    (r"\bachha\b", "accha"),
    (r"\bacha\b", "accha"),
    (r"\bplz\b", "please"),
    (r"\bpls\b", "please"),
    (r"\bkrna\b", "karna"),
    (r"\bkr\b", "kar"),
    (r"\bkrnge\b", "karenge"),
    (r"\bhogaa?\b", "hoga"),
]


class IndicNormalizer:
    """
    Normalizes Indic & Hinglish script, performs phonetic harmonization,
    and converts Devanagari text to Romanized code-mixed representation.
    """

    @classmethod
    def contains_devanagari(cls, text: str) -> bool:
        """Checks if text contains Devanagari Unicode characters (U+0900 to U+097F)."""
        return any("\u0900" <= char <= "\u097f" for char in text)

    @classmethod
    def transliterate_devanagari_to_roman(cls, text: str) -> str:
        """
        Converts Devanagari words to phonetic Romanized Hinglish.
        Uses dictionary lookup for common words, followed by character-level transliteration.
        """
        if not cls.contains_devanagari(text):
            return text

        words = text.split()
        converted_words = []

        for word in words:
            # Strip punctuation for lookup
            clean_word = re.sub(r"[^\u0900-\u097F]", "", word)
            punct_before = re.match(r"^[^\u0900-\u097Fa-zA-Z0-9]*", word).group(0)
            punct_after = re.search(r"[^\u0900-\u097Fa-zA-Z0-9]*$", word).group(0)

            if clean_word in DEVANAGARI_LEXICON:
                roman_word = DEVANAGARI_LEXICON[clean_word]
                converted_words.append(f"{punct_before}{roman_word}{punct_after}")
            elif cls.contains_devanagari(word):
                # Character-level phonetic conversion
                res = []
                chars = list(word)
                i = 0
                while i < len(chars):
                    ch = chars[i]
                    if ch in DEVANAGARI_VOWELS:
                        res.append(DEVANAGARI_VOWELS[ch])
                    elif ch in DEVANAGARI_CONSONANTS:
                        base = DEVANAGARI_CONSONANTS[ch]
                        # Look ahead for matra or halant
                        if i + 1 < len(chars) and chars[i + 1] in DEVANAGARI_MATRAS:
                            matra = chars[i + 1]
                            res.append(base + DEVANAGARI_MATRAS[matra])
                            i += 1
                        elif i + 1 < len(chars) and chars[i + 1] == "्":  # Halant
                            res.append(base)
                            i += 1
                        else:
                            # Inherited schwa 'a' if not at the very end of word
                            res.append(
                                base + ("a" if i + 1 < len(chars) and chars[i + 1] in DEVANAGARI_CONSONANTS else "")
                            )
                    elif ch in DEVANAGARI_MATRAS:
                        res.append(DEVANAGARI_MATRAS[ch])
                    else:
                        res.append(ch)
                    i += 1
                converted_words.append("".join(res))
            else:
                converted_words.append(word)

        return " ".join(converted_words)

    @classmethod
    def harmonize_hinglish_phonetics(cls, text: str) -> str:
        """Standardizes chat slang and phonetic spelling variations in Romanized Hinglish."""
        result = text
        for pattern, replacement in HINGLISH_PHONETIC_HARMONIZER:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """
        Master normalization pipeline:
        1. Transliterates Devanagari to Romanized text.
        2. Harmonizes phonetic spelling variants.
        3. Cleans whitespace while preserving sentence punctuation.
        """
        if not text:
            return ""

        # Step 1: Transliterate Devanagari if present
        if cls.contains_devanagari(text):
            text = cls.transliterate_devanagari_to_roman(text)

        # Step 2: Harmonize Hinglish phonetic spelling
        text = cls.harmonize_hinglish_phonetics(text)

        # Step 3: Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
