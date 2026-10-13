"""Shared helpers for the L2-ARCTIC phone-recognition study.

Everything here is dependency-free (standard library only) so that it can be
used both by the inventory analysis (runs anywhere) and by the full PFER
experiment (which additionally needs torch / transformers / panphon).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# Speakers -> L1 (from the L2-ARCTIC v5.0 README)
# --------------------------------------------------------------------------
SPEAKER_L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}
L1_ORDER = ["Arabic", "Mandarin", "Hindi", "Korean", "Spanish", "Vietnamese"]

# ISO 639-3 codes used to pull each L1's inventory from PHOIBLE.
# Arabic: the L2-ARCTIC speakers' home varieties are not documented, so the
# primary analysis uses Standard Arabic (arb); a sensitivity analysis adds the
# Egyptian, Moroccan and Najdi inventories that PHOIBLE contains.
L1_ISO = {
    "Arabic": ["arb"], "Mandarin": ["cmn"], "Hindi": ["hin"],
    "Korean": ["kor"], "Spanish": ["spa"], "Vietnamese": ["vie"],
}
ARABIC_ALL_ISO = ["arb", "arz", "ary", "ars"]

# --------------------------------------------------------------------------
# ARPAbet -> IPA.  Stress-aware: the 2024 code stripped stress digits before
# mapping, so every unstressed AH0 ("the", "a") became /ʌ/ instead of /ə/.
# --------------------------------------------------------------------------
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
UNSTRESSED = {"AH": "ə", "ER": "ɚ"}
ARPABET_EXTRA = {"AX": "ə", "AXR": "ɚ"}   # used by some L2-ARCTIC annotators
SILENCE = {"sil", "sp", "spn", "", "SIL", "SP", "SPN"}


def arpabet_to_ipa(label: str, stress_aware: bool = True) -> str | None:
    """Map one ARPAbet label (e.g. 'AH0', 'T') to an IPA segment.

    Returns None for silences / unknown labels.
    """
    label = label.strip()
    if label in SILENCE:
        return None
    # annotator noise seen in L2-ARCTIC: 'ER)', 'D_', 'W`', 'AA*1', 'AH!'
    label = re.sub(r"[^A-Za-z0-9]", "", label)
    base = re.sub(r"\d", "", label).upper()
    stress = re.sub(r"\D", "", label)
    if base in ARPABET_EXTRA:
        return ARPABET_EXTRA[base]
    if stress_aware and stress == "0" and base in UNSTRESSED:
        return UNSTRESSED[base]
    return ARPABET_TO_IPA.get(base)


# The 39 English target phones the corpus transcripts can contain (+ ə, ɚ).
ENGLISH_TARGETS = sorted(set(ARPABET_TO_IPA.values()))
ENGLISH_TARGETS_STRESS = sorted(set(ARPABET_TO_IPA.values()) | set(UNSTRESSED.values()))

# --------------------------------------------------------------------------
# IPA normalisation and segmentation
# --------------------------------------------------------------------------
_REPLACEMENTS = [
    ("͡", ""), ("͜", ""),  # tie bars  t͡ʃ -> tʃ
    ("g", "ɡ"),                        # ASCII g -> IPA script g (U+0261)
    ("t̠ʃ", "tʃ"), ("d̠ʒ", "dʒ"),       # PHOIBLE marks postalveolar affricates
    ("ʧ", "tʃ"), ("ʤ", "dʒ"),
    ("'", ""), ("ˈ", ""), ("ˌ", ""), (".", ""), (" ", ""),
]

# Diacritics that do not change the phoneme identity for our coarse "loose"
# matching (length, aspiration, dental/advanced/retracted, non-syllabic, ...)
LOOSE_STRIP = "ːˑʰ̪̟̠̞̝̯̙̘̃ʷˤˠʲ̤̥̬̹̜̆̚"


def normalize_ipa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    for a, b in _REPLACEMENTS:
        s = s.replace(a, b)
    return unicodedata.normalize("NFC", s)


def loose(seg: str) -> str:
    seg = unicodedata.normalize("NFD", normalize_ipa(seg))
    seg = "".join(ch for ch in seg if ch not in LOOSE_STRIP)
    return unicodedata.normalize("NFC", seg)


DIPHTHONGS = ["aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"]
UNTIED_AFFRICATES = ["tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "tʂ", "ɖʐ"]
_MULTI = sorted(DIPHTHONGS + UNTIED_AFFRICATES, key=len, reverse=True)
_MODIFIERS = set("ːˑʰʷʲˠˤ̩̯̪̟̠̞̝̙̘̤̥̬̹̜̃̆̚")


def segment(ipa: str, merge_untied_affricates: bool = False) -> list[str]:
    """Split an IPA string into segments.

    * Combining diacritics / modifier letters attach to the previous segment.
    * A tie bar joins its two neighbours (t͡s -> one affricate).
    * English diphthongs (aɪ, aʊ, eɪ, oʊ, ɔɪ) are kept whole, so they can be
      aligned against the single diphthong symbol in the reference.
    * Untied pairs such as 'ts' are two segments (t + s) unless
      merge_untied_affricates=True, which is only needed for the 2024 unspaced
      reference strings where tʃ/dʒ were written without tie bars.

    Replaces lingpy.ipa2tokens on unspaced strings, which merged adjacent
    vowels across word boundaries into pseudo-diphthongs (e.g. 'ʌoʊ').
    """
    s = unicodedata.normalize("NFD", str(ipa))
    for a, b in _REPLACEMENTS:
        if a in ("\u0361", "\u035c"):
            continue                          # keep tie bars for now
        s = s.replace(a, b)
    multi = _MULTI if merge_untied_affricates else sorted(DIPHTHONGS, key=len, reverse=True)
    out: list[str] = []
    i, join_next = 0, False
    while i < len(s):
        ch = s[i]
        if ch in ("\u0361", "\u035c"):
            join_next = True
            i += 1
            continue
        if join_next and out:
            out[-1] += ch
            join_next = False
            i += 1
            continue
        if ch in _MODIFIERS and out:
            out[-1] += ch
            i += 1
            continue
        for m in multi:
            if s.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            out.append(ch)
            i += 1
    return [unicodedata.normalize("NFC", x) for x in out]


# --------------------------------------------------------------------------
# Minimal Praat TextGrid reader (long text format, as shipped in L2-ARCTIC)
# --------------------------------------------------------------------------
def read_textgrid(path: str | Path) -> dict[str, list[tuple[float, float, str]]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    current = None
    xmin = xmax = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("name ="):
            current = line.split("=", 1)[1].strip().strip('"')
            tiers[current] = []
        elif line.startswith("xmin =") and current is not None:
            xmin = float(line.split("=")[1])
        elif line.startswith("xmax =") and current is not None:
            xmax = float(line.split("=")[1])
        elif line.startswith("text =") and current is not None:
            label = line.split("=", 1)[1].strip().strip('"')
            tiers[current].append((xmin, xmax, label))
    return tiers


def canonical_ipa_from_textgrid(path, stress_aware=True) -> list[str]:
    tiers = read_textgrid(path)
    phones = tiers.get("phones") or tiers.get("phone") or []
    out = []
    for _, _, lab in phones:
        # annotation TextGrids use "CPL,PPL,type"; take the canonical phone
        lab = lab.split(",")[0].strip()
        p = arpabet_to_ipa(lab, stress_aware)
        if p:
            out.append(p)
    return out


def perceived_ipa_from_annotation(path, stress_aware=True) -> list[str]:
    """For the 3,599 manually annotated utterances: what the speaker
    actually said (perceived phone; deletions dropped, insertions kept)."""
    tiers = read_textgrid(path)
    phones = tiers.get("phones") or []
    out = []
    for _, _, lab in phones:
        parts = [p.strip() for p in lab.split(",")]
        perceived = parts[1] if len(parts) >= 3 else parts[0]
        if perceived.lower() in {"err", "deletion"} or (len(parts) >= 3 and parts[2] == "d"):
            continue
        perceived = perceived.rstrip("*")
        p = arpabet_to_ipa(perceived, stress_aware)
        if p:
            out.append(p)
    return out
