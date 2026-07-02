"""
Multi-language morphology utilities for Turkic (AZ, TR) and Uralic (HU, FI).

Provides:
- Vowel harmony classification per language
- Verb infinitive suffix removal
- Consonant mutation patterns
- Suffix inventories for rule-based baselines

References:
- Göksel, A. & Kerslake, C. (2005). Turkish: A Comprehensive Grammar. Routledge.
- Csató, É. Á. & Johanson, L. (1998). The Turkic Languages. Routledge.
- Karlsson, F. (2008). Finnish: An Essential Grammar. Routledge.
- Rounds, C. (2001). Hungarian: An Essential Grammar. Routledge.
"""
from typing import Dict, List, Set, Tuple

# ─── Vowel harmony classes ────────────────────────────────────────────
# Turkish/Azerbaijani: 8-vowel system with front/back + rounded/unrounded
TURKIC_FRONT_VOWELS = set("eiöüəɪ")  # front (incl. AZ-specific ə)
TURKIC_BACK_VOWELS = set("aıou")
TURKIC_VOWELS = TURKIC_FRONT_VOWELS | TURKIC_BACK_VOWELS

# Hungarian: 14-vowel system with front/back + rounded/unrounded
HUNGARIAN_FRONT_VOWELS = set("eéiíöőüű")
HUNGARIAN_BACK_VOWELS = set("aáoóuú")
HUNGARIAN_VOWELS = HUNGARIAN_FRONT_VOWELS | HUNGARIAN_BACK_VOWELS

# Finnish: 8-vowel system with front/back; e/i are neutral
FINNISH_FRONT_VOWELS = set("äöy")
FINNISH_BACK_VOWELS = set("aou")
FINNISH_NEUTRAL_VOWELS = set("ei")
FINNISH_VOWELS = FINNISH_FRONT_VOWELS | FINNISH_BACK_VOWELS | FINNISH_NEUTRAL_VOWELS


# ─── Verb infinitive suffixes (used to extract verb stems) ────────────
VERB_INFINITIVE_SUFFIXES: Dict[str, List[str]] = {
    "aze": ["mək", "maq"],         # Azerbaijani: -mək (front) / -maq (back)
    "tur": ["mek", "mak"],          # Turkish: -mek (front) / -mak (back)
    "hun": ["ni"],                   # Hungarian: -ni (e.g., látni "to see")
    "fin": ["da", "dä", "ta", "tä"], # Finnish 1st infinitive: -da/-dä/-ta/-tä
}


# ─── Consonant mutations before vowel-initial suffixes ────────────────
# Turkic: voicing of word-final stops before vowel-initial suffix
TURKIC_MUTATIONS = {
    "p": "b",
    "ç": "c",
    "t": "d",
    "k": "ğ",
    "q": "ğ",
}

# Hungarian: lengthening of stem-final vowels before some suffixes
HUNGARIAN_MUTATIONS = {
    "a": "á",
    "e": "é",
}

# Finnish: consonant gradation (kpt → ∅ or weakened)
FINNISH_MUTATIONS = {
    "kk": "k", "pp": "p", "tt": "t",
    "k": "", "p": "v", "t": "d",
}

CONSONANT_MUTATIONS: Dict[str, Dict[str, str]] = {
    "aze": TURKIC_MUTATIONS,
    "tur": TURKIC_MUTATIONS,
    "hun": HUNGARIAN_MUTATIONS,
    "fin": FINNISH_MUTATIONS,
}


# ─── Suffix inventories (for rule-based baselines) ────────────────────

# Azerbaijani — common nominal suffixes ordered by length (longest first)
AZ_SUFFIXES: List[str] = [
    # 4-char
    "ların", "lərin", "lərdə", "lardan", "lərdən",
    "dakı", "dəki", "nızın", "lərimizin",
    # 3-char
    "lar", "lər", "ları", "ləri", "lara", "lərə",
    "lardan", "lərdən", "ların", "lərin",
    "nın", "nin", "nun", "nün",
    "ya", "yə", "ya", "yə",
    "dır", "dir", "dur", "dür",
    "sın", "sin", "sun", "sün",
    # 2-char
    "da", "də", "dan", "dən",
    "ın", "in", "un", "ün",
    "ım", "im", "um", "üm",
    "lı", "li", "lu", "lü",
    "lə", "la",
    # 1-char
    "a", "ə", "ı", "i", "u", "ü",
]

# Turkish — common nominal suffixes
TR_SUFFIXES: List[str] = [
    "lerinin", "larının", "ların", "lerin",
    "lardan", "lerden", "larda", "lerde",
    "ların", "lerin", "lara", "lere",
    "ları", "leri", "larım", "lerim",
    "daki", "deki", "taki", "teki",
    "lar", "ler",
    "nın", "nin", "nun", "nün",
    "dır", "dir", "dur", "dür",
    "tır", "tir", "tur", "tür",
    "da", "de", "ta", "te",
    "dan", "den", "tan", "ten",
    "ın", "in", "un", "ün",
    "yı", "yi", "yu", "yü",
    "ya", "ye",
    "lı", "li", "lu", "lü",
    "sı", "si", "su", "sü",
    "a", "e", "ı", "i", "u", "ü",
]

# Hungarian — common nominal suffixes (case + possessive + plural)
HU_SUFFIXES: List[str] = [
    "ainknak", "einknek",
    "aitoknak", "eiteknek",
    "jaiknak", "jeiknek",
    "imban", "imben",
    "unkban", "ünkben",
    "ainkat", "einket",
    # case+plural
    "okban", "ekben", "akban", "öknek",
    "ainak", "einek", "ainket",
    "ban", "ben",       # inessive (in)
    "ból", "ből",       # elative (out of)
    "ba",  "be",        # illative (into)
    "ra",  "re",        # sublative (onto)
    "ról", "ről",       # delative (off of)
    "tól", "től",       # ablative (from)
    "hoz", "hez", "höz", # allative (to)
    "nál", "nél",       # adessive (at)
    "val", "vel",       # instrumental (with)
    "ért",              # causal-final (for)
    "kor",              # temporal (at the time of)
    "ig",               # terminative (until)
    "ul",  "ül",        # essive-modal
    "nak", "nek",       # dative
    "at",  "et", "ot", "öt",  # accusative
    "ok",  "ek", "ak", "ök",  # plural
    "om",  "em", "am",        # 1sg possessive
    "od",  "ed", "ad",        # 2sg possessive
    "uk",  "ük",              # 3pl possessive
    "ja",  "je",              # 3sg possessive
    "k",                        # plural (after vowel)
    "n",                        # superessive (on)
    "t",                        # accusative (after vowel)
]

# Finnish — common nominal case suffixes
FI_SUFFIXES: List[str] = [
    "issansa", "issänsä",
    "issaan", "issään",
    "iltaan", "iltään",
    "ihinsä", "ihinsa",
    "lleen", "lleni", "llesi",
    "ssa",  "ssä",      # inessive (in)
    "sta",  "stä",      # elative (out of)
    "han",  "hen", "hon", "hön",  # illative
    "lla",  "llä",      # adessive (at/on)
    "lta",  "ltä",      # ablative (from)
    "lle",              # allative (to)
    "ksi",              # translative (becoming)
    "tta",  "ttä",      # abessive (without)
    "ineen",            # comitative
    "an",   "än", "en", "on", "in", "un", "yn", "ön",  # illative (short)
    "na",   "nä",       # essive
    "in",               # genitive plural / instructive
    "ja",   "jä",       # plural partitive
    "ni",   "si",       # 1sg/2sg possessive
    "nsa",  "nsä",      # 3sg/3pl possessive
    "mme",  "nne",      # 1pl/2pl possessive
    "n",                # genitive
    "t",                # plural nominative
    "a",   "ä",         # partitive
]


SUFFIX_INVENTORIES: Dict[str, List[str]] = {
    "aze": sorted(set(AZ_SUFFIXES), key=lambda s: -len(s)),
    "tur": sorted(set(TR_SUFFIXES), key=lambda s: -len(s)),
    "hun": sorted(set(HU_SUFFIXES), key=lambda s: -len(s)),
    "fin": sorted(set(FI_SUFFIXES), key=lambda s: -len(s)),
}


# ─── Distance metrics ────────────────────────────────────────────────

def suffix_overlap(lang_a: str, lang_b: str) -> float:
    """Jaccard similarity of suffix inventories between two languages."""
    sa = set(SUFFIX_INVENTORIES[lang_a])
    sb = set(SUFFIX_INVENTORIES[lang_b])
    if not (sa | sb):
        return 0.0
    return len(sa & sb) / len(sa | sb)


def vowel_overlap(lang_a: str, lang_b: str) -> float:
    """Jaccard similarity of vowel inventories."""
    inventories = {
        "aze": TURKIC_VOWELS, "tur": TURKIC_VOWELS,
        "hun": HUNGARIAN_VOWELS, "fin": FINNISH_VOWELS,
    }
    va = inventories[lang_a]
    vb = inventories[lang_b]
    if not (va | vb):
        return 0.0
    return len(va & vb) / len(va | vb)


def language_family(lang: str) -> str:
    """Returns the language family for a language code."""
    return {
        "aze": "Turkic", "tur": "Turkic",
        "hun": "Uralic", "fin": "Uralic",
    }.get(lang, "Unknown")


def is_intra_family(lang_a: str, lang_b: str) -> bool:
    """True if both languages are in the same family."""
    return language_family(lang_a) == language_family(lang_b)


if __name__ == "__main__":
    # Sanity check
    languages = ["aze", "tur", "hun", "fin"]
    print(f"{'pair':<12} {'family':<6} {'sfx_overlap':<12} {'vowel_overlap'}")
    print("-" * 50)
    for a in languages:
        for b in languages:
            if a < b:
                tag = "intra" if is_intra_family(a, b) else "cross"
                print(f"{a}-{b:<8} {tag:<6} {suffix_overlap(a, b):<12.4f} {vowel_overlap(a, b):.4f}")
