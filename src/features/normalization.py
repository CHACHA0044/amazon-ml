"""Text preprocessing and normalization module for Entity Resolution.

Handles:
- Unicode NFKD normalization & diacritics stripping (vital for French, multilingual, noisy OCR text)
- Lowercasing, punctuation normalization, whitespace collapsing
- Common entity prefix/suffix standardization (Inc, LLC, Corp, Pvt Ltd, etc.)
- Street/address term standardization (Rd, St, Ave, Rue, Blvd, etc.)
"""

import re
import unicodedata

# Address abbreviation mappings (US, India, France)
ADDRESS_ABBRS = {
    r"\bstreet\b": "st",
    r"\broad\b": "rd",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bapartment\b": "apt",
    r"\bsuite\b": "ste",
    r"\bhighway\b": "hwy",
    r"\bnagar\b": "ngr",
    r"\bmarg\b": "mrg",
    r"\ball[eé]e\b": "all",
    r"\brue\b": "r",
    r"\bplace\b": "pl",
}

# Legal entity suffixes to normalize
CORP_SUFFIXES = {
    r"\binc\b|\bincorporated\b": "inc",
    r"\bllc\b|\bl\.l\.c\.\b": "llc",
    r"\bltd\b|\blimited\b": "ltd",
    r"\bpvt\b|\bprivate\b": "pvt",
    r"\bcorp\b|\bcorporation\b": "corp",
    r"\bco\b|\bcompany\b": "co",
    r"\bsarl\b|\bs\.a\.r\.l\.\b": "sarl",
    r"\bsas\b|\bs\.a\.s\.\b": "sas",
}


def normalize_text(text: str, remove_accents: bool = True) -> str:
    """Normalize general text string.
    
    1. Unicode decomposition (NFKD)
    2. Optional ASCII transliteration / accent removal
    3. Lowercase
    4. Replace punctuation with single spaces
    5. Collapse consecutive whitespace
    """
    if not text:
        return ""
    
    if remove_accents:
        # Decompose unicode characters into base char + combining diacritic, then filter out diacritics
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    else:
        text = unicodedata.normalize("NFC", text)
        
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Normalize business name with legal entity standardization."""
    text = normalize_text(name, remove_accents=True)
    for pattern, repl in CORP_SUFFIXES.items():
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(address: str) -> str:
    """Normalize business address with street/unit abbreviation standardization."""
    if not address:
        return ""
    text = normalize_text(address, remove_accents=True)
    for pattern, repl in ADDRESS_ABBRS.items():
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()
