"""Shared paths, normalization, and ground-truth loading for discovery scripts.

Read-only with respect to the raw dataset under dataset/.
All derived artifacts are written to work/ (gitignored via *.parquet).
"""
import os
import re
import unicodedata

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
WORK_DIR = os.path.join(REPO_ROOT, "work")
os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

TRAIN_RECORDS = os.path.join(WORK_DIR, "train_records.parquet")
TEST_RECORDS = os.path.join(WORK_DIR, "test_records.parquet")
GT_LONG = os.path.join(WORK_DIR, "gt_long.parquet")


CORP_SUFFIXES = {
    r"\binc\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bllc\b": "llc",
    r"\bl\.l\.c\.\b": "llc",
    r"\bltd\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bpvt\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bcorp\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\b": "co",
    r"\bcompany\b": "co",
    r"\bsarl\b": "sarl",
    r"\bs\.a\.r\.l\.\b": "sarl",
    r"\bsas\b": "sas",
    r"\bs\.a\.s\.\b": "sas",
}

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
    r"\ballee\b": "all",
    r"\brue\b": "r",
    r"\bplace\b": "pl",
}


def fold_accent(s):
    """NFKD + strip combining marks (skip fast-path when pure ASCII)."""
    if not s:
        return s
    if s.isascii():
        return s
    folded = unicodedata.normalize("NFKD", s)
    return "".join(c for c in folded if not unicodedata.combining(c))


def normalize_core(s):
    """Lowercase, deaccent, punctuation -> space, collapse whitespace."""
    if not s:
        return ""
    t = fold_accent(s).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def apply_maps(text, mapping):
    out = text
    for pat, repl in mapping.items():
        out = re.sub(pat, repl, out)
    return re.sub(r"\s+", " ", out).strip()


def normalize_name(name, with_suffix_map=False):
    t = normalize_core(name)
    if with_suffix_map:
        t = apply_maps(t, CORP_SUFFIXES)
    return t


def normalize_address(addr, with_abbr_map=True):
    if not addr:
        return ""
    t = normalize_core(addr)
    if with_abbr_map:
        t = apply_maps(t, ADDRESS_ABBRS)
    return t


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(norm_text):
    return norm_text.split()


def numeric_tokens(raw_addr):
    """Sorted, de-duplicated list of lowercased tokens that contain digits."""
    if not raw_addr:
        return []
    out = []
    for tok in raw_addr.lower().split():
        if any(ch.isdigit() for ch in tok):
            out.append(tok)
    return sorted(set(out))