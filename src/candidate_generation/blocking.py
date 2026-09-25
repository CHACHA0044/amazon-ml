"""Multi-Channel Candidate Generation / Blocking Module for Entity Resolution.

Query-filtered, high-speed multi-channel inverted index partitioned by country.
  - Channel 1 (name_exact): Country + Normalized Name block
  - Channel 2 (name_suf_exact): Country + Suffix Normalized Name block
  - Channel 3 (address_exact): Country + Normalized Address block (excluding empties)
  - Channel 4 (rare_name_token): Country + Salient distinctive name token (len >= 3)
  - Channel 5 (numeric): Country + Specific address numeric token (house#/pincode)
  - Channel 6 (addr_token): Country + Informative address token (locality/street name, len >= 4)

Bitmask tracking preserves full channel provenance per candidate pair.
"""

import collections
import gc
import time
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

# Generic stop words and corporate suffixes to exclude from token indexing
NAME_STOPWORDS = {
    "ltd", "pvt", "inc", "llc", "corp", "co", "limited", "private", "corporation",
    "company", "services", "solutions", "technologies", "technology", "enterprises",
    "associates", "group", "partners", "international", "global", "trading", "brothers",
    "ventures", "industries", "consultancy", "consultants", "care", "and", "the", "of",
    "in", "for", "center", "centre", "hub", "studio", "sarl", "sas", "llp", "pllc"
}

ADDR_STOPWORDS = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "apartment", "suite",
    "highway", "nagar", "marg", "allee", "rue", "place", "st", "rd", "ave", "blvd",
    "dr", "ln", "apt", "ste", "hwy", "ngr", "mrg", "all", "r", "pl", "floor", "unit",
    "block", "near", "opp", "opposite", "behind", "dist", "district", "city", "state",
    "india", "usa", "france", "north", "south", "east", "west"
}

# Bitmask flags for channel provenance
CH_NAME_EXACT = 1 << 0      # 1
CH_NAME_SUF_EXACT = 1 << 1  # 2
CH_RARE_NAME_TOK = 1 << 2   # 4
CH_NUMERIC = 1 << 3         # 8
CH_ADDR_EXACT = 1 << 4      # 16
CH_ADDR_TOK = 1 << 5        # 32


class MultiChannelIndex:
    """Query-filtered multi-channel inverted index partitioned by country."""

    def __init__(
        self,
        max_name_block_size: int = 150,
        max_rare_token_df: int = 250,
        max_numeric_df: int = 120,
        max_addr_block_size: int = 50,
        max_addr_tok_df: int = 100,
    ):
        self.max_name_block_size = max_name_block_size
        self.max_rare_token_df = max_rare_token_df
        self.max_numeric_df = max_numeric_df
        self.max_addr_block_size = max_addr_block_size
        self.max_addr_tok_df = max_addr_tok_df

        # Inverted indices partitioned by country: country -> {key -> list[cand_idx]}
        self.name_exact_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
        self.name_suf_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
        self.token_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
        self.numeric_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
        self.addr_exact_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
        self.addr_tok_idx: Dict[str, Dict[str, List[int]]] = collections.defaultdict(dict)

        # Candidate record attribute arrays
        self.cand_ids: List[str] = []
        self.cand_countries: List[str] = []
        self.cand_name_norms: List[str] = []
        self.cand_name_sufs: List[str] = []
        self.cand_addr_norms: List[str] = []
        self.cand_nums: List[List[str]] = []
        self.cand_addr_missing: List[int] = []

    def build_index_for_queries(
        self,
        candidate_df: pl.DataFrame,
        query_records: Dict[str, Dict],
    ):
        """Build multi-channel index indexed against active query vocabulary."""
        t0 = time.time()
        n_rows = candidate_df.height
        print(f"Building query-filtered index from {n_rows:,} candidate records...", flush=True)

        # Extract target query vocabularies
        q_names = set(q["name_norm"] for q in query_records.values() if q["name_norm"])
        q_sufs = set(q["name_suf"] for q in query_records.values() if q["name_suf"])
        q_addrs = set(q["addr_norm"] for q in query_records.values() if q["addr_norm"])
        q_toks = set(
            t for q in query_records.values() for t in q["name_toks"]
            if len(t) >= 3 and t not in NAME_STOPWORDS
        )
        q_addr_toks = set(
            t for q in query_records.values() for t in q["addr_toks"]
            if len(t) >= 4 and t not in ADDR_STOPWORDS and not any(ch.isdigit() for ch in t)
        )
        q_nums = set(
            num for q in query_records.values() for num in q["nums"]
            if len(num) >= 2 and any(ch.isdigit() for ch in num)
        )

        print(f"Query vocabulary: {len(q_names):,} names, {len(q_addrs):,} addrs, {len(q_toks):,} name_toks, {len(q_addr_toks):,} addr_toks, {len(q_nums):,} nums.", flush=True)

        # Extract candidate vectors
        self.cand_ids = candidate_df["id"].to_list()
        self.cand_countries = candidate_df["country"].to_list()
        self.cand_name_norms = candidate_df["name_norm"].fill_null("").to_list()
        self.cand_name_sufs = candidate_df["name_norm_suf"].fill_null("").to_list()
        self.cand_addr_norms = candidate_df["addr_norm"].fill_null("").to_list()
        self.cand_addr_missing = candidate_df["addr_missing"].fill_null(0).to_list()

        raw_addr_nums = candidate_df["addr_num"].fill_null("").to_list()
        self.cand_nums = [[t for t in s.split() if t] for s in raw_addr_nums]

        del candidate_df
        gc.collect()

        # Single fast pass populating inverted maps for query tokens
        t1 = time.time()
        for i in range(n_rows):
            country = self.cand_countries[i]
            n_norm = self.cand_name_norms[i]
            n_suf = self.cand_name_sufs[i]
            a_norm = self.cand_addr_norms[i]
            a_miss = self.cand_addr_missing[i]
            nums = self.cand_nums[i]

            # 1. Exact Name
            if n_norm in q_names:
                if n_norm not in self.name_exact_idx[country]:
                    self.name_exact_idx[country][n_norm] = [i]
                else:
                    self.name_exact_idx[country][n_norm].append(i)

            # 2. Suffix Name
            if n_suf in q_sufs and n_suf != n_norm:
                if n_suf not in self.name_suf_idx[country]:
                    self.name_suf_idx[country][n_suf] = [i]
                else:
                    self.name_suf_idx[country][n_suf].append(i)

            # 3. Exact Address (excluding empties)
            if a_norm in q_addrs and a_miss == 0 and len(a_norm) >= 5:
                if a_norm not in self.addr_exact_idx[country]:
                    self.addr_exact_idx[country][a_norm] = [i]
                else:
                    self.addr_exact_idx[country][a_norm].append(i)

            # 4. Query Name Tokens
            if n_norm:
                for tok in set(n_norm.split()) & q_toks:
                    if tok not in self.token_idx[country]:
                        self.token_idx[country][tok] = [i]
                    else:
                        self.token_idx[country][tok].append(i)

            # 5. Query Address Tokens (Locality/Street)
            if a_norm and a_miss == 0:
                for tok in set(a_norm.split()) & q_addr_toks:
                    if tok not in self.addr_tok_idx[country]:
                        self.addr_tok_idx[country][tok] = [i]
                    else:
                        self.addr_tok_idx[country][tok].append(i)

            # 6. Query Numerics
            for num in set(nums) & q_nums:
                if num not in self.numeric_idx[country]:
                    self.numeric_idx[country][num] = [i]
                else:
                    self.numeric_idx[country][num].append(i)

        print(f"Pass complete in {time.time()-t1:.1f}s. Pruning oversized blocks...", flush=True)
        self._prune_blocks()
        print(f"Query-filtered index built in {time.time()-t0:.1f}s total.", flush=True)

    def _prune_blocks(self):
        """Cap oversized blocks to prevent combinatorial explosions."""
        pruned_name = 0
        for country, name_dict in self.name_exact_idx.items():
            for k in list(name_dict.keys()):
                if len(name_dict[k]) > self.max_name_block_size:
                    del name_dict[k]
                    pruned_name += 1

        for country, suf_dict in self.name_suf_idx.items():
            for k in list(suf_dict.keys()):
                if len(suf_dict[k]) > self.max_name_block_size:
                    del suf_dict[k]

        for country, addr_dict in self.addr_exact_idx.items():
            for k in list(addr_dict.keys()):
                if len(addr_dict[k]) > self.max_addr_block_size:
                    del addr_dict[k]

        for country, tok_dict in self.token_idx.items():
            for k in list(tok_dict.keys()):
                if len(tok_dict[k]) > self.max_rare_token_df:
                    del tok_dict[k]

        for country, addr_tok_dict in self.addr_tok_idx.items():
            for k in list(addr_tok_dict.keys()):
                if len(addr_tok_dict[k]) > self.max_addr_tok_df:
                    del addr_tok_dict[k]

        for country, num_dict in self.numeric_idx.items():
            for k in list(num_dict.keys()):
                if len(num_dict[k]) > self.max_numeric_df:
                    del num_dict[k]

        print(f"Pruned oversized blocks: {pruned_name:,} name blocks pruned.", flush=True)

    def generate_candidates_for_query(
        self,
        country: str,
        name_norm: str,
        name_suf: str,
        addr_norm: str,
        nums: List[str],
        max_candidates_per_query: int = 150,
    ) -> Dict[int, int]:
        """Generate candidate indices with channel provenance bitmask for a single query."""
        cand_masks: Dict[int, int] = collections.defaultdict(int)

        name_map = self.name_exact_idx.get(country, {})
        suf_map = self.name_suf_idx.get(country, {})
        addr_map = self.addr_exact_idx.get(country, {})
        tok_map = self.token_idx.get(country, {})
        addr_tok_map = self.addr_tok_idx.get(country, {})
        num_map = self.numeric_idx.get(country, {})

        # 1. Exact Name Channel
        if name_norm and name_norm in name_map:
            for c_idx in name_map[name_norm]:
                cand_masks[c_idx] |= CH_NAME_EXACT

        # 2. Suffix Name Channel
        if name_suf and name_suf in suf_map:
            for c_idx in suf_map[name_suf]:
                cand_masks[c_idx] |= CH_NAME_SUF_EXACT

        # 3. Exact Address Channel
        if addr_norm and len(addr_norm) >= 5 and addr_norm in addr_map:
            for c_idx in addr_map[addr_norm]:
                cand_masks[c_idx] |= CH_ADDR_EXACT

        # 4. Rare / Medium Name Tokens
        if name_norm:
            q_tokens = set(name_norm.split()) - NAME_STOPWORDS
            for t in q_tokens:
                if len(t) >= 3 and t in tok_map:
                    for c_idx in tok_map[t]:
                        cand_masks[c_idx] |= CH_RARE_NAME_TOK

        # 5. Informative Address Tokens (Locality/Street)
        if addr_norm:
            q_a_tokens = set(addr_norm.split()) - ADDR_STOPWORDS
            for t in q_a_tokens:
                if len(t) >= 4 and t in addr_tok_map:
                    for c_idx in addr_tok_map[t]:
                        cand_masks[c_idx] |= CH_ADDR_TOK

        # 6. Meaningful Numeric Channel
        for num in set(nums):
            if len(num) >= 2 and num in num_map:
                for c_idx in num_map[num]:
                    cand_masks[c_idx] |= CH_NUMERIC

        # If candidates exceed budget, prioritize candidates with multiple channel confirmations
        if len(cand_masks) > max_candidates_per_query:
            sorted_cands = sorted(
                cand_masks.items(),
                key=lambda item: bin(item[1]).count("1"),
                reverse=True
            )
            return dict(sorted_cands[:max_candidates_per_query])

        return cand_masks
