"""03/04 - Hard-negative ambiguity + positive evidence hierarchy (from pair_pool).

03: among same-name / same-address candidates, how ambiguous are they (FP risk)?
04: on true matches, which evidence (name / address / numeric / rare-token) is decisive?
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import polars as pl

from common import WORK_DIR
from pairfeat import auc_score

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)
POOL = os.path.join(WORK_DIR, "pair_pool.parquet")

NAME_STRONG = 0.8
ADDR_STRONG = 0.5


def rate_col(df, col, levels):
    out = {}
    for lo, name in levels:
        m = df[col] >= lo
        out[name] = {"n": int(m.sum()), "rate": round(float(m.mean()), 5)}
    return out


def main():
    pool = pl.read_parquet(POOL)
    pos = pool.filter(pl.col("label") == 1)
    neg = pool.filter(pl.col("label") == 0)
    n_pos, n_neg = int(pos.height), int(neg.height)

    # ---------------- 03 ambiguity ----------------
    same_name = pool["pool_name_eq"].to_numpy()
    same_addr = pool["pool_addr_eq"].to_numpy()

    amb = {"n_pos": n_pos, "n_neg": n_neg}
    # same-name candidates: how many are negatives (non-matches)?
    subm = pool.filter("pool_name_eq")
    amb["same_name_pairs"] = int(subm.height)
    amb["same_name_neg_frac"] = round(float(subm["label"].mean()) , 5) if subm.height else 0
    # of same-name NEGATIVES, how often is the address still confusably similar?
    sn_neg = subm.filter(pl.col("label") == 0)
    def f_frac(df, col, lo):
        return round(float((df[col] >= lo).mean()), 5) if df.height else 0
    amb["same_name_neg_addr_strong_frac"] = f_frac(sn_neg, "addr_tok_jac", 0.5)
    amb["same_name_neg_addr_char2_0_6_frac"] = f_frac(sn_neg, "addr_char2", 0.6)
    amb["same_name_neg_num_overlap1_frac"] = f_frac(sn_neg, "num_overlap", 1)
    amb["same_name_neg_both_addr_strong_info"] = {
        "n": int(sn_neg.filter((pl.col("addr_tok_jac") >= 0.5) & (pl.col("num_overlap") >= 1)).height),
        "frac": f_frac(sn_neg.filter("label") if False else sn_neg, "num_overlap", 1),
    }
    same_addr_neg = pool.filter(pl.col("pool_addr_eq") & (pl.col("label") == 0))
    amb["same_addr_neg_n"] = int(same_addr_neg.height)
    amb["same_addr_neg_name_strong_frac"] = f_frac(same_addr_neg, "name_tok_jac", 0.5)
    # channel-level detail
    per_ch = {}
    for ch in ["n_nameblk", "n_addrblk", "n_raretok", "n_random"]:
        sub = pool.filter((pl.col("channel") == ch) & (pl.col("label") == 0))
        per_ch[ch] = {
            "n": int(sub.height),
            "name_eq": round(float((sub["pool_name_eq"]).mean()), 5),
            "addr_eq": round(float((sub["pool_addr_eq"]).mean()), 5),
            "name_tok_jac_ge_0_5": round(float((sub["name_tok_jac"] >= 0.5).mean()), 5),
            "addr_tok_jac_ge_0_5": round(float((sub["addr_tok_jac"] >= 0.5).mean()), 5),
            "num_overlap_ge_1": round(float((sub["num_overlap"] >= 1).mean()), 5),
            "rare_shared_ge_1": round(float((sub["rare_shared"] >= 1).mean()), 5),
        }
    amb["per_channel_neg"] = per_ch
    # hardest negatives: both name and address strongly confusable
    hard = neg.filter((pl.col("name_tok_jac") >= NAME_STRONG) & (pl.col("addr_tok_jac") >= ADDR_STRONG))
    amb["hard_both_strong_neg"] = int(hard.height)
    hard2 = neg.filter((pl.col("name_tok_jac") >= NAME_STRONG) & (pl.col("num_overlap") >= 1))
    amb["hard_namestrong_num"] = int(hard2.height)
    with open(os.path.join(OUT, "03_ambiguity.json"), "w") as f:
        json.dump(amb, f, indent=2)

    # ---------------- 04 evidence hierarchy ----------------
    pn = pos["name_tok_jac"].to_numpy()
    pc = pos["name_contain_c"].to_numpy()
    pa = pos["addr_tok_jac"].to_numpy()
    pnu = pos["num_overlap"].to_numpy()
    pr = pos["rare_shared"].to_numpy()
    pn2 = pos["name_char2"].to_numpy()
    pam = pos["addr_char2"].to_numpy()

    name_strong = (pn >= NAME_STRONG) & (pc >= NAME_STRONG)
    addr_strong = (pa >= ADDR_STRONG) | (pnu >= 1)
    num_strong = pnu >= 1
    rare_strong = pr >= 1

    evidence = {
        "n_pos": n_pos,
        "name_strong": {"n": int(name_strong.sum()), "rate": round(float(name_strong.mean()), 5)},
        "addr_strong": {"n": int(addr_strong.sum()), "rate": round(float(addr_strong.mean()), 5)},
        "numeric_strong": {"n": int(num_strong.sum()), "rate": round(float(num_strong.mean()), 5)},
        "rare_strong": {"n": int(rare_strong.sum()), "rate": round(float(rare_strong.mean()), 5)},
        "name_and_addr_strong": {"n": int((name_strong & addr_strong).sum()),
                                 "rate": round(float((name_strong & addr_strong).mean()), 5)},
        "name_only_strong": {"n": int((name_strong & ~addr_strong).sum()),
                             "rate": round(float((name_strong & ~addr_strong).mean()), 5)},
        "addr_only_strong": {"n": int((addr_strong & ~name_strong).sum()),
                             "rate": round(float((addr_strong & ~name_strong).mean()), 5)},
        "neither_strong": {"n": int((~name_strong & ~addr_strong).sum()),
                           "rate": round(float((~name_strong & ~addr_strong).mean()), 5)},
        "name_gate_recall": rate_col(pos, "name_tok_jac", [(0.4, "0.4"), (0.6, "0.6"),
                                                           (0.8, "0.8"), (0.95, "0.95")]),
        "addr_gate_recall": rate_col(pos, "addr_tok_jac", [(0.2, "0.2"), (0.4, "0.4"),
                                                           (0.6, "0.6"), (0.8, "0.8")]),
        "num_overlap_recall": rate_col(pos, "num_overlap", [(1, "1"), (2, "2")]),
        "name_char2_recall": rate_col(pos, "name_char2", [(0.7, "0.7"), (0.9, "0.9")]),
        "addr_char2_recall": rate_col(pos, "addr_char2", [(0.5, "0.5"), (0.8, "0.8")]),
    }
    # rescue analysis: when name is weak, what addr/numeric evidence covers it?
    weak_name = ~name_strong
    wn_idx = np.flatnonzero(weak_name)
    wn_rescue = {
        "name_weak_n": int(wn_idx.size),
        "name_weak_frac": round(float(weak_name.mean()), 5),
        "addr_strong": round(float(addr_strong[wn_idx].mean()), 5) if wn_idx.size else 0,
        "numeric_strong": round(float(num_strong[wn_idx].mean()), 5) if wn_idx.size else 0,
        "addr_char2_0_6": round(float((pam[wn_idx] >= 0.6).mean()), 5) if wn_idx.size else 0,
        "avg_addr_tok_jac": round(float(pa[wn_idx].mean()), 5) if wn_idx.size else 0,
    }
    weak_addr = ~addr_strong
    wa_idx = np.flatnonzero(weak_addr)
    wa_rescue = {
        "addr_weak_n": int(wa_idx.size),
        "name_strong": round(float(name_strong[wa_idx].mean()), 5) if wa_idx.size else 0,
        "avg_name_tok_jac": round(float(pn[wa_idx].mean()), 5) if wa_idx.size else 0,
    }
    evidence["rescue_when_name_weak"] = wn_rescue
    evidence["rescue_when_addr_weak"] = wa_rescue
    # AUC within country for the headline features
    for country in ["India", "US"]:
        pcs = pos.filter(pl.col("country") == country)
        ncs = neg.filter(pl.col("country") == country)
        if ncs.height == 0 or pcs.height == 0:
            continue
        evidence[f"auc_{country}"] = {
            "name_tok_jac": round(float(auc_score(pcs["name_tok_jac"].to_numpy(), ncs["name_tok_jac"].to_numpy())), 5),
            "addr_tok_jac": round(float(auc_score(pcs["addr_tok_jac"].to_numpy(), ncs["addr_tok_jac"].to_numpy())), 5),
            "addr_char2": round(float(auc_score(pcs["addr_char2"].to_numpy(), ncs["addr_char2"].to_numpy())), 5),
            "num_overlap": round(float(auc_score(pcs["num_overlap"].to_numpy(), ncs["num_overlap"].to_numpy())), 5),
            "rare_shared": round(float(auc_score(pcs["rare_shared"].to_numpy(), ncs["rare_shared"].to_numpy())), 5),
            "name_char2": round(float(auc_score(pcs["name_char2"].to_numpy(), ncs["name_char2"].to_numpy())), 5),
        }
    with open(os.path.join(OUT, "04_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(amb, indent=2))
    print("---- evidence ----")
    print(json.dumps(evidence, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()