import csv, collections, json, math, os, time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")
DOCS_DIR  = os.path.join(REPO_ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def fmt_bytes(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def has_non_ascii(s):
    try: s.encode("ascii"); return False
    except UnicodeEncodeError: return True

def percentile(data, p):
    if not data: return 0
    data = sorted(data)
    k = (len(data)-1)*p/100
    f,c = math.floor(k), math.ceil(k)
    if f==c: return data[int(k)]
    return data[f]*(c-k)+data[c]*(k-f)

def dist_stats(lst):
    if not lst: return {}
    return {"min":min(lst),"max":max(lst),"mean":round(sum(lst)/len(lst),2),
            "p25":round(percentile(lst,25),2),"p50":round(percentile(lst,50),2),
            "p75":round(percentile(lst,75),2),"p95":round(percentile(lst,95),2)}

def audit_source(path, label):
    print(f"\n  [{label}] scanning {fmt_bytes(os.path.getsize(path))} ...")
    t0 = time.time()
    row_count = 0
    null_c = collections.defaultdict(int)
    pfx_c = collections.defaultdict(int)
    ctry_c = collections.defaultdict(int)
    nl,al,nt,at_ = [],[],[],[]
    non_a_n = non_a_a = 0
    seen_ids = set()
    dup_id = 0
    cols = None
    exp = "S1-" if "source1" in label else ("S2-" if "source2" in label else "S3-")
    unexp_pfx = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        cols = rdr.fieldnames
        for row in rdr:
            row_count += 1
            eid  = (row.get("entity_id") or "").strip()
            name = (row.get("business_name") or "").strip()
            addr = (row.get("business_address") or "").strip()
            ctry = (row.get("country") or "").strip()
            if not eid:
                null_c["entity_id"] += 1
            else:
                pfx = eid[:3]
                pfx_c[pfx] += 1
                if not eid.startswith(exp): unexp_pfx += 1
                if eid in seen_ids: dup_id += 1
                seen_ids.add(eid)
            if not name: null_c["business_name"] += 1
            else:
                nl.append(len(name)); nt.append(len(name.split()))
                if has_non_ascii(name): non_a_n += 1
            if not addr: null_c["business_address"] += 1
            else:
                al.append(len(addr)); at_.append(len(addr.split()))
                if has_non_ascii(addr): non_a_a += 1
            if not ctry: null_c["country"] += 1
            else: ctry_c[ctry] += 1
    elapsed = time.time()-t0
    print(f"    rows={row_count:,}  unique_ids={len(seen_ids):,}  dup_ids={dup_id}  elapsed={elapsed:.1f}s")
    print(f"    nulls={dict(null_c)}")
    print(f"    countries={dict(ctry_c)}")
    return {"label":label,"size_bytes":os.path.getsize(path),"size_human":fmt_bytes(os.path.getsize(path)),
            "columns":list(cols) if cols else [],"row_count":row_count,"unique_ids":len(seen_ids),
            "dup_id_count":dup_id,"unexpected_prefix":unexp_pfx,"prefix_dist":dict(pfx_c),
            "null_counts":dict(null_c),
            "missing_pct":{k:round(v/row_count*100,3) if row_count else 0 for k,v in null_c.items()},
            "country_dist":dict(sorted(ctry_c.items(),key=lambda x:-x[1])),
            "name_len_dist":dist_stats(nl),"addr_len_dist":dist_stats(al),
            "name_tok_dist":dist_stats(nt),"addr_tok_dist":dist_stats(at_),
            "non_ascii_name_count":non_a_n,
            "non_ascii_name_pct":round(non_a_n/row_count*100,3) if row_count else 0,
            "non_ascii_addr_count":non_a_a,
            "non_ascii_addr_pct":round(non_a_a/row_count*100,3) if row_count else 0,
            "audit_sec":round(elapsed,1)}

def audit_gt(path):
    print(f"\n  [ground_truth] scanning ...")
    t0 = time.time()
    row_count = 0
    s2_m,s3_m = set(),set()
    mps1,s2ps1,s3ps1 = [],[],[]
    zero = one = multi = both = 0
    s2_part = collections.Counter()
    s3_part = collections.Counter()
    cols = None
    with open(path, encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        cols = rdr.fieldnames
        for row in rdr:
            row_count += 1
            mraw = (row.get("matched_entity_ids") or "").strip()
            mids = [m.strip() for m in mraw.split(",") if m.strip()] if mraw else []
            n = len(mids)
            mps1.append(n)
            s2h = [m for m in mids if m.startswith("S2-")]
            s3h = [m for m in mids if m.startswith("S3-")]
            s2ps1.append(len(s2h)); s3ps1.append(len(s3h))
            for m in s2h: s2_m.add(m); s2_part[m] += 1
            for m in s3h: s3_m.add(m); s3_part[m] += 1
            if n==0: zero += 1
            elif n==1: one += 1
            else: multi += 1
            if s2h and s3h: both += 1
    match_dist = collections.Counter(mps1)
    elapsed = time.time()-t0
    print(f"    rows={row_count:,}  singletons={zero:,}  one={one:,}  multi={multi:,}  elapsed={elapsed:.1f}s")
    print(f"    total_matches={sum(mps1):,}  max_per_s1={max(mps1) if mps1 else 0}")
    print(f"    both_s2_s3={both:,}  s2_unique={len(s2_m):,}  s3_unique={len(s3_m):,}")
    return {"columns":list(cols) if cols else [],"row_count":row_count,
            "s1_zero_matches":zero,"s1_one_match":one,"s1_multi_matches":multi,"s1_both_s2s3":both,
            "s2_unique_matched":len(s2_m),"s3_unique_matched":len(s3_m),
            "total_matches":sum(mps1),"max_matches_per_s1":max(mps1) if mps1 else 0,
            "singleton_pct":round(zero/row_count*100,3) if row_count else 0,
            "mps1_dist":dist_stats(mps1),"s2ps1_dist":dist_stats(s2ps1),"s3ps1_dist":dist_stats(s3ps1),
            "match_freq":{str(k):v for k,v in sorted(match_dist.items())},
            "s2_multi_participation":sum(1 for v in s2_part.values() if v>1),
            "s3_multi_participation":sum(1 for v in s3_part.values() if v>1),
            "audit_sec":round(elapsed,1)}

def load_ids(path):
    ids = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            eid = (row.get("entity_id") or "").strip()
            if eid: ids.add(eid)
    return ids

FILES = {
    "train_source1": os.path.join(TRAIN_DIR,"train_source1.tsv"),
    "train_source2": os.path.join(TRAIN_DIR,"train_source2.tsv"),
    "train_source3": os.path.join(TRAIN_DIR,"train_source3.tsv"),
    "train_ground_truth": os.path.join(TRAIN_DIR,"train_ground_truth.tsv"),
    "test_source1":  os.path.join(TEST_DIR,"test_source1.tsv"),
    "test_source2":  os.path.join(TEST_DIR,"test_source2.tsv"),
    "test_source3":  os.path.join(TEST_DIR,"test_source3.tsv"),
}

print("="*60)
print("Amazon ML Challenge 2026 - Dataset Audit")
print(f"pandas={HAS_PANDAS}  polars={HAS_POLARS}")
print("="*60)

print("\nFile Inventory:")
for k,p in FILES.items():
    exists = os.path.exists(p)
    sz = fmt_bytes(os.path.getsize(p)) if exists else "MISSING"
    print(f"  {'OK':7} {sz:>12}  {k}")

all_p = {}
for k in ["train_source1","train_source2","train_source3",
          "test_source1","test_source2","test_source3"]:
    all_p[k] = audit_source(FILES[k], k)
all_p["train_ground_truth"] = audit_gt(FILES["train_ground_truth"])

print("\nLeakage ID check ...")
tr1=load_ids(FILES["train_source1"]); tr2=load_ids(FILES["train_source2"]); tr3=load_ids(FILES["train_source3"])
te1=load_ids(FILES["test_source1"]);  te2=load_ids(FILES["test_source2"]);  te3=load_ids(FILES["test_source3"])
leak = {"s1_ids_in_s2_train":len(tr1&tr2),"s1_ids_in_s3_train":len(tr1&tr3),
        "train_test_s1_overlap":len(tr1&te1),"train_test_s2_overlap":len(tr2&te2),"train_test_s3_overlap":len(tr3&te3)}
print(f"  {leak}")

ts1=all_p["test_source1"]["row_count"]; ts2=all_p["test_source2"]["row_count"]; ts3=all_p["test_source3"]["row_count"]
trs1=all_p["train_source1"]["row_count"]; trs2=all_p["train_source2"]["row_count"]; trs3=all_p["train_source3"]["row_count"]
cand_scale = {"test_naive_cartesian_pairs":ts1*(ts2+ts3),"train_naive_cartesian_pairs":trs1*(trs2+trs3)}
print(f"\nCandidate scale: {cand_scale}")

gt = all_p["train_ground_truth"]
total_s1 = gt["row_count"]
singletons = gt["s1_zero_matches"]
singleton_pct = gt["singleton_pct"]
baseline_none_f05 = round(singletons/total_s1,4) if total_s1 else 0
print(f"\nF0.5 Implications:")
print(f"  Total S1 train  : {total_s1:,}")
print(f"  Singletons      : {singletons:,} ({singleton_pct}%)")
print(f"  One-match       : {gt['s1_one_match']:,}")
print(f"  Multi-match     : {gt['s1_multi_matches']:,}")
print(f"  F0.5 predict-none baseline: {baseline_none_f05}")

tr_ctries = set()
for k in ["train_source1","train_source2","train_source3"]:
    tr_ctries |= set(all_p[k]["country_dist"].keys())
te_ctries = set()
for k in ["test_source1","test_source2","test_source3"]:
    te_ctries |= set(all_p[k]["country_dist"].keys())
print(f"\nCountry shift:")
print(f"  Train: {sorted(tr_ctries)}")
print(f"  Test : {sorted(te_ctries)}")
print(f"  NEW in test: {sorted(te_ctries - tr_ctries)}")

print("\nSample rows (train_source1):")
with open(FILES["train_source1"], encoding="utf-8", errors="replace") as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for i,row in enumerate(rdr):
        if i>=5: break
        print(f"  {dict(row)}")

print("\nSample rows (train_ground_truth):")
with open(FILES["train_ground_truth"], encoding="utf-8", errors="replace") as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for i,row in enumerate(rdr):
        if i>=5: break
        print(f"  {dict(row)}")

profile = {
    "audit_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "files": all_p, "leakage": leak, "candidate_scale": cand_scale,
    "f05_analysis": {"total_s1":total_s1,"singletons":singletons,
                     "singleton_pct":singleton_pct,"baseline_none_f05":baseline_none_f05},
    "country_shift": {"train":sorted(tr_ctries),"test":sorted(te_ctries),"new_in_test":sorted(te_ctries-tr_ctries)},
}
with open(os.path.join(DOCS_DIR,"data_profile.json"),"w",encoding="utf-8") as f:
    json.dump(profile,f,indent=2,ensure_ascii=False)
print("\nWrote docs/data_profile.json")
print("AUDIT DONE")
