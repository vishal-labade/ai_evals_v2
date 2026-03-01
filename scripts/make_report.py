import argparse, csv, os
from collections import defaultdict
from statistics import mean

def pct(x): 
    return f"{100*x:.1f}%"

def _parse_int(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return default
    try:
        return int(float(s))
    except Exception:
        return default

def _parse_float(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return default
    try:
        return float(s)
    except Exception:
        return default

def _parse_score(x):
    # score can be int-like or float-like; return float for mean()
    v = _parse_float(x, default=None)
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="results/metrics/<run_id>.csv")
    ap.add_argument("--out", default=None, help="results/reports/<run_id>.md")
    args = ap.parse_args()

    with open(args.metrics, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("No rows found")

    run_id = rows[0].get("run_id", "(unknown)")
    model = rows[0].get("model_name", "(unknown)")
    out = args.out or f"results/reports/{run_id}.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # ---- Parse score once and attach to each row ----
    missing_score_rows = 0
    for r in rows:
        s = _parse_score(r.get("score"))
        if s is None:
            missing_score_rows += 1
        r["_score"] = s

    present_scores = [r["_score"] for r in rows if r["_score"] is not None]
    if not present_scores:
        raise SystemExit("No valid scores found in metrics file (all missing/blank).")

    # ---- Latency ----
    lat = [_parse_int(r.get("latency_ms"), default=None) for r in rows]
    lat = [x for x in lat if x is not None]

    # ---- Grouped scores (skip missing) ----
    by_task = defaultdict(list)
    by_domain = defaultdict(list)
    by_diff = defaultdict(list)

    for r in rows:
        s = r["_score"]
        if s is None:
            continue
        by_task[r.get("task", "NA")].append(s)
        by_domain[r.get("domain", "NA")].append(s)
        by_diff[r.get("difficulty", "NA")].append(s)

    # ---- JSON validity (robust parsing) ----
    json_strict = []
    json_lenient = []
    for r in rows:
        if r.get("scoring_method") == "json_schema":
            js = _parse_int(r.get("json_valid_strict"), default=None)
            jl = _parse_int(r.get("json_valid_lenient"), default=None)
            if js is not None:
                json_strict.append(js)
            if jl is not None:
                json_lenient.append(jl)

    # ---- Hallucination proxies (robust parsing) ----
    nicv = [_parse_int(r.get("not_in_context_violation"), default=0) for r in rows]
    numinv = [_parse_int(r.get("numeric_invention_flag"), default=0) for r in rows]

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Run Report\n\n")
        f.write(f"- **Run ID:** `{run_id}`\n")
        f.write(f"- **Model:** `{model}`\n")
        f.write(f"- **N prompts:** {len(rows)}\n")
        f.write(f"- **Missing scores:** {missing_score_rows}\n\n")

        f.write("## Overall\n")
        f.write(f"- **Accuracy (score mean):** {pct(mean(present_scores))}\n")
        if lat:
            f.write(f"- **Avg latency:** {mean(lat):.0f} ms\n")
        if json_strict:
            f.write(f"- **JSON valid (strict):** {pct(mean(json_strict))}\n")
        if json_lenient:
            f.write(f"- **JSON valid (lenient):** {pct(mean(json_lenient))}\n")
        f.write(f"- **NOT_IN_CONTEXT violation rate:** {pct(mean(nicv))}\n")
        f.write(f"- **Numeric invention flag rate:** {pct(mean(numinv))}\n\n")

        f.write("## By task\n")
        for k in sorted(by_task):
            f.write(f"- {k}: {pct(mean(by_task[k]))} (n={len(by_task[k])})\n")

        f.write("\n## By domain\n")
        for k in sorted(by_domain):
            f.write(f"- {k}: {pct(mean(by_domain[k]))} (n={len(by_domain[k])})\n")

        f.write("\n## By difficulty\n")
        for k in sorted(by_diff):
            f.write(f"- {k}: {pct(mean(by_diff[k]))} (n={len(by_diff[k])})\n")

    print("Wrote:", out)

if __name__ == "__main__":
    main()