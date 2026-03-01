import argparse, csv, glob, os
from collections import defaultdict
from statistics import mean, pstdev

def pct(x): 
    return f"{100*x:.1f}%"

def safe_float(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() in {"na","nan","none"}:
        return default
    try:
        return float(s)
    except Exception:
        return default

def safe_int(x, default=None):
    v = safe_float(x, default=None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def infer_temp(rows, fallback="NA"):
    # Prefer explicit temperature column if present and non-empty
    t = rows[0].get("temperature", "") if rows else ""
    t = str(t).strip()
    if t != "" and t.lower() not in {"na","nan","none"}:
        return t
    # Heuristic fallback using run_id naming convention
    rid = str(rows[0].get("run_id","")).strip()
    if "_t0_" in rid:
        return "0.00"
    return fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-glob", default="results/metrics/*.csv")
    ap.add_argument("--out", default="results/reports/leaderboard_by_setting.md")
    ap.add_argument("--skip_runs_with_missing_scores", action="store_true",
                    help="If set, drop any run that has missing/blank scores.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.metrics_glob))
    if not files:
        raise SystemExit("No metrics CSVs found.")

    groups = defaultdict(lambda: {
        "runs": 0,
        "acc": [],
        "lat": [],
        "nicv": [],
        "numinv": [],
        "json_strict": [],
        "json_lenient": [],
        "missing_scores": [],  # product-grade visibility
        "n_scored": [],
        "n_total": [],
    })

    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        model = rows[0].get("model_name","NA")
        temp = infer_temp(rows, fallback="NA")
        key = (model, temp)

        # ---- score parsing: missing is missing (NOT 0) ----
        scores = [safe_float(r.get("score"), default=None) for r in rows]
        present_scores = [s for s in scores if s is not None]
        missing = len(scores) - len(present_scores)

        if not present_scores:
            # If a file is completely broken, skip it rather than injecting zeros.
            # (You can also choose to fail hard)
            continue

        if args.skip_runs_with_missing_scores and missing > 0:
            continue

        # ---- latency (ms) ----
        lat = [safe_int(r.get("latency_ms"), default=None) for r in rows]
        lat = [x for x in lat if x is not None]

        # ---- hallucination proxies ----
        nicv = [safe_int(r.get("not_in_context_violation"), default=None) for r in rows]
        nicv = [x for x in nicv if x is not None]

        numinv = [safe_int(r.get("numeric_invention_flag"), default=None) for r in rows]
        numinv = [x for x in numinv if x is not None]

        # ---- json validity ----
        json_rows = [r for r in rows if r.get("scoring_method") == "json_schema"]
        js = [safe_int(r.get("json_valid_strict"), default=None) for r in json_rows] if json_rows else []
        js = [x for x in js if x is not None]
        jl = [safe_int(r.get("json_valid_lenient"), default=None) for r in json_rows] if json_rows else []
        jl = [x for x in jl if x is not None]

        g = groups[key]
        g["runs"] += 1
        g["acc"].append(mean(present_scores))
        if lat: g["lat"].append(mean(lat))
        if nicv: g["nicv"].append(mean(nicv))
        if numinv: g["numinv"].append(mean(numinv))
        if js: g["json_strict"].append(mean(js))
        if jl: g["json_lenient"].append(mean(jl))
        g["missing_scores"].append(missing)
        g["n_scored"].append(len(present_scores))
        g["n_total"].append(len(scores))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Sort by mean accuracy desc
    items = []
    for (model, temp), g in groups.items():
        mu = mean(g["acc"])
        sd = pstdev(g["acc"]) if len(g["acc"]) > 1 else 0.0
        items.append((mu, model, temp, g, sd))
    items.sort(reverse=True, key=lambda x: x[0])

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Leaderboard by model + temperature\n\n")
        out.write("| Model | Temp | Runs | Mean Acc | Std | Min | Max | Avg latency | NOT_IN_CONTEXT viol | Numeric invent | Missing scores |\n")
        out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for mu, model, temp, g, sd in items:
            mn = min(g["acc"])
            mx = max(g["acc"])
            lat = mean(g["lat"]) if g["lat"] else None
            nicv = mean(g["nicv"]) if g["nicv"] else None
            numinv = mean(g["numinv"]) if g["numinv"] else None
            miss = int(mean(g["missing_scores"])) if g["missing_scores"] else 0
            out.write(
                f"| `{model}` | {temp} | {g['runs']} | {pct(mu)} | {pct(sd)} | {pct(mn)} | {pct(mx)} | "
                f"{(str(int(lat))+' ms') if lat is not None else 'NA'} | "
                f"{pct(nicv) if nicv is not None else 'NA'} | "
                f"{pct(numinv) if numinv is not None else 'NA'} | "
                f"{miss} |\n"
            )

        out.write("\n## Interpretation\n")
        out.write("- Mean Acc is computed over **present (non-missing) prompt scores** within each run.\n")
        out.write("- If you see Missing scores > 0, fix upstream scoring to always emit score or log score_error.\n")
        out.write("- Treat **Temp=0** as the deterministic capability baseline.\n")

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()