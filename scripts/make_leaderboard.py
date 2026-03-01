import argparse, csv, glob, os
from collections import defaultdict
from statistics import mean

def pct(x): return f"{100*x:.1f}%"

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def summarize_metrics(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return None

    run_id = rows[0].get("run_id", os.path.splitext(os.path.basename(csv_path))[0])
    model = rows[0].get("model_name", "NA")
    n = len(rows)

    score = [safe_int(r.get("score","0")) for r in rows]
    lat = [safe_int(r.get("latency_ms","0")) for r in rows if r.get("latency_ms")]

    json_rows = [r for r in rows if r.get("scoring_method") == "json_schema"]
    json_strict = [safe_int(r.get("json_valid_strict","0")) for r in json_rows] if json_rows else []
    json_lenient = [safe_int(r.get("json_valid_lenient","0")) for r in json_rows] if json_rows else []

    nicv = [safe_int(r.get("not_in_context_violation","0")) for r in rows if r.get("not_in_context_violation") not in (None,"")]
    numinv = [safe_int(r.get("numeric_invention_flag","0")) for r in rows if r.get("numeric_invention_flag") not in (None,"")]

    return {
        "run_id": run_id,
        "model": model,
        "n": n,
        "accuracy": mean(score) if score else 0.0,
        "avg_latency_ms": mean(lat) if lat else None,
        "json_strict": mean(json_strict) if json_strict else None,
        "json_lenient": mean(json_lenient) if json_lenient else None,
        "not_in_ctx_violation": mean(nicv) if nicv else None,
        "numeric_invention": mean(numinv) if numinv else None,
        "csv_path": csv_path,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-glob", default="results/metrics/*.csv")
    ap.add_argument("--out", default="results/reports/leaderboard.md")
    args = ap.parse_args()

    files = sorted(glob.glob(args.metrics_glob))
    if not files:
        raise SystemExit(f"No metrics files found: {args.metrics_glob}")

    summaries = []
    for f in files:
        s = summarize_metrics(f)
        if s:
            summaries.append(s)

    # Sort by accuracy desc, then latency asc (if available)
    summaries.sort(key=lambda x: (-x["accuracy"], x["avg_latency_ms"] if x["avg_latency_ms"] is not None else 1e18))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Leaderboard\n\n")
        out.write("| Run | Model | N | Accuracy | Avg latency | JSON strict | JSON lenient | NOT_IN_CONTEXT violation | Numeric invention |\n")
        out.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for s in summaries:
            out.write(
                f"| `{s['run_id']}` | `{s['model']}` | {s['n']} | {pct(s['accuracy'])} | "
                f"{(str(int(s['avg_latency_ms']))+' ms') if s['avg_latency_ms'] is not None else 'NA'} | "
                f"{pct(s['json_strict']) if s['json_strict'] is not None else 'NA'} | "
                f"{pct(s['json_lenient']) if s['json_lenient'] is not None else 'NA'} | "
                f"{pct(s['not_in_ctx_violation']) if s['not_in_ctx_violation'] is not None else 'NA'} | "
                f"{pct(s['numeric_invention']) if s['numeric_invention'] is not None else 'NA'} |\n"
            )

        out.write("\n## Notes\n")
        out.write("- Accuracy is the mean of `score` across prompts.\n")
        out.write("- JSON strict/lenient are computed only for `scoring_method=json_schema` prompts.\n")
        out.write("- NOT_IN_CONTEXT violation and numeric invention are V1 deterministic hallucination proxies.\n")

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()