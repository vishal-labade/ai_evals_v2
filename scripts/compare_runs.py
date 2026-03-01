import argparse
import csv
import os
from collections import defaultdict
from statistics import mean

def pct(x):
    return f"{100*x:.1f}%"

def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows

def summarize(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(int(r["score"]))
    return {k: (mean(v), len(v)) for k, v in g.items()}

def top_failures(rows, n=10):
    # Return prompt_ids with score=0, sorted by latency desc (useful debugging) then prompt_id
    fails = [r for r in rows if int(r["score"]) == 0]
    fails.sort(key=lambda r: (-(int(r["latency_ms"]) if r.get("latency_ms") else 0), r["prompt_id"]))
    return fails[:n]

def overall(rows):
    return mean(int(r["score"]) for r in rows)

def mean_col(rows, col):
    vals = []
    for r in rows:
        v = r.get(col)
        if v is None or v == "":
            continue
        try:
            vals.append(int(v))
        except:
            pass
    return mean(vals) if vals else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="metrics CSV A (baseline)")
    ap.add_argument("--b", required=True, help="metrics CSV B (candidate)")
    ap.add_argument("--out", default=None, help="output markdown (default results/reports/compare_<a>_vs_<b>.md)")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    A = load_rows(args.a)
    B = load_rows(args.b)

    run_a = A[0].get("run_id", os.path.basename(args.a))
    run_b = B[0].get("run_id", os.path.basename(args.b))
    model_a = A[0].get("model_name", "NA")
    model_b = B[0].get("model_name", "NA")

    # Build prompt_id -> row mapping (assumes same suite)
    mapA = {r["prompt_id"]: r for r in A}
    mapB = {r["prompt_id"]: r for r in B}
    shared = sorted(set(mapA) & set(mapB))

    if not shared:
        raise SystemExit("No shared prompt_ids between runs.")

    # Overall
    acc_a = overall(A)
    acc_b = overall(B)

    # Hallucination proxies (if present)
    nicv_a = mean_col(A, "not_in_context_violation")
    nicv_b = mean_col(B, "not_in_context_violation")
    numinv_a = mean_col(A, "numeric_invention_flag")
    numinv_b = mean_col(B, "numeric_invention_flag")

    # By slices
    by_task_a = summarize(A, "task")
    by_task_b = summarize(B, "task")
    by_domain_a = summarize(A, "domain")
    by_domain_b = summarize(B, "domain")
    by_diff_a = summarize(A, "difficulty")
    by_diff_b = summarize(B, "difficulty")

    # Per-prompt deltas
    improved = []
    regressed = []
    unchanged = 0

    for pid in shared:
        sa = int(mapA[pid]["score"])
        sb = int(mapB[pid]["score"])
        if sb > sa:
            improved.append(pid)
        elif sb < sa:
            regressed.append(pid)
        else:
            unchanged += 1

    # Top regressed/improved prompts with metadata
    def prompt_rows(pids, which="B", k=10):
        out = []
        for pid in pids[:k]:
            r = mapB[pid] if which == "B" else mapA[pid]
            out.append({
                "prompt_id": pid,
                "task": r.get("task"),
                "domain": r.get("domain"),
                "difficulty": r.get("difficulty"),
                "latency_ms": r.get("latency_ms"),
            })
        return out

    # Sort improved/regressed by slice importance: hard first, then domain, then prompt_id
    diff_rank = {"hard": 0, "medium": 1, "easy": 2}
    improved.sort(key=lambda pid: (diff_rank.get(mapB[pid].get("difficulty","medium"), 1), mapB[pid].get("task",""), pid))
    regressed.sort(key=lambda pid: (diff_rank.get(mapB[pid].get("difficulty","medium"), 1), mapB[pid].get("task",""), pid))

    # Output
    out_path = args.out or f"results/reports/compare_{run_a}_vs_{run_b}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Run Comparison\n\n")
        f.write(f"- **A (baseline):** `{run_a}` (`{model_a}`)\n")
        f.write(f"- **B (candidate):** `{run_b}` (`{model_b}`)\n")
        f.write(f"- **Shared prompts:** {len(shared)}\n\n")

        f.write("## Overall\n")
        f.write(f"- Accuracy A: **{pct(acc_a)}**\n")
        f.write(f"- Accuracy B: **{pct(acc_b)}**\n")
        f.write(f"- Delta (B - A): **{pct(acc_b - acc_a)}**\n\n")

        if nicv_a is not None or nicv_b is not None:
            f.write("## Hallucination proxies (V1)\n")
            f.write(f"- NOT_IN_CONTEXT violation A: {pct(nicv_a) if nicv_a is not None else 'NA'}\n")
            f.write(f"- NOT_IN_CONTEXT violation B: {pct(nicv_b) if nicv_b is not None else 'NA'}\n")
            f.write(f"- Numeric invention A: {pct(numinv_a) if numinv_a is not None else 'NA'}\n")
            f.write(f"- Numeric invention B: {pct(numinv_b) if numinv_b is not None else 'NA'}\n\n")

        f.write("## Prompt-level changes\n")
        f.write(f"- Improved (B>A): **{len(improved)}**\n")
        f.write(f"- Regressed (B<A): **{len(regressed)}**\n")
        f.write(f"- Unchanged: **{unchanged}**\n\n")

        def slice_table(title, sa, sb):
            keys = sorted(set(sa) | set(sb))
            f.write(f"## {title}\n")
            f.write("| Slice | n | Acc A | Acc B | Delta |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for k in keys:
                a = sa.get(k, (0.0, 0))
                b = sb.get(k, (0.0, 0))
                # Use n from B if present else A
                n = b[1] if b[1] else a[1]
                f.write(f"| {k} | {n} | {pct(a[0])} | {pct(b[0])} | {pct(b[0]-a[0])} |\n")
            f.write("\n")

        slice_table("By task", by_task_a, by_task_b)
        slice_table("By domain", by_domain_a, by_domain_b)
        slice_table("By difficulty", by_diff_a, by_diff_b)

        f.write("## Top improved prompt_ids\n")
        for r in prompt_rows(improved, which="B", k=args.topk):
            f.write(f"- {r['prompt_id']} ({r['task']}/{r['domain']}/{r['difficulty']})\n")
        f.write("\n")

        f.write("## Top regressed prompt_ids\n")
        for r in prompt_rows(regressed, which="B", k=args.topk):
            f.write(f"- {r['prompt_id']} ({r['task']}/{r['domain']}/{r['difficulty']})\n")
        f.write("\n")

    print("Wrote:", out_path)

if __name__ == "__main__":
    main()