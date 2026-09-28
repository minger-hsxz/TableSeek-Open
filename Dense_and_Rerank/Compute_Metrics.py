"""
compute_metrics.py

Compute Top-k Recall (R@k), Top-k Group Recall (GR@k), and
Top-1 Discriminative Score (DS@1) for a single output JSON file.
Results include overall metrics and breakdowns by question_type.

Usage:
    python compute_metrics.py \
        --input  /path/to/output.json \
        --output /path/to/metrics.csv \
        --mode   Embedding   # or Rerank
"""

import argparse
import ast
import json
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inter-table retrieval metrics from a single JSON file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input JSON file (output.json)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the resulting CSV file."
    )
    parser.add_argument(
        "--mode", required=True, choices=["Embedding", "Rerank"],
        help="Retrieval mode: 'Embedding' uses output_text; "
             "'Rerank' uses output_text_rerank."
    )
    parser.add_argument(
        "--max_k", type=int, default=5,
        help="Maximum k for top-k metrics (default: 5)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction parsing helpers
# ---------------------------------------------------------------------------

def parse_pred_list(raw_text):
    """
    Parse a stringified Python list into a list of (db_id, table_name) tuples.
    Entries that do not contain '::' are stored as (None, None).
    """
    try:
        pred_list = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return []

    parsed = []
    for p in pred_list:
        if "::" not in p:
            parsed.append((None, None))
        else:
            db_part, table_part = p.split("::", 1)
            parsed.append((db_part, table_part))
    return parsed


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _summarize_metrics(per_query, max_k):
    """
    Return aggregated metric values.

    R@k and GR@k are averaged over queries. DS@1 is computed globally as
    mean(R@1) / mean(GR@1), not as the mean of per-query DS@1 values.
    """
    if per_query.empty:
        return {}

    summary = {}
    for k in range(1, max_k + 1):
        summary[f"R@{k}"] = per_query[f"R@{k}"].mean()
        summary[f"GR@{k}"] = per_query[f"GR@{k}"].mean()

    gr1 = summary["GR@1"]
    summary["DS@1"] = summary["R@1"] / gr1 if gr1 > 0 else 0.0
    return summary


def compute_metrics(data, pred_key, max_k=5):
    """
    Compute R@k and GR@k for each query, then aggregate.

    DS@1 is computed at the aggregate level as mean(R@1) / mean(GR@1).

    Parameters
    ----------
    data     : list of dict  — loaded JSON records
    pred_key : str           — field name that holds the ranked prediction list
    max_k    : int           — maximum k to evaluate

    Returns
    -------
    per_query          : pd.DataFrame with one row per query and columns for each metric
    summary            : dict with averaged metric values over all queries
    summary_by_type    : dict mapping question_type -> averaged metric values
    """
    records = []

    for item in data:
        db_id      = item["db_id"]
        # Ground-truth: the single target table for answerability (R@k)
        target_table = item["highlighted_table"][0]          # e.g. "address_T0"
        # Ground-truth: the full sibling group for coarse relevance (GR@k)
        group_tables = set(item["all_tables_in_group"])      # e.g. {"address_T0", ...}
        group_size   = len(group_tables)

        if group_size == 0:
            continue

        # Parse the ranked prediction list for this mode
        parsed = parse_pred_list(item.get(pred_key, "[]"))

        row = {
            "db_id": db_id,
            "target_table": target_table,
            "group_size": group_size,
            "question_type": item.get("question_type", "unknown"),
        }

        # ---- R@k : has the exact target table appeared in top-k? ----
        target_found_at = None   # first rank (1-based) where target appears
        for k in range(1, max_k + 1):
            if k - 1 < len(parsed):
                p_db, p_table = parsed[k - 1]
                if p_db == db_id and p_table == target_table:
                    target_found_at = k
            row[f"R@{k}"] = 1.0 if (target_found_at is not None and
                                      target_found_at <= k) else 0.0

        # ---- GR@k : fraction of the sibling group covered in top-k ----
        # We accumulate hits from the ranked list for each k.
        seen_group_hits = 0
        for k in range(1, max_k + 1):
            if k - 1 < len(parsed):
                p_db, p_table = parsed[k - 1]
                if p_db == db_id and p_table in group_tables:
                    seen_group_hits += 1
            # denominator: min(|G_i|, k)  — as defined in the paper
            denom = min(group_size, k)
            row[f"GR@{k}"] = seen_group_hits / denom

        records.append(row)

    per_query = pd.DataFrame(records)

    summary = _summarize_metrics(per_query, max_k)

    summary_by_type = {}
    if not per_query.empty and "question_type" in per_query.columns:
        for qtype, group in per_query.groupby("question_type", sort=True):
            summary_by_type[qtype] = _summarize_metrics(group, max_k)

    return per_query, summary, summary_by_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Choose the prediction field based on the selected mode
    pred_key = "output_text" if args.mode == "Embedding" else "output_text_rerank"

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} records from {args.input}")
    print(f"[INFO] Mode: {args.mode}  →  using field '{pred_key}'")

    # Compute metrics
    per_query, summary, summary_by_type = compute_metrics(
        data, pred_key, max_k=args.max_k
    )

    def _print_summary(label, metrics):
        print(f"\n----- {label} -----")
        for metric, val in metrics.items():
            print(f"  {metric:8s}: {val:.4f}")

    # Print summary to console
    print("\n===== Metric Summary =====")
    _print_summary("Overall (all)", summary)
    for qtype in sorted(summary_by_type):
        _print_summary(f"question_type = {qtype}", summary_by_type[qtype])
    print("==========================\n")

    # Build output DataFrame: overall + per-question_type rows
    output_rows = []
    overall_row = {"question_type": "all"}
    overall_row.update({metric: round(val, 6) for metric, val in summary.items()})
    output_rows.append(overall_row)

    for qtype in sorted(summary_by_type):
        type_row = {"question_type": qtype}
        type_row.update(
            {metric: round(val, 6) for metric, val in summary_by_type[qtype].items()}
        )
        output_rows.append(type_row)

    output_df = pd.DataFrame(output_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    output_df.to_csv(args.output, index=False)
    print(f"[OK] Results written to {args.output}")


if __name__ == "__main__":
    main()