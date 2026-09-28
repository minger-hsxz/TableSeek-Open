"""
TableSeek: structure-aware multi-round table retrieval + QA.

CLI entry point for running TableSeek on a query JSON file against a table corpus.
"""

import json
import os
import sys
import logging
import datetime
import time
import warnings
import tqdm
import traceback
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Allow running as ``python TableSeek.py`` from this directory.
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from tableseek.tables import load_all_tables, build_table_key
from tableseek.logging_utils import setup_logging
from tableseek.patterns import TASK_TYPES
from tableseek.retrieve import multi_round_retrieve_and_qa
from tableseek.metrics import (
    cal_score3,
    cal_tabfact_score,
    cal_tabfact_aggregate_metrics,
)


def evaluate_prediction(gold_answer, raw_output, task_type):
    if task_type == "fact_verification":
        gold = gold_answer[0] if isinstance(gold_answer, list) and gold_answer else gold_answer
        return cal_tabfact_score(raw_output, gold)

    gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    precision, recall, f1 = cal_score3(gold_list, raw_output)
    return precision, recall, f1


def validate_content_type(value):
    valid_values = ["question", "question_template", "question_sentence", "markdown_table"]
    return value if value in valid_values else "question"


def build_gold_table_key(conv):
    db_id = conv.get("db_id")
    highlighted = conv.get("highlighted_table") or []
    if not db_id or not highlighted:
        return None
    return build_table_key(db_id, highlighted[0])


def gold_table_in_first_qa(retrieved_keys, gold_key):
    if gold_key is None:
        return 0
    return int(gold_key in set(retrieved_keys))


def gold_table_in_qa(retrieved_keys, fallback_keys, gold_key):
    if gold_key is None:
        return 0
    qa_tables = set(retrieved_keys) | set(fallback_keys)
    return int(gold_key in qa_tables)


def _process_one(
    idx,
    conv,
    content_type,
    table_store,
    all_table_keys,
    top_k,
    n_candidates,
    preview_chars,
    max_matched_rows,
    max_rows_per_pattern,
    max_line_chars,
    max_steps,
    max_select_tables,
    task_type,
    llm_base_url,
    llm_api_key,
    llm_model,
):
    try:
        query = conv[content_type]
        gold_answer = conv.get("answer", [])

        result = multi_round_retrieve_and_qa(
            query=query,
            table_store=table_store,
            all_table_keys=all_table_keys,
            task_type=task_type,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            top_k=top_k,
            n_candidates=n_candidates,
            preview_chars=preview_chars,
            max_matched_rows=max_matched_rows,
            max_rows_per_pattern=max_rows_per_pattern,
            max_line_chars=max_line_chars,
            max_steps=max_steps,
            max_select_tables=max_select_tables,
        )

        precision, recall, f1 = evaluate_prediction(
            gold_answer, result["raw_output"], task_type
        )
        gold_table_key = build_gold_table_key(conv)
        qa_first_gold_table_hit = gold_table_in_first_qa(
            result["retrieved_keys"], gold_table_key
        )
        qa_gold_table_hit = gold_table_in_qa(
            result["retrieved_keys"], result["fallback_keys"], gold_table_key
        )

        output_dict = {
            **conv,
            "retrieved_keys": result["retrieved_keys"],
            "retrieved_scores": result["retrieved_scores"],
            "llm_patterns": result["llm_patterns"],
            "fallback_keys": result["fallback_keys"],
            "steps_used": result["steps_used"],
            "task_type": task_type,
            "output_text": result["raw_output"],
            "gold_answer": gold_answer,
            "gold_table_key": gold_table_key,
            "qa_first_gold_table_hit": qa_first_gold_table_hit,
            "qa_gold_table_hit": qa_gold_table_hit,
            "qa_precision": precision,
            "qa_recall": recall,
            "qa_f1": f1,
            "step_log": result["step_log"],
            "action_counts": result["action_counts"],
        }

    except Exception:
        logging.error(traceback.format_exc())
        output_dict = {
            **conv,
            "retrieved_keys": [],
            "retrieved_scores": [],
            "llm_patterns": [],
            "fallback_keys": [],
            "steps_used": 0,
            "task_type": task_type,
            "output_text": "error",
            "gold_answer": conv.get("answer", []),
            "gold_table_key": build_gold_table_key(conv),
            "qa_first_gold_table_hit": 0,
            "qa_gold_table_hit": 0,
            "qa_precision": 0.0,
            "qa_recall": 0.0,
            "qa_f1": 0.0,
            "step_log": [],
            "action_counts": {
                "pattern_gen": 0,
                "refine_select": 0,
                "refine_regenerate": 0,
                "refine_none": 0,
                "qa": 0,
                "qa_fallback": 0,
            },
        }

    return idx, output_dict


def evaluate(
    output_path,
    all_data,
    content_type,
    table_format,
    tables_path,
    tables_json_path,
    top_k,
    task_type="tableqa",
    num_workers=8,
    n_candidates=5,
    preview_chars=800,
    max_matched_rows=12,
    max_rows_per_pattern=3,
    max_line_chars=400,
    max_steps=6,
    max_select_tables=3,
    llm_base_url="http://127.0.0.1:8004/v1",
    llm_api_key="xxx",
    llm_model="Qwen/Qwen3-30B-A3B-Thinking-2507",
):
    start_time = time.time()
    logging.info("output_path: %s", output_path)
    logging.info("Loading tables from: %s", tables_json_path)

    table_store = load_all_tables(tables_json_path, tables_path, table_format)
    all_table_keys = list(table_store.keys())
    logging.info("Loaded %d tables.", len(all_table_keys))

    os.makedirs(output_path, exist_ok=True)
    results = [None] * len(all_data)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                idx,
                conv,
                content_type,
                table_store,
                all_table_keys,
                top_k,
                n_candidates,
                preview_chars,
                max_matched_rows,
                max_rows_per_pattern,
                max_line_chars,
                max_steps,
                max_select_tables,
                task_type,
                llm_base_url,
                llm_api_key,
                llm_model,
            ): idx
            for idx, conv in enumerate(all_data)
        }

        with tqdm.tqdm(total=len(all_data), desc="TableSeek Retrieve+QA") as pbar:
            for future in as_completed(futures):
                idx, output_dict = future.result()
                results[idx] = output_dict
                pbar.update(1)

    output_jsonl = os.path.join(output_path, "output.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(output_path, "output.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    patterns_records = [
        {
            "idx": i,
            "query": item.get(content_type, ""),
            "llm_patterns": item.get("llm_patterns", []),
            "steps_used": item.get("steps_used", 0),
            "fallback_keys": item.get("fallback_keys", []),
            "step_log": item.get("step_log", []),
            "action_counts": item.get("action_counts", {}),
        }
        for i, item in enumerate(results)
    ]
    patterns_jsonl = os.path.join(output_path, "patterns.jsonl")
    patterns_json = os.path.join(output_path, "patterns.json")
    with open(patterns_jsonl, "w", encoding="utf-8") as f:
        for rec in patterns_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(patterns_json, "w", encoding="utf-8") as f:
        json.dump(patterns_records, f, ensure_ascii=False, indent=4)
    logging.info("Saved patterns (with step_log) to %s and %s", patterns_jsonl, patterns_json)

    elapsed_hours = (time.time() - start_time) / 3600.0
    _finalize_output(output_path, results, elapsed_hours=elapsed_hours)


def _finalize_output(output_path, results, elapsed_hours=0.0):
    valid = [r for r in results if r is not None]
    n = len(valid)
    if n == 0:
        logging.warning("No valid results to compute metrics.")
        return

    avg_precision = sum(r["qa_precision"] for r in valid) / n
    avg_recall = sum(r["qa_recall"] for r in valid) / n
    avg_f1 = sum(r["qa_f1"] for r in valid) / n
    avg_first_gold_hit = sum(r.get("qa_first_gold_table_hit", 0) for r in valid) / n
    avg_gold_hit = sum(r.get("qa_gold_table_hit", 0) for r in valid) / n
    avg_steps = sum(r.get("steps_used", 0) for r in valid) / n

    task_type = valid[0].get("task_type", "tableqa")
    if task_type == "fact_verification":
        avg_precision, avg_recall, avg_f1 = cal_tabfact_aggregate_metrics(valid)

    metrics = {
        "data_num": n,
        "qa_precision": avg_precision,
        "qa_recall": avg_recall,
        "qa_f1": avg_f1,
        "qa_first_gold_table_hit": avg_first_gold_hit,
        "qa_gold_table_hit": avg_gold_hit,
        "avg_steps": avg_steps,
        "elapsed_hours": elapsed_hours,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    csv_path = os.path.join(output_path, "result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logging.info("[%s] QA metrics: %s", output_path, metrics)

    action_keys = [
        "pattern_gen",
        "refine_select",
        "refine_regenerate",
        "refine_none",
        "qa",
        "qa_fallback",
    ]
    total_counts = {k: 0 for k in action_keys}
    for r in valid:
        for k in action_keys:
            total_counts[k] += r.get("action_counts", {}).get(k, 0)

    avg_action_counts = {k: total_counts[k] / n for k in action_keys}

    action_stats = {
        "data_num": n,
        "total_action_counts": total_counts,
        "avg_action_counts_per_sample": avg_action_counts,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    action_stats_json = os.path.join(output_path, "action_stats.json")
    with open(action_stats_json, "w", encoding="utf-8") as f:
        json.dump(action_stats, f, ensure_ascii=False, indent=4)

    action_stats_csv = os.path.join(output_path, "action_stats.csv")
    with open(action_stats_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["action", "total_count", "avg_per_sample"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in action_keys:
            writer.writerow(
                {
                    "action": k,
                    "total_count": total_counts[k],
                    "avg_per_sample": avg_action_counts[k],
                }
            )

    logging.info("[%s] Action stats: %s", output_path, action_stats)


def main():
    parser = argparse.ArgumentParser(description="TableSeek: multi-round table retrieval + QA")
    parser.add_argument("--json-path", type=str, default="data/query_data/TCRAG_Questions.json")
    parser.add_argument("--tables_path", type=str, default="data/table/TCRAG_8k")
    parser.add_argument("--tables_json_path", type=str, default="data/query_data/TCRAG_Tables.json")
    parser.add_argument("--content_type", type=validate_content_type, default="question")
    parser.add_argument(
        "--table_format",
        type=str,
        default="default",
        choices=[
            "markdown",
            "html",
            "csv",
            "default",
            "sentence",
            "sentence_shuffled",
            "json",
            "latex",
            "xml",
        ],
    )
    parser.add_argument("--output-path", type=str, default="results/TCR")
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Fallback table count when refine returns no selection.",
    )
    parser.add_argument(
        "--max-select-tables",
        type=int,
        default=1,
        help="Max tables the LLM may select during refine (1 to this value).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=5,
        help="Number of top tables shown to LLM for refine decision.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Characters of each candidate table head shown to LLM during refine.",
    )
    parser.add_argument(
        "--max-matched-rows",
        type=int,
        default=5,
        help="Max pattern-matched rows appended per candidate table in refine.",
    )
    parser.add_argument(
        "--max-rows-per-pattern",
        type=int,
        default=1,
        help="Max matched rows contributed by each search pattern in refine.",
    )
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=400,
        help="Truncate individual matched rows longer than this in refine.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Max LLM calls per query (pattern generation + refine rounds + QA).",
    )
    parser.add_argument("--task-type", type=str, default="tableqa", choices=TASK_TYPES)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:8004/v1")
    parser.add_argument("--llm-api-key", type=str, default="xxx")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
    )
    args = parser.parse_args()

    setup_logging(args.output_path)
    logging.info(vars(args))

    all_data = json.load(open(args.json_path, encoding="utf-8"))
    os.makedirs(args.output_path, exist_ok=True)

    evaluate(
        output_path=args.output_path,
        all_data=all_data,
        content_type=args.content_type,
        table_format=args.table_format,
        tables_path=args.tables_path,
        tables_json_path=args.tables_json_path,
        top_k=args.top_k,
        task_type=args.task_type,
        num_workers=args.num_workers,
        n_candidates=args.n_candidates,
        preview_chars=args.preview_chars,
        max_matched_rows=args.max_matched_rows,
        max_rows_per_pattern=args.max_rows_per_pattern,
        max_line_chars=args.max_line_chars,
        max_steps=args.max_steps,
        max_select_tables=args.max_select_tables,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
