"""
Multi-round table retrieval + QA.

Round structure for each query (bounded by max_steps):
  Step 1: Generate initial patterns, rank all tables, take top-n_candidates.
  Step 2: Show LLM the top-n_candidates previews; LLM either
            (a) selects up to max_select_tables tables -> run QA on all selected tables, done
            (b) regenerates patterns -> re-rank, repeat from step 2
  If QA output is empty, fall through to the next-best unselected candidate and retry QA one table at a time.
  Stop when max_steps is exhausted or a non-empty answer is found.

  NOTE: at least one QA step is always guaranteed regardless of max_steps,
  because the refine loop reserves the final step for QA.
"""

import logging

from tableseek.patterns import (
    build_refine_preview,
    llm_generate_patterns,
    llm_refine_decision,
    rank_tables_by_patterns,
)
from tableseek.qa import (
    build_table_context,
    llm_qa,
    llm_tabfact,
    is_empty_output,
    is_tabfact_mismatch,
    finalize_tabfact_output,
)


def _get_preview(
    table_store: dict,
    key: str,
    preview_chars: int,
    patterns: list,
    max_matched_rows: int,
    max_rows_per_pattern: int,
    max_line_chars: int,
) -> str:
    table_text = table_store.get(key, {}).get('table_str', '')
    return build_refine_preview(
        table_text,
        patterns,
        preview_chars,
        max_matched_rows=max_matched_rows,
        max_rows_per_pattern=max_rows_per_pattern,
        max_line_chars=max_line_chars,
    )


def _run_task_llm(
    query: str,
    table_context: str,
    task_type: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> str:
    if task_type == 'fact_verification':
        return llm_tabfact(query, table_context, llm_base_url, llm_api_key, llm_model)
    return llm_qa(query, table_context, llm_base_url, llm_api_key, llm_model)


def _top5_from_ranked(ranked: list) -> dict:
    top5 = ranked[:5]
    return {
        "top5_keys": [k for k, _ in top5],
        "top5_scores": [s for _, s in top5],
    }


def _is_bad_output(raw_output: str, task_type: str) -> bool:
    if task_type == 'fact_verification':
        return is_tabfact_mismatch(raw_output)
    return is_empty_output(raw_output)


def _resolve_selected_keys(
    decision: dict,
    anon_id_to_key: dict,
    ranked: list,
    max_select_tables: int,
) -> list:
    """Map anonymous table IDs from refine decision to real table keys (1-max_select_tables)."""
    selected_ids = decision.get("selected_ids") or []
    if not selected_ids and decision.get("selected_id"):
        selected_ids = [decision["selected_id"]]

    selected_keys = []
    unknown_ids = []
    for selected_id in selected_ids:
        if len(selected_keys) >= max_select_tables:
            break
        key = anon_id_to_key.get(selected_id)
        if key is None:
            unknown_ids.append(selected_id)
            continue
        if key not in selected_keys:
            selected_keys.append(key)

    if unknown_ids:
        logging.warning(
            f"LLM returned unknown anonymous IDs {unknown_ids}, ignoring them"
        )
    if not selected_keys:
        logging.warning("LLM select action had no valid IDs, falling back to top-1")
        selected_keys = [ranked[0][0]] if ranked else []
    return selected_keys


def multi_round_retrieve_and_qa(
    query: str,
    table_store: dict,
    all_table_keys: list,
    task_type: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    top_k: int = 1,
    n_candidates: int = 5,
    preview_chars: int = 800,
    max_matched_rows: int = 12,
    max_rows_per_pattern: int = 3,
    max_line_chars: int = 400,
    max_steps: int = 6,
    max_select_tables: int = 3,
) -> dict:
    """
    Execute multi-round retrieval + QA for a single query.

    Parameters
    ----------
    top_k         : final number of tables to pass into the QA context
    n_candidates  : number of top tables to show LLM during the refine round
                    (intentionally different from top_k)
    preview_chars : characters of each candidate table head shown to LLM
    max_matched_rows    : max grep-matched rows appended per candidate table
    max_rows_per_pattern: max matched rows contributed by each regex pattern
    max_line_chars      : truncate individual matched rows longer than this
    max_steps     : hard cap on total LLM calls (pattern-gen + refine + QA each count).
                    At least 1 step is always reserved for QA regardless of this value.
    max_select_tables : max tables the LLM may select during refine (1 to this value).

    Returns
    -------
    dict with keys: retrieved_keys, retrieved_scores, llm_patterns, raw_output,
                    steps_used, fallback_keys, step_log, action_counts
    """
    steps_used = 0
    patterns = []
    ranked = []
    selected_keys = []
    fallback_keys = []
    raw_output = ""

    # step_log: list of dicts, one per LLM call, recording what happened
    step_log = []
    # action_counts: tally of each operation type across this query
    action_counts = {
        "pattern_gen":  0,
        "refine_select": 0,
        "refine_regenerate": 0,
        "refine_none": 0,
        "qa": 0,
        "qa_fallback": 0,
    }

    # ── Step 1: initial pattern generation ──────────────────────────────────
    if steps_used >= max_steps:
        return _empty_result(query, patterns, steps_used)

    patterns = llm_generate_patterns(
        query, task_type, llm_base_url, llm_api_key, llm_model
    )
    steps_used += 1
    action_counts["pattern_gen"] += 1
    ranked = rank_tables_by_patterns(table_store, all_table_keys, patterns)
    step_log.append({
        "step": steps_used,
        "action": "pattern_gen",
        "patterns": patterns,
        **_top5_from_ranked(ranked),
    })

    # ── Refine loop ──────────────────────────────────────────────────────────
    # Reserve the last step for QA: loop only while at least 2 steps remain
    # (1 for this refine call + 1 for the mandatory QA call that follows).
    while steps_used < max_steps - 1:
        candidates = ranked[:n_candidates]

        # Build anonymous ID mapping: Table_0, Table_1, ... -> real key
        # Real keys must never be sent to the LLM (they may leak the answer).
        anon_id_to_key = {f"Table_{i}": key for i, (key, _) in enumerate(candidates)}
        anon_previews = [
            (
                anon_id,
                _get_preview(
                    table_store,
                    real_key,
                    preview_chars,
                    patterns,
                    max_matched_rows,
                    max_rows_per_pattern,
                    max_line_chars,
                ),
            )
            for anon_id, real_key in anon_id_to_key.items()
        ]

        decision = llm_refine_decision(
            query=query,
            task_type=task_type,
            candidate_previews=anon_previews,
            current_patterns=patterns,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            max_select_tables=max_select_tables,
        )
        steps_used += 1

        action = decision.get("action", "none")

        if action == "select":
            selected_ids = decision.get("selected_ids") or []
            if not selected_ids and decision.get("selected_id"):
                selected_ids = [decision["selected_id"]]
            selected_keys = _resolve_selected_keys(
                decision, anon_id_to_key, ranked, max_select_tables
            )
            action_counts["refine_select"] += 1
            step_log.append({
                "step": steps_used,
                "action": "refine_select",
                "selected_anon_ids": selected_ids,
                "selected_keys": selected_keys,
                "candidates_shown": list(anon_id_to_key.keys()),
            })
            break

        elif action == "regenerate":
            new_patterns = decision.get("patterns", [])
            if new_patterns:
                patterns = new_patterns
                ranked = rank_tables_by_patterns(table_store, all_table_keys, patterns)
            else:
                logging.warning("LLM returned empty patterns on regenerate; keeping previous ranking")
            action_counts["refine_regenerate"] += 1
            step_log.append({
                "step": steps_used,
                "action": "refine_regenerate",
                "new_patterns": new_patterns,
                **_top5_from_ranked(ranked),
            })
            # Continue loop

        else:
            # 'none' or unrecognised: just pick the top-1
            action_counts["refine_none"] += 1
            step_log.append({
                "step": steps_used,
                "action": "refine_none",
                "raw_action": action,
            })
            break

    # ── Determine final retrieval list ───────────────────────────────────────
    if selected_keys:
        final_keys = selected_keys
    else:
        final_keys = [k for k, _ in ranked[:top_k]]

    retrieved_keys   = final_keys
    retrieved_scores = [
        next((s for k, s in ranked if k == key), 0.0)
        for key in final_keys
    ]

    # ── QA with fallback on empty / mismatch output ───────────────────────────
    table_context = build_table_context(retrieved_keys, table_store)
    raw_output = _run_task_llm(
        query, table_context, task_type, llm_base_url, llm_api_key, llm_model
    )
    steps_used += 1
    action_counts["qa"] += 1
    step_log.append({
        "step": steps_used,
        "action": "qa",
        "retrieved_keys": retrieved_keys,
        "output_empty": _is_bad_output(raw_output, task_type),
    })

    # If output is empty (or tabfact mismatch), try unselected candidates one by one.
    used_keys = set(retrieved_keys)
    remaining_candidates = [k for k, _ in ranked if k not in used_keys]
    for fallback_key in remaining_candidates:
        if not _is_bad_output(raw_output, task_type):
            break
        if steps_used >= max_steps:
            break

        fallback_keys.append(fallback_key)
        logging.info(
            f"QA output unsatisfactory, falling back to next unselected candidate: {fallback_key}"
        )
        fallback_context = build_table_context([fallback_key], table_store)
        raw_output = _run_task_llm(
            query, fallback_context, task_type, llm_base_url, llm_api_key, llm_model
        )
        steps_used += 1
        action_counts["qa_fallback"] += 1
        step_log.append({
            "step": steps_used,
            "action": "qa_fallback",
            "fallback_key": fallback_key,
            "output_empty": _is_bad_output(raw_output, task_type),
        })

    # If TabFact still ends as mismatch, treat as unsupported (false).
    if task_type == 'fact_verification':
        raw_output = finalize_tabfact_output(raw_output)

    return {
        "retrieved_keys":   retrieved_keys,
        "retrieved_scores": retrieved_scores,
        "llm_patterns":     patterns,
        "raw_output":       raw_output,
        "steps_used":       steps_used,
        "fallback_keys":    fallback_keys,
        "step_log":         step_log,
        "action_counts":    action_counts,
    }


def _empty_result(query: str, patterns: list, steps_used: int) -> dict:
    return {
        "retrieved_keys":   [],
        "retrieved_scores": [],
        "llm_patterns":     patterns,
        "raw_output":       "",
        "steps_used":       steps_used,
        "fallback_keys":    [],
        "step_log":         [],
        "action_counts":    {
            "pattern_gen":       0,
            "refine_select":     0,
            "refine_regenerate": 0,
            "refine_none":       0,
            "qa":                0,
            "qa_fallback":       0,
        },
    }