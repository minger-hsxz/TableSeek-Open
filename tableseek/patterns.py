import re
import json
import logging
import threading
from typing import List, Optional
from openai import OpenAI

QUERY_PREFIX_MAP = {
    'tableqa': (
        'Given a TableQA query, retrieve a relevant table that can answer the query. '
        'Note that the retrieved table should contain sufficient information to provide an answer, '
        'rather than resulting in an empty answer. '
    ),
    'fact_verification': (
        'Given a claim about tabular data, retrieve a relevant table whose schema and content '
        'can be used to verify the claim. '
        'Focus on tables whose column names and row structure are relevant to the claim, '
        'even if an exact answer cannot be directly read off. '
    ),
}

TASK_TYPES = list(QUERY_PREFIX_MAP.keys())

_llm_client: Optional[OpenAI] = None
_llm_client_lock = threading.Lock()


def get_llm_client(base_url: str, api_key: str) -> OpenAI:
    global _llm_client
    with _llm_client_lock:
        if _llm_client is None:
            _llm_client = OpenAI(base_url=base_url, api_key=api_key)
    return _llm_client


# ─── Initial pattern generation ─────────────────────────────────────────────────

GREP_SYSTEM_PROMPT = """\
You are a table retrieval assistant. Your job is to identify which tables are relevant to a user query by generating regex patterns (like grep) to match against table text strings.

You will be given:
1. A natural language query
2. A list of table keys (identifiers)

Your task:
- Analyse the query carefully
- Generate a list of Python-compatible regex patterns (case-insensitive) that, when matched against a table's text representation, would indicate that table is relevant
- Patterns should target: column names, entity names, key terms, or values likely present in a relevant table
- Avoid overly generic patterns (e.g. single common words)
- Prefer specific noun phrases, entity names, numeric identifiers, or domain-specific column names

Output format — respond with a JSON object (no markdown fences, no extra text):
{
  "patterns": ["pattern1", "pattern2", ...]
}

The patterns will be applied with re.search(..., table_text, re.IGNORECASE).
Generate between 3 and 10 patterns.
"""

# ─── Multi-round refinement prompt ──────────────────────────────────────────────

REFINE_SYSTEM_PROMPT_OLD = """\
You are a table retrieval assistant helping refine a search.

You will be given:
1. The original query
2. The top candidate table previews (first N characters of each table), each labelled with an anonymous ID like "Table_0", "Table_1", etc.
3. The current regex patterns used

Your task: Decide whether to:
  A) Select one of the candidate tables as the answer (if one clearly matches)
  B) Generate a new set of regex patterns to search again (if the candidates all look wrong or too similar)

Output format — respond with a JSON object (no markdown fences, no extra text).

If selecting a table:
{
  "action": "select",
  "selected_id": "<anonymous ID of the selected table, e.g. Table_0>"
}

If regenerating patterns:
{
  "action": "regenerate",
  "patterns": ["pattern1", "pattern2", ...]
}

Be decisive. If any candidate table plausibly contains the answer, select it.
Generate between 3 and 10 patterns if regenerating.
"""

def _select_range_text(max_select_tables: int) -> str:
    max_select_tables = max(1, max_select_tables)
    if max_select_tables == 1:
        return "exactly 1 table"
    return f"1 to {max_select_tables} tables"


def _build_refine_system_prompt(max_select_tables: int) -> str:
    max_select_tables = max(1, max_select_tables)
    select_range = _select_range_text(max_select_tables)
    if max_select_tables == 1:
        select_guidance = (
            "You may select **exactly 1 table**. "
            "If a single table clearly contains the data needed to answer the query, select it immediately."
        )
        output_select_hint = "If selecting one table (1 anonymous ID):"
        example_ids = '["Table_0"]'
    else:
        select_guidance = (
            f"You may select **{select_range}** when:\n"
            "  - the query requires information from multiple tables (e.g. join / compare / aggregate across tables), OR\n"
            "  - several tables look equally plausible and you cannot confidently pick just one.\n"
            "  If a single table is clearly sufficient, select only that one table."
        )
        output_select_hint = f"If selecting one or more tables (1 to {max_select_tables} anonymous IDs):"
        example_ids = '["Table_0", "Table_1"]'

    return f"""\
You are an expert table retrieval assistant helping to iteratively refine a search based on table previews.
You will be given:
1. The original query
2. The top candidate table previews, labeled with anonymous IDs like "Table_0", "Table_1", etc.
   Each preview may include:
   - the beginning of the table (truncated if large)
   - an optional section "Rows matched by current grep patterns" listing rows/lines that matched the regex patterns (grep is row-oriented even though retrieval ranks whole tables)
3. The current regex patterns used

### Heuristic Guidelines for Decision Making:
- **Select (High Confidence):** If one or more candidate tables clearly contain the data needed to answer the query, select them immediately. {select_guidance}
- **Regenerate by Value Search (Schema Collision):** If the top K candidate tables share very similar structures/schemas (e.g., same column headers), but none of their visible values match the query's target entity, DO NOT select randomly. Instead, generate new regex patterns focusing on **specific values, keywords, or entities** from the original query to filter out the correct table.
- **Regenerate by Schema Shift (Irrelevant Results):** If all candidate tables are completely irrelevant (wrong domain, wrong format, or totally different context), your previous schema-based regex has failed. Generate new regex patterns exploring **alternative schemas, synonyms, or different table structures** that might exist in the database.
- **Regenerate by Broadening (Narrow Misses):** If the tables are in the right domain but the format is slightly off, generate broader or fuzzy regex patterns to catch variations.

### Output Format:
Output format — respond with a JSON object (no markdown fences, no extra text).
{output_select_hint}
{{
  "action": "select",
  "selected_ids": {example_ids}
}}
If regenerating patterns (generate between 3 and 10 patterns):
{{
  "action": "regenerate",
  "patterns": ["pattern1", "pattern2", "pattern3"]
}}
Be decisive and strategic. Analyze the gap between the query and the candidates to choose the best regeneration strategy.
"""


def _build_grep_user_prompt(query: str, task_type: str) -> str:
    task_hint = QUERY_PREFIX_MAP.get(task_type, QUERY_PREFIX_MAP['tableqa'])
    return (
        f"Task context: {task_hint}\n\n"
        f"Query: {query}\n\n"
        "Generate regex patterns to grep relevant tables."
    )


_TRUNCATION_MARKER = "[... table truncated ...]"
_GREP_MATCHES_HEADER = "--- Rows matched by current grep patterns ---"


def _truncate_line(line: str, max_line_chars: int) -> str:
    if max_line_chars <= 0 or len(line) <= max_line_chars:
        return line
    return line[:max_line_chars] + "..."


def _is_skippable_grep_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    # Markdown table separator rows, e.g. |---|---|
    if re.match(r"^[\|\s\-:]+$", stripped):
        return True
    return False


def extract_grep_matched_lines(
    table_text: str,
    patterns: list,
    max_rows_per_pattern: int = 3,
    max_total_rows: int = 12,
    max_line_chars: int = 400,
) -> List[str]:
    """
    Collect table lines that match grep patterns, with per-pattern and global caps.
    Patterns are applied row-wise (line-wise) against the table text representation.
    """
    if not patterns or not table_text:
        return []

    lines = table_text.splitlines()
    matched = []  # type: List[str]
    seen = set()  # type: set

    for pat in patterns:
        pattern_hits = 0
        for line in lines:
            if pattern_hits >= max_rows_per_pattern or len(matched) >= max_total_rows:
                break
            if _is_skippable_grep_line(line) or line in seen:
                continue
            if _safe_search(pat, line):
                seen.add(line)
                matched.append(_truncate_line(line, max_line_chars))
                pattern_hits += 1

        if len(matched) >= max_total_rows:
            break

    return matched


def build_refine_preview(
    table_text: str,
    patterns: list,
    preview_chars: int,
    max_matched_rows: int = 12,
    max_rows_per_pattern: int = 3,
    max_line_chars: int = 400,
) -> str:
    """
    Build a refine-round preview: truncated table head plus grep-matched rows
    not already visible in the head section.
    """
    if not table_text:
        return ""

    head = table_text[:preview_chars]
    truncated = len(table_text) > preview_chars

    matched_lines = extract_grep_matched_lines(
        table_text,
        patterns,
        max_rows_per_pattern=max_rows_per_pattern,
        max_total_rows=max_matched_rows,
        max_line_chars=max_line_chars,
    )
    extra_lines = [ln for ln in matched_lines if ln not in head]

    parts = [head]
    if truncated:
        parts.append(_TRUNCATION_MARKER)
    if extra_lines:
        parts.append(_GREP_MATCHES_HEADER)
        parts.extend(extra_lines)

    return "\n".join(parts)


def _build_refine_user_prompt(
    query: str,
    task_type: str,
    candidate_previews: list,
    current_patterns: list,
    max_select_tables: int = 3,
) -> str:
    """
    candidate_previews: list of (anon_id, preview_text).
    Real table keys must NOT appear here.
    """
    max_select_tables = max(1, max_select_tables)
    select_range = _select_range_text(max_select_tables)
    task_hint = QUERY_PREFIX_MAP.get(task_type, QUERY_PREFIX_MAP['tableqa'])
    previews_str = "\n\n".join(
        f"[{anon_id}]\n{preview}" for anon_id, preview in candidate_previews
    )
    if max_select_tables == 1:
        select_instruction = (
            "Decide: select exactly 1 table (by its anonymous ID) or regenerate patterns."
        )
    else:
        select_instruction = (
            f"Decide: select {select_range} (by their anonymous IDs) or regenerate patterns. "
            f"If the query needs multiple tables, or you truly cannot decide among plausible candidates, "
            f"select up to {max_select_tables} tables."
        )
    return (
        f"Task context: {task_hint}\n\n"
        f"Query: {query}\n\n"
        f"Current patterns: {json.dumps(current_patterns)}\n\n"
        f"Top candidate table previews:\n{previews_str}\n\n"
        f"{select_instruction}"
    )


def _extract_patterns_from_response(raw_text: str) -> list:
    think_end = raw_text.rfind("</think>")
    answer_text = raw_text[think_end + len("</think>"):].strip() if think_end != -1 else raw_text.strip()
    answer_text = re.sub(r"^```(?:json)?\s*", "", answer_text.strip())
    answer_text = re.sub(r"\s*```$", "", answer_text.strip())

    try:
        data = json.loads(answer_text)
        patterns = data.get("patterns", [])
        if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
            return patterns
    except json.JSONDecodeError:
        pass

    fallback = re.findall(r'"([^"]{2,})"', answer_text)
    logging.warning(f"JSON parse failed; falling back to regex extraction, found {len(fallback)} patterns")
    return fallback[:10]


def _extract_refine_decision(raw_text: str) -> dict:
    think_end = raw_text.rfind("</think>")
    answer_text = raw_text[think_end + len("</think>"):].strip() if think_end != -1 else raw_text.strip()
    answer_text = re.sub(r"^```(?:json)?\s*", "", answer_text.strip())
    answer_text = re.sub(r"\s*```$", "", answer_text.strip())

    try:
        data = json.loads(answer_text)
        action = data.get("action")
        if action == "select":
            selected_ids = data.get("selected_ids")
            if not isinstance(selected_ids, list):
                single_id = data.get("selected_id")
                selected_ids = [single_id] if single_id else []
            data["selected_ids"] = [
                sid for sid in selected_ids
                if isinstance(sid, str) and sid.strip()
            ]
            return data
        if action == "regenerate":
            return data
    except json.JSONDecodeError:
        pass

    logging.warning("Failed to parse refine decision, defaulting to no-op")
    return {"action": "none"}


def llm_generate_patterns(
    query: str,
    task_type: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    max_retries: int = 3,
) -> list:
    client = get_llm_client(llm_base_url, llm_api_key)
    user_msg = _build_grep_user_prompt(query, task_type)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": GREP_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8192,
                timeout=18000,
                top_p=1.0,
                seed=42,
            )
            raw = response.choices[0].message.content or ""
            patterns = _extract_patterns_from_response(raw)
            logging.info(f"LLM grep patterns for [{query[:60]}]: {patterns}")
            return patterns
        except Exception as e:
            last_error = e
            logging.warning(f"LLM grep call attempt {attempt}/{max_retries} failed: {e}")

    logging.error(f"All {max_retries} grep LLM attempts failed for [{query[:60]}]: {last_error}")
    return []


def llm_refine_decision(
    query: str,
    task_type: str,
    candidate_previews: list,
    current_patterns: list,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    max_select_tables: int = 3,
    max_retries: int = 3,
) -> dict:
    """
    Ask LLM to either select a table from candidates or regenerate patterns.
    Returns dict with 'action' in ('select', 'regenerate', 'none').
    """
    max_select_tables = max(1, max_select_tables)
    client = get_llm_client(llm_base_url, llm_api_key)
    user_msg = _build_refine_user_prompt(
        query, task_type, candidate_previews, current_patterns, max_select_tables
    )
    system_msg = _build_refine_system_prompt(max_select_tables)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8192,
                timeout=18000,
                top_p=1.0,
                seed=42,
            )
            raw = response.choices[0].message.content or ""
            decision = _extract_refine_decision(raw)
            logging.info(f"LLM refine decision for [{query[:60]}]: {decision}")
            logging.info(f"LLM refine raw output for [{user_msg}]: {raw}")
            return decision
        except Exception as e:
            last_error = e
            logging.warning(f"LLM refine call attempt {attempt}/{max_retries} failed: {e}")

    logging.error(f"All {max_retries} refine LLM attempts failed for [{query[:60]}]: {last_error}")
    return {"action": "none"}


def score_table_by_patterns(table_text: str, patterns: list) -> float:
    if not patterns:
        return 0.0
    hits = sum(
        1 for pat in patterns
        if _safe_search(pat, table_text)
    )
    return hits / len(patterns)


def _safe_search(pat: str, text: str) -> bool:
    try:
        return bool(re.search(pat, text, re.IGNORECASE))
    except re.error:
        return False


def rank_tables_by_patterns(table_store: dict, all_table_keys: list, patterns: list) -> list:
    """Return sorted list of (key, score) by descending score."""
    scores = {
        key: score_table_by_patterns(table_store[key].get('table_str', ''), patterns)
        for key in all_table_keys
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)