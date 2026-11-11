"""Map provider metric names to standard metric codes (pure functions, no I/O).

Rules
-----
1. Names are matched through :func:`config.metric_aliases.lookup_metric`
   (normalised, statement-specific).
2. Several aliases of one metric for the same period:
   * all non-missing values equal  -> one MAPPED record (duplicates counted)
   * values differ                 -> the highest-precedence alias wins, the record
                                      is marked CONFLICT_RESOLVED and the conflict
                                      is reported. Values are never summed/averaged.
   * the preferred alias is missing but another has a value -> the available
     value is used (missing is not a value).
3. Unknown names are kept as ``UNMAPPED:<original name>`` with their raw value.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.database.models import MAPPING_CONFLICT, MAPPING_MAPPED, MAPPING_UNMAPPED
from config.metric_aliases import lookup_metric, unmapped_code


@dataclass(frozen=True)
class RawInput:
    """The fields of a raw fact the standardiser needs."""

    raw_fact_id: int | None
    statement_type: str
    original_metric_name: str
    fiscal_period_end: date
    original_value: Decimal | None


@dataclass
class StandardizedRecord:
    statement_type: str
    metric_code: str
    metric_name: str
    fiscal_period_end: date
    value: Decimal | None
    source_metric_name: str
    mapping_status: str
    raw_fact_id: int | None
    messages: list[str] = field(default_factory=list)


@dataclass
class StandardizationResult:
    records: list[StandardizedRecord] = field(default_factory=list)
    unmapped_names: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    duplicate_alias_count: int = 0


def standardize(raw_facts: Iterable[RawInput]) -> StandardizationResult:
    result = StandardizationResult()
    groups: dict[tuple[str, str, date], list[tuple[int, RawInput]]] = defaultdict(list)
    definitions = {}
    unmapped: set[str] = set()

    for fact in raw_facts:
        match = lookup_metric(fact.statement_type, fact.original_metric_name)
        if match is None:
            unmapped.add(f"{fact.statement_type}: {fact.original_metric_name}")
            result.records.append(StandardizedRecord(
                statement_type=fact.statement_type,
                metric_code=unmapped_code(fact.original_metric_name),
                metric_name=fact.original_metric_name[:200],
                fiscal_period_end=fact.fiscal_period_end, value=fact.original_value,
                source_metric_name=fact.original_metric_name[:200],
                mapping_status=MAPPING_UNMAPPED, raw_fact_id=fact.raw_fact_id,
                messages=["No standard mapping; original name and value preserved."]))
            continue
        definition, precedence = match
        definitions[definition.code] = definition
        groups[(fact.statement_type, definition.code, fact.fiscal_period_end)].append(
            (precedence, fact))

    for (statement_type, code, period_end), candidates in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        definition = definitions[code]
        candidates.sort(key=lambda pair: pair[0])
        with_values = [(p, f) for p, f in candidates if f.original_value is not None]
        chosen = with_values[0][1] if with_values else candidates[0][1]
        status = MAPPING_MAPPED
        messages: list[str] = []
        distinct = {f.original_value for _, f in with_values}
        if len(candidates) > 1:
            if len(distinct) > 1:
                status = MAPPING_CONFLICT
                detail = ", ".join(f"{f.original_metric_name}={f.original_value}"
                                   for _, f in with_values)
                message = (f"Mapping conflict for {code} {period_end}: {detail}. Used "
                           f"'{chosen.original_metric_name}' by alias precedence.")
                messages.append(message)
                result.conflicts.append(f"{statement_type}: {message}")
            else:
                result.duplicate_alias_count += len(candidates) - 1
                names = ", ".join(f.original_metric_name for _, f in candidates)
                messages.append(f"Duplicate aliases agree ({names}); stored once.")
        result.records.append(StandardizedRecord(
            statement_type=statement_type, metric_code=code, metric_name=definition.name,
            fiscal_period_end=period_end, value=chosen.original_value,
            source_metric_name=chosen.original_metric_name[:200], mapping_status=status,
            raw_fact_id=chosen.raw_fact_id, messages=messages))

    result.unmapped_names = sorted(unmapped)
    return result
