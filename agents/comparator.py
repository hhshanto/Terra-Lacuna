import json
import logging
from itertools import combinations

from utils.llm_client import call_llm_json

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """You are a systematic review specialist. Compare two paper extractions and identify evidence-backed tensions.

Return JSON only:
{
  "pair_summary": "one paragraph summary of overlap and differences",
  "contradictions": [
    {
      "topic": "short label",
      "claim_a": "claim from paper A",
      "claim_b": "claim from paper B",
      "papers": ["paper_a_file", "paper_b_file"],
      "evidence_quotes": [
        {"paper": "paper_a_file", "quote": "quote from extraction evidence"},
        {"paper": "paper_b_file", "quote": "quote from extraction evidence"}
      ],
      "confidence": "low|medium|high"
    }
  ],
  "shared_assumptions": [
    {
      "assumption": "assumption both papers rely on",
      "papers": ["paper_a_file", "paper_b_file"],
      "evidence_quotes": [
        {"paper": "paper_a_file", "quote": "quote"},
        {"paper": "paper_b_file", "quote": "quote"}
      ]
    }
  ],
  "understudied_populations": [
    {
      "population": "missing group/context",
      "why_missing": "why this pair suggests the gap",
      "papers": ["paper_a_file", "paper_b_file"]
    }
  ],
  "method_tension": {
    "dominant_method": "main method seen in this pair",
    "what_it_misses": "what this method leaves unanswered"
  }
}

Rules:
- Do not add claims without evidence from the provided extraction data.
- If no clear contradiction exists, return an empty contradictions list.
- Return JSON only."""

AGGREGATION_PROMPT = """You are a systematic review specialist. Aggregate pairwise comparison outputs into a final cross-paper synthesis.

Return JSON only:
{
  "contradictions": [
    {
      "topic": "short label",
      "claim_a": "...",
      "claim_b": "...",
      "papers": ["file1", "file2"],
      "evidence_quotes": [{"paper": "file", "quote": "..."}],
      "confidence": "low|medium|high"
    }
  ],
  "shared_assumptions": [
    {
      "assumption": "assumption everyone makes but rarely tests",
      "papers": ["file1", "file2"],
      "evidence_quotes": [{"paper": "file", "quote": "..."}]
    }
  ],
  "understudied_populations": [
    {
      "population": "group or context consistently absent",
      "why_missing": "why it appears understudied",
      "papers": ["file1", "file2"]
    }
  ],
  "methodological_patterns": {
    "dominant_method": "...",
    "what_it_misses": "...",
    "supporting_papers": ["file1", "file2"]
  },
  "priority_gaps": [
    {
      "gap": "specific gap statement",
      "why_it_matters": "practical value",
      "suggested_study": "study design that can close the gap",
      "supporting_papers": ["file1", "file2"]
    }
  ]
}

Rules:
- Use only the pairwise outputs.
- Keep contradictions distinct and non-duplicative.
- Prefer uncertainty labels over overconfident claims.
- Return JSON only."""

SINGLE_PAPER_FALLBACK_PROMPT = """You are a systematic review specialist. Only one paper is available.

Return JSON only with this schema:
{
  "contradictions": [],
  "shared_assumptions": [],
  "understudied_populations": [],
  "methodological_patterns": {
    "dominant_method": "method reported in the paper",
    "what_it_misses": "what cannot be concluded from one-paper evidence",
    "supporting_papers": ["paper_file"]
  },
  "priority_gaps": [
    {
      "gap": "most obvious gap from this paper",
      "why_it_matters": "why this matters",
      "suggested_study": "next study to run",
      "supporting_papers": ["paper_file"]
    }
  ]
}

Return JSON only."""


def compare_papers(extractions: list[dict]) -> dict:
    """Compare across paper extractions via pairwise pass then global aggregation."""
    logging.info("Comparing across all papers...")
    print("  -> Comparing across all papers...")
    if len(extractions) == 1:
        return _compare_single_paper(extractions[0])

    pairwise_results = _run_pairwise_comparisons(extractions)
    serialized_pairwise_results = json.dumps(pairwise_results, indent=2)

    return call_llm_json(
        system_prompt=AGGREGATION_PROMPT,
        user_message=serialized_pairwise_results,
        agent_name="comparator",
        output_schema="comparator",
    )


def _run_pairwise_comparisons(extractions: list[dict]) -> list[dict]:
    """Compare all paper pairs and return pairwise analysis objects."""
    pairwise_results = []
    for left_paper, right_paper in combinations(extractions, 2):
        pair_payload = {
            "paper_a": left_paper,
            "paper_b": right_paper,
        }
        pair_result = call_llm_json(
            system_prompt=PAIRWISE_PROMPT,
            user_message=json.dumps(pair_payload, indent=2),
            agent_name="comparator",
            output_schema="generic",
        )
        pairwise_results.append(pair_result)
    return pairwise_results


def _compare_single_paper(single_extraction: dict) -> dict:
    """Build a constrained comparison output when only one paper is present."""
    return call_llm_json(
        system_prompt=SINGLE_PAPER_FALLBACK_PROMPT,
        user_message=json.dumps(single_extraction, indent=2),
        agent_name="comparator",
        output_schema="comparator",
    )
