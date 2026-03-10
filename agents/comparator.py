import json
import logging

from utils.llm_client import call_llm_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a systematic review specialist. Given structured summaries of multiple papers, identify patterns and tensions across them. Return JSON only:
{
  "contradictions": [
    {"claim_a": "...", "claim_b": "...", "papers": ["file1", "file2"]}
  ],
  "shared_assumptions": ["assumption everyone makes but nobody tests 1", "..."],
  "understudied_populations": ["group or context consistently absent 1", "..."],
  "methodological_patterns": {
    "dominant_method": "...",
    "what_it_misses": "..."
  }
}"""


def compare_papers(extractions: list[dict]) -> dict:
    """Compare across all paper extractions to find tensions and blind spots."""
    logging.info("Comparing across all papers...")
    print("  -> Comparing across all papers...")
    serialized_extractions = json.dumps(extractions, indent=2)
    return call_llm_json(
        system_prompt=SYSTEM_PROMPT,
        user_message=serialized_extractions,
        agent_name="comparator",
    )
