import json
import logging

from utils.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research strategist. Given a structured comparison of papers on a topic, write a clear honest gap analysis. Write like a person, not a press release. Be specific. Flag uncertainty. Use these sections:

## What the Field Has Established
Short paragraph on what papers agree on.

## Where the Evidence Conflicts
Explain contradictions and why they might exist.

## Gaps Worth Pursuing
3 to 5 specific actionable gaps, each with: what the gap is, why it matters, what kind of study could address it.

For each gap, include:
- supporting papers
- one evidence quote
- confidence level: low, medium, or high
- why the gap is still unresolved

Include a markdown table after the bullet list with columns:
Gap | Supporting Papers | Contradicting Papers | Proposed Study | Feasibility | Confidence

## Caveats
What cannot be concluded from this paper set alone.

Rules:
- Do not make claims without evidence from the provided comparison JSON.
- If evidence is weak, say so explicitly.
- Prefer uncertainty over speculation."""


def synthesize_gaps(comparison: dict, topic: str) -> str:
    """Write a gap analysis report from the cross-paper comparison."""
    logging.info("Writing gap analysis...")
    print("  -> Writing gap analysis...")
    user_message = (
        f"Topic: {topic}\n\n"
        f"Cross-paper comparison:\n{json.dumps(comparison, indent=2)}"
    )
    return call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="synthesizer",
    )
