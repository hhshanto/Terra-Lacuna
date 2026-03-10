import json
import logging

from utils.llm_client import call_llm_json

logger = logging.getLogger(__name__)

ABSTRACT_WORD_THRESHOLD = 500

EXTRACTION_PROMPT = """You are a research analyst. Extract the following from the paper text and return JSON only:
{
  "title": "paper title or Unknown",
  "research_question": "the core question the paper tries to answer (1-2 sentences)",
  "methodology": "how they studied it e.g. RCT, survey, meta-analysis, qualitative",
  "sample": "who or what was studied -- population, size, context",
  "key_findings": ["finding 1", "finding 2"],
  "limitations": ["limitation the authors admit 1", "limitation 2"],
  "what_they_did_not_study": ["gap or scope exclusion 1", "gap 2"]
}"""

ABSTRACT_ONLY_PROMPT = """You are a research analyst. The following text is only an abstract (not a full paper).
Extract only what the abstract explicitly states. Return JSON only:
{
  "title": "paper title or Unknown",
  "research_question": "the core question (1-2 sentences)",
  "methodology": "method if mentioned, otherwise 'Not available (abstract only)'",
  "sample": "sample if mentioned, otherwise 'Not available (abstract only)'",
  "key_findings": ["finding explicitly stated in abstract"],
  "limitations": ["Not available (abstract only)"],
  "what_they_did_not_study": ["Not available (abstract only)"]
}
Do not infer or guess fields that are not explicitly stated in the abstract."""

VERIFICATION_PROMPT = """You are a fact-checking assistant. You will receive a paper's original text and a JSON extraction of that paper.

Your job:
1. Check each field in the extraction against the original text.
2. If a field is accurate, keep it as-is.
3. If a field is hallucinated (not supported by the text), correct it using only what the text says.
4. If the text is too short (e.g. just an abstract) and a field cannot be determined, set it to "Not available from text".
5. For list fields, remove any items not supported by the text.

Return the corrected JSON in the same schema. Do not add commentary -- return JSON only."""


def _count_words(text: str) -> int:
    return len(text.split())


def _select_extraction_prompt(paper_text: str) -> str:
    """Choose full or abstract-only prompt based on word count."""
    word_count = _count_words(paper_text)
    if word_count < ABSTRACT_WORD_THRESHOLD:
        logger.info("Short text detected (%d words) -- using abstract-only prompt", word_count)
        return ABSTRACT_ONLY_PROMPT
    return EXTRACTION_PROMPT


def extract_paper(paper_text: str, filename: str) -> dict:
    """Extract structured metadata from a single paper."""
    logging.info("Extracting: %s", filename)
    print(f"  -> Extracting: {filename}")

    prompt = _select_extraction_prompt(paper_text)
    is_abstract_only = _count_words(paper_text) < ABSTRACT_WORD_THRESHOLD

    extraction = call_llm_json(
        system_prompt=prompt,
        user_message=paper_text,
        agent_name="extractor",
    )
    extraction["source_file"] = filename
    extraction["is_abstract_only"] = is_abstract_only
    return extraction


def verify_extraction(paper_text: str, extraction: dict) -> dict:
    """Verify an extraction against the original paper text. Returns corrected extraction."""
    filename = extraction.get("source_file", "unknown")
    logging.info("Verifying extraction: %s", filename)
    print(f"  -> Verifying: {filename}")

    extraction_without_source = {k: v for k, v in extraction.items() if k != "source_file"}
    user_message = (
        f"=== ORIGINAL TEXT ===\n{paper_text}\n\n"
        f"=== EXTRACTION TO VERIFY ===\n{json.dumps(extraction_without_source, indent=2)}"
    )

    verified = call_llm_json(
        system_prompt=VERIFICATION_PROMPT,
        user_message=user_message,
        agent_name="extractor",
    )
    verified["source_file"] = filename
    verified["is_verified"] = True
    return verified
