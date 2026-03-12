import logging
import json
import os
import re
import time
from pathlib import Path

import feedparser
import requests
import yaml

from utils.llm_client import call_llm_json

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _config_file:
    _FETCHER_CONFIG = yaml.safe_load(_config_file)

_API_URLS = _FETCHER_CONFIG.get("api_urls", {})
SEMANTIC_SCHOLAR_SEARCH_URL = _API_URLS.get(
    "semantic_scholar", "https://api.semanticscholar.org/graph/v1/paper/search"
)
ARXIV_QUERY_URL = _API_URLS.get(
    "arxiv", "http://export.arxiv.org/api/query"
)
PUBMED_SEARCH_URL = _API_URLS.get(
    "pubmed_search", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = _API_URLS.get(
    "pubmed_fetch", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

RATE_LIMIT_SECONDS = 3
MAX_FILENAME_LENGTH = 80
REQUEST_TIMEOUT_SECONDS = 30
PDF_DOWNLOAD_TIMEOUT_SECONDS = 60
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
MAX_HTTP_RETRIES = 4
INITIAL_RETRY_BACKOFF_SECONDS = 2
MAX_RETRY_AFTER_SECONDS = 60
DEFAULT_USER_AGENT = "Terra-Lacuna/1.0 (research-fetcher)"


def safe_filename(title: str) -> str:
    """Strip special characters, replace spaces with underscores, truncate."""
    cleaned_name = re.sub(r"[^\w\s-]", "", title)
    cleaned_name = re.sub(r"\s+", "_", cleaned_name.strip())
    return cleaned_name[:MAX_FILENAME_LENGTH]


def _build_request_headers(source_name: str) -> dict:
    """Build request headers and include optional source-specific keys."""
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json, application/xml, text/plain;q=0.9, */*;q=0.8",
    }
    if source_name == "semantic_scholar":
        semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if semantic_scholar_api_key:
            headers["x-api-key"] = semantic_scholar_api_key
    return headers


def _get_retry_wait_seconds(response: requests.Response, attempt_number: int) -> int:
    """Return wait seconds based on Retry-After header or exponential backoff."""
    retry_after_header = response.headers.get("Retry-After", "").strip()
    if retry_after_header.isdigit():
        retry_after_seconds = int(retry_after_header)
        return max(1, min(retry_after_seconds, MAX_RETRY_AFTER_SECONDS))

    backoff_seconds = INITIAL_RETRY_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
    return min(backoff_seconds, MAX_RETRY_AFTER_SECONDS)


def _request_with_retries(
    url: str,
    source_name: str,
    params: dict | None = None,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> requests.Response:
    """Perform GET request with retry/backoff for transient HTTP failures."""
    headers = _build_request_headers(source_name)
    last_error = None

    for attempt_number in range(1, MAX_HTTP_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds, headers=headers)
        except requests.RequestException as error:
            last_error = error
            if attempt_number == MAX_HTTP_RETRIES:
                break
            wait_seconds = INITIAL_RETRY_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
            logger.warning(
                "%s request failed (attempt %d/%d): %s -- retrying in %ds",
                source_name,
                attempt_number,
                MAX_HTTP_RETRIES,
                error,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429:
            if attempt_number == MAX_HTTP_RETRIES:
                response.raise_for_status()
            wait_seconds = _get_retry_wait_seconds(response, attempt_number)
            logger.warning(
                "%s rate limited (429) on attempt %d/%d -- retrying in %ds",
                source_name,
                attempt_number,
                MAX_HTTP_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt_number == MAX_HTTP_RETRIES:
                break
            wait_seconds = INITIAL_RETRY_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
            logger.warning(
                "%s request failed with status %s (attempt %d/%d) -- retrying in %ds",
                source_name,
                response.status_code,
                attempt_number,
                MAX_HTTP_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{source_name} request failed after retries")


def _extract_pdf_url_from_entry(entry: dict) -> str | None:
    """Find the PDF link in an arXiv feed entry."""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href")
    return None


def _build_paper_record(
    title: str,
    abstract: str,
    year,
    pdf_url: str | None,
    source: str,
    doi: str | None = None,
) -> dict:
    normalized_year = _parse_year_value(year)
    return {
        "title": title,
        "abstract": abstract,
        "year": normalized_year,
        "pdf_url": pdf_url,
        "source": source,
        "doi": doi,
    }


def _parse_year_value(year_value) -> int | None:
    """Parse year from number or text and return a normalized int year."""
    if year_value is None:
        return None

    if isinstance(year_value, int):
        return year_value if 1900 <= year_value <= 2100 else None

    year_text = str(year_value).strip()
    if not year_text:
        return None

    if year_text.isdigit() and len(year_text) == 4:
        parsed_year = int(year_text)
        return parsed_year if 1900 <= parsed_year <= 2100 else None

    match = YEAR_PATTERN.search(year_text)
    if not match:
        return None
    parsed_year = int(match.group(1))
    return parsed_year if 1900 <= parsed_year <= 2100 else None


def fetch_from_semantic_scholar(topic: str, limit: int) -> list[dict]:
    """Search Semantic Scholar for papers on a topic."""
    params = {
        "query": topic,
        "limit": limit,
        "fields": "title,year,abstract,openAccessPdf,externalIds",
        "sort": "publicationDate:desc",
    }
    try:
        response = _request_with_retries(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            "semantic_scholar",
            params=params,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        logging.error("Semantic Scholar request failed: %s", error)
        raise

    response_data = response.json()
    results = []
    for paper in response_data.get("data", []):
        open_access_pdf = paper.get("openAccessPdf")
        external_ids = paper.get("externalIds", {})
        pdf_url = None
        if open_access_pdf and isinstance(open_access_pdf, dict):
            pdf_url = open_access_pdf.get("url")

        record = _build_paper_record(
            title=paper.get("title", "Untitled"),
            abstract=paper.get("abstract", ""),
            year=paper.get("year"),
            pdf_url=pdf_url,
            source="semantic_scholar",
            doi=external_ids.get("DOI") if isinstance(external_ids, dict) else None,
        )
        results.append(record)

    time.sleep(RATE_LIMIT_SECONDS)
    return results


def fetch_from_arxiv(topic: str, limit: int) -> list[dict]:
    """Search arXiv for papers on a topic."""
    params = {
        "search_query": f"all:{topic}",
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        response = _request_with_retries(
            ARXIV_QUERY_URL,
            "arxiv",
            params=params,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        logging.error("arXiv request failed: %s", error)
        raise

    feed = feedparser.parse(response.text)
    results = []
    for entry in feed.entries:
        pdf_url = _extract_pdf_url_from_entry(entry)
        record = _build_paper_record(
            title=entry.get("title", "Untitled").replace("\n", " ").strip(),
            abstract=entry.get("summary", "").replace("\n", " ").strip(),
            year=entry.get("published", "")[:4] or None,
            pdf_url=pdf_url,
            source="arxiv",
            doi=entry.get("arxiv_doi"),
        )
        results.append(record)

    time.sleep(RATE_LIMIT_SECONDS)
    return results


def _search_pubmed_ids(topic: str, limit: int, api_key: str = "") -> list[str]:
    """Search PubMed and return a list of article IDs."""
    search_params = {
        "db": "pubmed",
        "term": topic,
        "retmax": limit,
        "sort": "date",
        "retmode": "json",
    }
    if api_key:
        search_params["api_key"] = api_key

    try:
        response = _request_with_retries(
            PUBMED_SEARCH_URL,
            "pubmed_search",
            params=search_params,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        logging.error("PubMed search request failed: %s", error)
        raise

    return response.json().get("esearchresult", {}).get("idlist", [])


def _fetch_pubmed_abstracts(article_ids: list[str], api_key: str = "") -> str:
    """Fetch abstract text for a list of PubMed article IDs."""
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(article_ids),
        "rettype": "abstract",
        "retmode": "text",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    try:
        response = _request_with_retries(
            PUBMED_FETCH_URL,
            "pubmed_fetch",
            params=fetch_params,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        logging.error("PubMed fetch request failed: %s", error)
        raise

    return response.text


def fetch_from_pubmed(topic: str, limit: int, api_key: str = "") -> list[dict]:
    """Search PubMed for papers on a topic."""
    article_ids = _search_pubmed_ids(topic, limit, api_key)
    if not article_ids:
        return []

    rate_limit_delay = 0.1 if api_key else 0.4
    time.sleep(rate_limit_delay)

    raw_text = _fetch_pubmed_abstracts(article_ids, api_key)

    blocks = re.split(r"\n{2,}(?=\d+\.\s)", raw_text.strip())
    results = []
    for block in blocks:
        lines = block.strip().split("\n")
        paper_title = lines[0] if lines else "Untitled"
        paper_title = re.sub(r"^\d+\.\s*", "", paper_title).strip()
        record = _build_paper_record(
            title=paper_title,
            abstract=block.strip(),
            year=_parse_year_value(block),
            pdf_url=None,
            source="pubmed",
            doi=None,
        )
        results.append(record)

    time.sleep(rate_limit_delay)
    return results


SOURCE_FETCHERS = {
    "semantic_scholar": lambda topic, limit, _key: fetch_from_semantic_scholar(topic, limit),
    "arxiv": lambda topic, limit, _key: fetch_from_arxiv(topic, limit),
    "pubmed": fetch_from_pubmed,
}

RELEVANCE_PROMPT = """You are a research relevance judge. Given a research topic and a paper's title and abstract, rate how relevant the paper is to the topic.

Return JSON only:
{
  "relevance_score": <integer from 1 to 5>,
  "reason": "<one sentence explaining your rating>"
}

Scoring guide:
5 = Directly addresses the topic
4 = Closely related, covers key aspects
3 = Somewhat related, tangential overlap
2 = Loosely related, minimal overlap
1 = Not relevant to the topic"""

DEFAULT_RELEVANCE_THRESHOLD = 3


def _score_paper_relevance(paper: dict, topic: str) -> dict | None:
    """Score a single paper's relevance. Returns the paper with score, or None on failure."""
    user_message = (
        f"Research topic: {topic}\n\n"
        f"Paper title: {paper.get('title', 'Unknown')}\n\n"
        f"Paper abstract: {paper.get('abstract', 'No abstract available')}"
    )
    try:
        result = call_llm_json(
            system_prompt=RELEVANCE_PROMPT,
            user_message=user_message,
            agent_name="extractor",
            output_schema="relevance",
        )
        paper["relevance_score"] = result.get("relevance_score", 0)
        paper["relevance_reason"] = result.get("reason", "")
        return paper
    except Exception as error:
        logger.error("Relevance scoring failed for '%s': %s -- keeping paper", paper.get("title"), error)
        paper["relevance_score"] = DEFAULT_RELEVANCE_THRESHOLD
        paper["relevance_reason"] = "Scoring failed -- kept by default"
        return paper


def filter_papers_by_relevance(
    papers: list[dict],
    topic: str,
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Score relevance and return kept papers plus dropped papers."""
    scored_papers = []
    for paper in papers:
        scored = _score_paper_relevance(paper, topic)
        if scored is not None:
            scored_papers.append(scored)

    relevant_papers = [p for p in scored_papers if p.get("relevance_score", 0) >= threshold]
    dropped_papers = [p for p in scored_papers if p.get("relevance_score", 0) < threshold]
    dropped_count = len(scored_papers) - len(relevant_papers)
    if dropped_count > 0:
        logger.info("Relevance filter dropped %d/%d papers (threshold=%d)", dropped_count, len(scored_papers), threshold)
    return relevant_papers, dropped_papers


def _collect_papers_from_sources(topic: str, sources: list[str], papers_per_source: int, pubmed_api_key: str) -> list[dict]:
    """Fetch papers from each configured source, logging errors without crashing."""
    all_papers = []
    for source_name in sources:
        fetcher_function = SOURCE_FETCHERS.get(source_name)
        if not fetcher_function:
            logging.error("Unknown paper source: %s", source_name)
            continue
        try:
            papers = fetcher_function(topic, papers_per_source, pubmed_api_key)
            all_papers.extend(papers)
        except Exception as error:
            logging.error("Error fetching from %s: %s", source_name, error)
    return all_papers


def _deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicates by DOI first, then normalized title."""
    seen_dois = set()
    seen_titles = set()
    unique_papers = []

    for paper in papers:
        normalized_doi = str(paper.get("doi", "")).strip().lower()
        if normalized_doi:
            if normalized_doi in seen_dois:
                continue
            seen_dois.add(normalized_doi)
            unique_papers.append(paper)
            continue

        normalized_title = paper["title"].lower().strip()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        unique_papers.append(paper)
    return unique_papers


def _is_paper_in_year_range(paper: dict, year_from: int | None, year_to: int | None) -> bool:
    """Return True if paper year satisfies configured range, or year filter is not set."""
    if year_from is None and year_to is None:
        return True

    paper_year = _parse_year_value(paper.get("year"))
    if paper_year is None:
        return False

    if year_from is not None and paper_year < year_from:
        return False
    if year_to is not None and paper_year > year_to:
        return False
    return True


def _filter_papers_by_year(
    papers: list[dict],
    year_from: int | None,
    year_to: int | None,
) -> tuple[list[dict], list[dict]]:
    """Filter papers by year and return kept plus dropped records."""
    kept_papers = []
    dropped_papers = []
    for paper in papers:
        if _is_paper_in_year_range(paper, year_from, year_to):
            kept_papers.append(paper)
            continue
        dropped_papers.append(paper)
    return kept_papers, dropped_papers


def _download_paper_pdf(pdf_url: str, destination_path: Path) -> bool:
    """Download a PDF file. Returns True on success, False on failure."""
    try:
        pdf_response = _request_with_retries(
            pdf_url,
            "pdf_download",
            params=None,
            timeout_seconds=PDF_DOWNLOAD_TIMEOUT_SECONDS,
        )
        destination_path.write_bytes(pdf_response.content)
        return True
    except requests.RequestException as error:
        logging.error("Failed to download PDF from %s: %s", pdf_url, error)
        return False


def _save_abstract_as_text(paper: dict, destination_path: Path) -> None:
    """Save a paper's abstract as a .txt file with metadata header."""
    header = f"Title: {paper['title']}\n"
    if paper.get("year"):
        header += f"Year: {paper['year']}\n"
    header += f"Source: {paper['source']}\n\n"
    destination_path.write_text(header + paper["abstract"], encoding="utf-8")


def _save_single_paper(paper: dict, output_directory: Path, should_download_pdfs: bool, should_save_abstracts: bool) -> bool:
    """Try to save a single paper as PDF or abstract. Returns True if saved."""
    filename = safe_filename(paper["title"])
    if not filename:
        return False

    if paper.get("pdf_url") and should_download_pdfs:
        pdf_path = output_directory / f"{filename}.pdf"
        is_downloaded = _download_paper_pdf(paper["pdf_url"], pdf_path)
        if is_downloaded:
            logging.info("Downloaded: %s", paper["title"])
            return True

    if paper.get("abstract") and should_save_abstracts:
        text_path = output_directory / f"{filename}.txt"
        _save_abstract_as_text(paper, text_path)
        logging.info("Saved abstract: %s", paper["title"])
        return True

    logging.info("Skipped (no full text): %s", paper["title"])
    return False


def _save_relevance_audit(
    topic: str,
    threshold: int,
    year_from: int | None,
    year_to: int | None,
    kept_papers: list[dict],
    dropped_papers: list[dict],
) -> None:
    """Save relevance scoring decisions so filtering remains transparent."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    audit_path = DATA_DIRECTORY / "relevance_audit.json"
    audit_data = {
        "topic": topic,
        "threshold": threshold,
        "year_from": year_from,
        "year_to": year_to,
        "kept_count": len(kept_papers),
        "dropped_count": len(dropped_papers),
        "kept": [
            {
                "title": paper.get("title"),
                "source": paper.get("source"),
                "doi": paper.get("doi"),
                "relevance_score": paper.get("relevance_score"),
                "relevance_reason": paper.get("relevance_reason"),
            }
            for paper in kept_papers
        ],
        "dropped": [
            {
                "title": paper.get("title"),
                "source": paper.get("source"),
                "doi": paper.get("doi"),
                "relevance_score": paper.get("relevance_score"),
                "relevance_reason": paper.get("relevance_reason"),
            }
            for paper in dropped_papers
        ],
    }
    audit_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")


def fetch_papers(topic: str, output_dir: str, config: dict) -> int:
    """Fetch papers from configured sources, filter by relevance, and save to output_dir.

    Returns the number of files saved.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    sources = config.get("sources", ["semantic_scholar", "arxiv"])
    max_papers = config.get("max_papers", 20)
    should_download_pdfs = config.get("download_pdfs", True)
    should_save_abstracts = config.get("save_abstracts_as_txt", True)
    pubmed_api_key = config.get("pubmed_api_key", "")
    relevance_threshold = config.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD)
    year_from = _parse_year_value(config.get("year_from"))
    year_to = _parse_year_value(config.get("year_to"))

    papers_per_source = max(1, max_papers // max(len(sources), 1))

    all_papers = _collect_papers_from_sources(topic, sources, papers_per_source, pubmed_api_key)
    unique_papers = _deduplicate_papers(all_papers)
    year_filtered_papers, year_dropped_papers = _filter_papers_by_year(unique_papers, year_from, year_to)
    if year_from is not None or year_to is not None:
        logger.info(
            "Year filter kept %d/%d papers (from=%s, to=%s)",
            len(year_filtered_papers),
            len(unique_papers),
            year_from,
            year_to,
        )

    filtered_papers, dropped_papers = filter_papers_by_relevance(
        year_filtered_papers,
        topic,
        threshold=relevance_threshold,
    )
    all_dropped_papers = year_dropped_papers + dropped_papers
    _save_relevance_audit(
        topic,
        relevance_threshold,
        year_from,
        year_to,
        filtered_papers,
        all_dropped_papers,
    )
    logger.info("After relevance filter: %d/%d papers kept", len(filtered_papers), len(year_filtered_papers))

    saved_count = 0
    for paper in filtered_papers[:max_papers]:
        is_saved = _save_single_paper(paper, output_directory, should_download_pdfs, should_save_abstracts)
        if is_saved:
            saved_count += 1

    return saved_count
