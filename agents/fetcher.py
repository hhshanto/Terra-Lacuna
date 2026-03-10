import logging
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


def safe_filename(title: str) -> str:
    """Strip special characters, replace spaces with underscores, truncate."""
    cleaned_name = re.sub(r"[^\w\s-]", "", title)
    cleaned_name = re.sub(r"\s+", "_", cleaned_name.strip())
    return cleaned_name[:MAX_FILENAME_LENGTH]


def _extract_pdf_url_from_entry(entry: dict) -> str | None:
    """Find the PDF link in an arXiv feed entry."""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href")
    return None


def _build_paper_record(title: str, abstract: str, year, pdf_url: str | None, source: str) -> dict:
    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "pdf_url": pdf_url,
        "source": source,
    }


def fetch_from_semantic_scholar(topic: str, limit: int) -> list[dict]:
    """Search Semantic Scholar for papers on a topic."""
    params = {
        "query": topic,
        "limit": limit,
        "fields": "title,year,abstract,openAccessPdf",
        "sort": "publicationDate:desc",
    }
    try:
        response = requests.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        logging.error("Semantic Scholar request failed: %s", error)
        raise

    response_data = response.json()
    results = []
    for paper in response_data.get("data", []):
        open_access_pdf = paper.get("openAccessPdf")
        pdf_url = None
        if open_access_pdf and isinstance(open_access_pdf, dict):
            pdf_url = open_access_pdf.get("url")
        record = _build_paper_record(
            title=paper.get("title", "Untitled"),
            abstract=paper.get("abstract", ""),
            year=paper.get("year"),
            pdf_url=pdf_url,
            source="semantic_scholar",
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
        response = requests.get(ARXIV_QUERY_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
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
        response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
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
        response = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
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
            year=None,
            pdf_url=None,
            source="pubmed",
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
        )
        paper["relevance_score"] = result.get("relevance_score", 0)
        paper["relevance_reason"] = result.get("reason", "")
        return paper
    except Exception as error:
        logger.error("Relevance scoring failed for '%s': %s -- keeping paper", paper.get("title"), error)
        paper["relevance_score"] = DEFAULT_RELEVANCE_THRESHOLD
        paper["relevance_reason"] = "Scoring failed -- kept by default"
        return paper


def filter_papers_by_relevance(papers: list[dict], topic: str, threshold: int = DEFAULT_RELEVANCE_THRESHOLD) -> list[dict]:
    """Score each paper's relevance to the topic and drop those below the threshold."""
    scored_papers = []
    for paper in papers:
        scored = _score_paper_relevance(paper, topic)
        if scored is not None:
            scored_papers.append(scored)

    relevant_papers = [p for p in scored_papers if p.get("relevance_score", 0) >= threshold]
    dropped_count = len(scored_papers) - len(relevant_papers)
    if dropped_count > 0:
        logger.info("Relevance filter dropped %d/%d papers (threshold=%d)", dropped_count, len(scored_papers), threshold)
    return relevant_papers


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
    """Remove duplicate papers by normalized title."""
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        normalized_title = paper["title"].lower().strip()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        unique_papers.append(paper)
    return unique_papers


def _download_paper_pdf(pdf_url: str, destination_path: Path) -> bool:
    """Download a PDF file. Returns True on success, False on failure."""
    try:
        pdf_response = requests.get(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT_SECONDS)
        pdf_response.raise_for_status()
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

    papers_per_source = max(1, max_papers // max(len(sources), 1))

    all_papers = _collect_papers_from_sources(topic, sources, papers_per_source, pubmed_api_key)
    unique_papers = _deduplicate_papers(all_papers)

    filtered_papers = filter_papers_by_relevance(unique_papers, topic, threshold=relevance_threshold)
    logger.info("After relevance filter: %d/%d papers kept", len(filtered_papers), len(unique_papers))

    saved_count = 0
    for paper in filtered_papers[:max_papers]:
        is_saved = _save_single_paper(paper, output_directory, should_download_pdfs, should_save_abstracts)
        if is_saved:
            saved_count += 1

    return saved_count
