# Terra Lacuna

**Map the gaps in your research field.**

Terra Lacuna is a multi-agent CLI tool that reads a folder of academic papers and figures out where the research is thin, contradictory, or missing entirely. It runs four AI agents in sequence:

0. **Fetcher** — searches Semantic Scholar, arXiv, and PubMed for papers on your topic and downloads them automatically
1. **Extractor** — pulls structured metadata from each paper (title, methodology, findings, limitations, unstudied areas)
2. **Comparator** — compares across all papers to find contradictions, shared assumptions, and blind spots
3. **Synthesizer** — writes a final gap analysis report with actionable research directions

What takes days of manual literature review, Terra Lacuna does in minutes.

---

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root (automatically loaded):

```env
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Each agent can use a different AI provider. Set the API keys for the providers you plan to use:

| Provider      | Environment Variable     |
|---------------|--------------------------|
| Anthropic     | `ANTHROPIC_API_KEY`      |
| OpenAI        | `OPENAI_API_KEY`         |
| Azure OpenAI  | `AZURE_OPENAI_API_KEY`   |

---

## Configuration

Edit `config.yaml` to control which provider and model each agent uses:

```yaml
agents:
  extractor:
    provider: anthropic
    model: claude-opus-4-5
  comparator:
    provider: azure_openai
    model: gpt-4o
  synthesizer:
    provider: openai
    model: gpt-4o
```

### Provider options

- **`anthropic`** — Uses the Anthropic API. Set `ANTHROPIC_API_KEY`. Model names like `claude-opus-4-5`, `claude-sonnet-4-20250514`, etc.
- **`openai`** — Uses the OpenAI API. Set `OPENAI_API_KEY`. Model names like `gpt-4o`, `gpt-4-turbo`, etc.
- **`azure_openai`** — Uses Azure OpenAI Service. Set `AZURE_OPENAI_API_KEY` and optionally `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` in `.env`. Falls back to values in `config.yaml`.

### Fetcher configuration

```yaml
fetcher:
  sources:
    - semantic_scholar
    - arxiv
    - pubmed
  max_papers: 20
  download_pdfs: true
  save_abstracts_as_txt: true
  pubmed_api_key: ""
  year_from: 2020
  year_to: 2026
```

### Paper sources

| Source            | Free? | Rate limit                         | Best for                              |
|-------------------|-------|------------------------------------|---------------------------------------|
| Semantic Scholar  | Yes   | 100 requests / 5 min               | General academic, cross-discipline    |
| arXiv             | Yes   | ~3 seconds between requests        | CS, AI, physics, math, biology        |
| PubMed            | Yes   | 3 req/sec (10 with free API key)   | Medicine, psychology, neuroscience    |

> **Note:** Full PDFs are not always available. The APIs always return abstracts, but full text depends on open-access availability. arXiv is the best source for full PDFs. For paywalled papers, abstracts are saved as `.txt` files as a fallback.
>
> PubMed's API key is free — register at https://www.ncbi.nlm.nih.gov/account/ to raise your rate limit.

---

## Usage

### Full pipeline with automatic paper fetching

```bash
python main.py --topic "burnout in nurses" --fetch
```

This runs all four agents in sequence: fetch → extract → compare → synthesize.

### Full pipeline with your own papers

```bash
python main.py --papers ./papers --topic "attention and social media"
```

### Run individual steps

You can run each step separately to inspect intermediate results:

```bash
# Step 0 — Fetch papers
python main.py --topic "burnout in nurses" --step fetch
python main.py --topic "burnout in nurses" --step fetch --year-from 2020
python main.py --topic "burnout in nurses" --step fetch --year-from 2020 --year-to 2024

# Step 1 — Extract metadata (uses papers from fetch, or specify --papers)
python main.py --topic "burnout in nurses" --step extract
python main.py --topic "burnout in nurses" --step extract --papers ./my_papers

# Step 2 — Compare across papers (reads extractions.json)
python main.py --topic "burnout in nurses" --step compare

# Step 3 — Synthesize gap analysis (reads comparison.json)
python main.py --topic "burnout in nurses" --step synthesize
```

### With a custom config file

```bash
python main.py --papers ./papers --topic "burnout in nurses" --config custom_config.yaml
```

Place your `.txt` or `.pdf` papers in a folder and point `--papers` at it.

### Example output

```
[ AGENT 0 ] Fetching latest papers on "attention and social media"
  -> Downloaded: smartphone_use_and_attention_2024.pdf
  -> Saved abstract: tiktok_dopamine_loops_2023.txt
  ✓ 14 files ready

Found 14 paper(s) in 'papers/attention_and_social_media'.

[ AGENT 1 ] Extracting paper metadata
  -> Extracting: smith_2022.pdf
  -> Extracting: jones_2023.pdf
  -> Extracting: lee_2021.txt

[ AGENT 2 ] Comparing across papers
  -> Comparing across all papers...

[ AGENT 3 ] Writing gap analysis
  -> Writing gap analysis...

============================================================
# Terra Lacuna -- Gap Analysis: attention and social media
...
============================================================

Done -- data/gap_analysis.md
   Papers processed: 14
   Output files: data/extractions.json, data/comparison.json, data/gap_analysis.md
```

### Output files

| File                       | Contents                                          |
|----------------------------|---------------------------------------------------|
| `data/extractions.json`    | Structured metadata for each paper                |
| `data/comparison.json`     | Cross-paper contradictions, patterns, blind spots |
| `data/gap_analysis.md`     | Human-readable gap analysis report                |

---

## Limitations

Terra Lacuna helps you see the **shape** of gaps in a body of research — where evidence is missing, where studies contradict each other, and what populations or methods are underrepresented. But it cannot judge **importance**. Whether a gap is worth pursuing depends on your field, your resources, and your judgment. The agents find the gaps; deciding which ones matter is still yours.
