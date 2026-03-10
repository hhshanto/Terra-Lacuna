# Terra Lacuna

**Map the gaps in your research field.**

Terra Lacuna is a multi-agent CLI tool that reads a folder of academic papers and figures out where the research is thin, contradictory, or missing entirely. It runs three AI agents in sequence:

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

Each agent can use a different AI provider. Set the API keys for the providers you plan to use:

| Provider      | Environment Variable     |
|---------------|--------------------------|
| Anthropic     | `ANTHROPIC_API_KEY`      |
| OpenAI        | `OPENAI_API_KEY`         |
| Azure OpenAI  | `AZURE_OPENAI_API_KEY`   |

**Linux / macOS:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export AZURE_OPENAI_API_KEY="..."
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY = "sk-..."
$env:AZURE_OPENAI_API_KEY = "..."
```

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
- **`azure_openai`** — Uses Azure OpenAI Service. Set `AZURE_OPENAI_API_KEY` and configure the `azure_openai` section in the config:

```yaml
azure_openai:
  endpoint: https://YOUR_RESOURCE.openai.azure.com/
  api_version: "2024-02-01"
```

The `model` field for Azure should be your **deployment name** — find it in the Azure Portal under your OpenAI resource → Deployments.

---

## Usage

```bash
# Basic usage
python main.py --papers ./papers --topic "attention and social media"

# With a custom config file
python main.py --papers ./papers --topic "burnout in nurses" --config custom_config.yaml
```

Place your `.txt` or `.pdf` papers in a folder and point `--papers` at it.

### Example output

```
Found 5 paper(s) in './papers'.

[ AGENT 1 ] Extracting paper metadata
  → Extracting: smith_2022.pdf
  → Extracting: jones_2023.pdf
  → Extracting: lee_2021.txt
  → Extracting: garcia_2023.pdf
  → Extracting: chen_2022.pdf

[ AGENT 2 ] Comparing across papers
  → Comparing across all papers...

[ AGENT 3 ] Writing gap analysis
  → Writing gap analysis...

============================================================
# Terra Lacuna — Gap Analysis: attention and social media

## What the Field Has Established
...

## Where the Evidence Conflicts
...

## Gaps Worth Pursuing
...

## Caveats
...
============================================================

✅ Done — gap_analysis.md
   Papers processed: 5
   Output files: extractions.json, comparison.json, gap_analysis.md
```

### Output files

| File                | Contents                                          |
|---------------------|---------------------------------------------------|
| `extractions.json`  | Structured metadata for each paper                |
| `comparison.json`   | Cross-paper contradictions, patterns, blind spots |
| `gap_analysis.md`   | Human-readable gap analysis report                |

---

## Limitations

Terra Lacuna helps you see the **shape** of gaps in a body of research — where evidence is missing, where studies contradict each other, and what populations or methods are underrepresented. But it cannot judge **importance**. Whether a gap is worth pursuing depends on your field, your resources, and your judgment. The agents find the gaps; deciding which ones matter is still yours.
