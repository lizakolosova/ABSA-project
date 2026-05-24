# ABSA Comparison Project — top-level Makefile
#
# One-command usage:
#   make all          → pipeline + charts (full reproduction)
#   make pipeline     → run all three analysers, save evaluation_report.json
#   make charts       → regenerate docs/images/ PNGs from saved report
#   make test         → run pytest (mocked, no GPU/Ollama needed)
#   make install      → pip install -r requirements.txt + spaCy model

.PHONY: all install install-spacy pipeline pipeline-fast charts test lint clean

PYTHON  := python
PIP     := pip

# ── full reproduction ────────────────────────────────────────────────────────

all: pipeline charts

# ── dependency setup ─────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt

install-spacy: install
	$(PYTHON) -m spacy download en_core_web_trf

# ── pipeline ─────────────────────────────────────────────────────────────────

pipeline: install-spacy
	$(PYTHON) scripts/run_pipeline.py

# Skip TransformerABSA + LLMABSA for a quick local smoke-run
pipeline-fast: install-spacy
	$(PYTHON) scripts/run_pipeline.py --skip-transformer --skip-llm

# ── charts ───────────────────────────────────────────────────────────────────

charts:
	$(PYTHON) scripts/generate_charts.py

# ── tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

# ── quality ──────────────────────────────────────────────────────────────────

lint:
	flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203

# ── housekeeping ─────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f data/evaluation_report.json
