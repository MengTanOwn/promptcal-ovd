# Convenience wrappers around `promptcal`.  Every target is one command; read the
# command if you want to know exactly what runs.
PY      ?= python
BENCH   ?= coco
ALPHAS  ?= 0.05,0.10,0.20,0.30

.PHONY: help install test lint data-check extract wording calibration bootstrap \
        conformal transfer numbers figures paper clean

help:
	@echo "make install      install the package (analysis only)"
	@echo "make install-all  install with the extractor and plotting extras"
	@echo "make test         run the unit tests (CPU, no data)"
	@echo "make lint         ruff over src and tests"
	@echo "make data-check   report where the pipeline looks for its data"
	@echo "make extract BENCH=coco|lvis   run the detector, write feature shards"
	@echo "make wording      how far the score moves across wordings"
	@echo "make calibration  all calibrators, all wordings  (GPU recommended)"
	@echo "make bootstrap    image-level intervals"
	@echo "make conformal    thresholds with an FDR guarantee"
	@echo "make transfer     fit on COCO, deploy on LVIS   (needs both shard sets)"
	@echo "make numbers      write the paper's macros and tables"
	@echo "make figures      draw the paper's figures"
	@echo "make paper        numbers + figures + pdflatex"

install:
	$(PY) -m pip install -e .

install-all:
	$(PY) -m pip install -e ".[extract,figures,dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

data-check:
	$(PY) -c "from promptcal import config; print(config.describe())"

extract:
	$(PY) -m promptcal.cli extract --bench $(BENCH)

wording:
	$(PY) -m promptcal.cli wording

calibration:
	$(PY) -m promptcal.cli calibration

bootstrap:
	$(PY) -m promptcal.cli bootstrap

conformal:
	$(PY) -m promptcal.cli conformal --alphas $(ALPHAS)

transfer:
	$(PY) -m promptcal.cli transfer

numbers:
	$(PY) -m promptcal.cli macros

figures:
	$(PY) -m promptcal.cli figures

paper: numbers figures
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	  bibtex main && pdflatex -interaction=nonstopmode main.tex && \
	  pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
