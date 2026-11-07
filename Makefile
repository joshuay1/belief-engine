PYTHON ?= python3.11
.DEFAULT_GOAL := help
.PHONY: help check example test-classifier rebuild-replay-stats

help:
	@echo "make check                Verify the code and included files (offline)"
	@echo "make example              Show replacement behavior in each update mode"
	@echo "make test-classifier      Check scorer integration (classifier extra required)"
	@echo "make rebuild-replay-stats Rebuild saved DEBATE fold statistics into output/replay"

check:
	$(MAKE) -f paper_repro/Makefile camera-ready-check PYTHON=$(PYTHON)

example:
	$(PYTHON) -B examples/update_modes.py

test-classifier:
	$(MAKE) -f paper_repro/Makefile test-classifier PYTHON=$(PYTHON)

rebuild-replay-stats:
	$(PYTHON) -B validation/scripts/summarize_debate_rmse_uncertainty.py --output-dir output/replay
