PYTHON ?= python3.11
ALL_TOPICS_DIR ?= validation/results/all_topics_u0.2_a0.4_8models

.PHONY: help reviewer-check camera-ready-check test test-classifier smoke audit-release check-artifacts check-release-manifest clean-generated rebuild-paper-artifacts rebuild-debate-rmse-stats rebuild-five-topic-table

help:
	@echo "Paper reproducibility helper targets:"
	@echo "  reviewer-check           - run all cold reviewer checks"
	@echo "  smoke                    - check Python version, syntax, and key imports"
	@echo "  audit-release            - scan for generated files, raw runs, local paths, and secrets"
	@echo "  check-artifacts          - verify validation/paper_artifacts/MANIFEST.sha256"
	@echo "  check-release-manifest   - verify paper_repro/manifest.sha256"
	@echo "  clean-generated          - remove local bytecode/cache files"
	@echo "  rebuild-paper-artifacts  - rebuild selected paper artifact bundle"
	@echo "  rebuild-debate-rmse-stats - rebuild DEBATE fold RMSE CI/test tables"
	@echo "  rebuild-five-topic-table - rebuild five-topic 6-panel figures and metrics table"

reviewer-check: camera-ready-check

camera-ready-check: smoke test audit-release check-artifacts check-release-manifest

test:
	$(PYTHON) paper_repro/scripts/run_tests.py

test-classifier:
	$(PYTHON) paper_repro/scripts/run_tests.py --classifier

smoke:
	$(PYTHON) paper_repro/scripts/reviewer_check.py

audit-release:
	$(PYTHON) paper_repro/scripts/audit_release.py

check-artifacts:
	$(PYTHON) paper_repro/scripts/check_artifacts.py

check-release-manifest:
	$(PYTHON) paper_repro/scripts/check_artifacts.py --manifest paper_repro/manifest.sha256

clean-generated:
	find . \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -prune -exec rm -rf {} +

rebuild-paper-artifacts:
	$(PYTHON) validation/scripts/prepare_paper_artifacts.py

rebuild-debate-rmse-stats:
	$(PYTHON) validation/scripts/summarize_debate_rmse_uncertainty.py

rebuild-five-topic-table:
	$(PYTHON) validation/scripts/build_6panel_topic_grids_and_metrics.py \
		--all-topics-dir $(ALL_TOPICS_DIR)
