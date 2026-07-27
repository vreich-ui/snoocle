# Snoocle server — convenience targets. Everything here is also runnable by
# hand; nothing in the build depends on make.

PY ?= .venv/bin/python

.PHONY: test pytest jstest acceptance icons lint help

help:
	@echo "make test        - the full gate: pytest (which also runs the JS tests)"
	@echo "make pytest      - Python tests only"
	@echo "make jstest      - JS unit tests only (node --test tests_js/)"
	@echo "make acceptance  - regenerate docs/ACCEPTANCE.md offline"
	@echo "make icons       - regenerate the player's PWA raster icons"

test: pytest

pytest:
	$(PY) -m pytest

# The pure browser logic — scroll model, chord display transforms, diagram
# mapping. Also invoked from tests/test_player_ui.py so `make test` covers it.
jstest:
	node --test "tests_js/*.test.mjs"

acceptance:
	$(PY) scripts/acceptance.py --offline

icons:
	$(PY) scripts/make_player_icons.py
