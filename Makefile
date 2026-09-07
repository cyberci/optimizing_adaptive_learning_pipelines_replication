PYTHON ?= python3

.PHONY: help venv install all train bench figures loadtest check clean distclean

help:
	@echo "make install    install the pinned dependencies into the active environment"
	@echo "make venv       create .venv and install into it"
	@echo "make all        train, measure and redraw everything (~45 min)"
	@echo "make bench      measurement only, reusing the shipped models (~40 min)"
	@echo "make figures    redraw the figures from the existing results/ (~10 s)"
	@echo "make loadtest   the optional HTTP cross-check (starts a local server)"
	@echo "make check      pre-flight check: parse every script, report missing dependencies"
	@echo "make clean      remove logs/, work/ and __pycache__"
	@echo "make distclean  also remove the regenerated results/ and figures/output/"

venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "activate with: . .venv/bin/activate"

install:
	$(PYTHON) -m pip install -r requirements.txt

all:
	./run_all.sh all

train:
	./run_all.sh train

bench:
	./run_all.sh bench

figures:
	./run_all.sh figures

loadtest:
	./run_all.sh loadtest

check:
	@$(PYTHON) tools/check.py

clean:
	rm -rf logs work
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -f results/*.json figures/output/*
