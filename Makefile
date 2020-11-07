.PHONY: addlicense all check_binaries clean cv holdout run tune_with_cv tune_with_holdout tune_mm tune_rm tune_sm

SHELL          := /bin/bash
VENV           := $(PWD)/venv
VIRTUALENV     := python3 -m venv
ACTIVATE       := $(VENV)/bin/activate
PYTHON         := $(VENV)/bin/python
PIP            := $(PYTHON) -m pip

PACKAGES       := recsys
REQUIREMENTS   := requirements.txt

LICENSE_TYPE   := "mit"
LICENSE_HOLDER := "Matthew Rossi"


define run_python
	@ echo -e "\n\n======================================"
	@ echo "$(PYTHON) $(1)"
	@ echo -e "======================================\n"
	@ PYTHONIOENCODING=UTF-8 $(PYTHON) $(1)
	@ echo -e "\n"
endef

all: run

# Make these targets quiet on pip.
lint: QUIET = --quiet

$(VENV): $(ACTIVATE)

$(ACTIVATE): requirements.txt setup.py $(PACKAGES)
	test -d $(VENV) || $(VIRTUALENV) $(VENV)
	$(PIP) install $(QUIET) --upgrade pip
	$(PIP) install $(QUIET) -r $(REQUIREMENTS)
	@ touch $(ACTIVATE)

$(FLAKE8): $(VENV)
	$(PIP) install $(QUIET) flake8

# Tune recommender system
tune_with_holdout: $(VENV)
	$(call run_python,scripts/grid_search_holdout.py)

tune_mm: $(VENV)
	$(call run_python,scripts/grid_search_holdout_mm.py)

tune_rm: $(VENV)
	$(call run_python,scripts/grid_search_holdout_rm.py)

tune_sm: $(VENV)
	$(call run_python,scripts/grid_search_holdout_sm.py)

tune_with_cv: $(VENV)
	$(call run_python,scripts/grid_search_cv.py)

# Evaluate recommender system
holdout: $(VENV)
	$(call run_python,scripts/holdout_eval.py)

cv: $(VENV)
	$(call run_python,scripts/cv_eval.py)

# Produce recommendations
run: $(VENV)
	$(call run_python,scripts/main.py)

addlicense:
	@ go get -u github.com/google/addlicense
	$(shell go env GOPATH)/bin/addlicense -c $(LICENSE_HOLDER) -l $(LICENSE_TYPE) .

clean:
	@ rm -rf $(VENV)
	@ rm -rf build/ dist/ *.egg-info/
	@ find . -path '*/__pycache__/*' -delete
	@ find . -type d -name '__pycache__' -delete
	@ find . -type f -name '*.pyo' -delete
	@ find . -type f -name '*.pyc' -delete

check_binaries:
	$(info If the following is empty you are missing a binary)
	@ whereis go
#	hint: requires installing golang-go (ppa:longsleep/golang-backports)
