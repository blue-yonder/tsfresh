WORKDIR := /tsfresh
TEST_IMAGE := tsfresh-test-image
TEST_DOCKERFILE := Dockerfile.testing
TEST_CONTAINER := tsfresh-test-container
# >= 3.9.2 ---> https://github.com/dask/distributed/issues/7956
PYTHON_VERSIONS := "3.7.12 3.8.12 3.9.12 3.10.12 3.11.0"
BLACK_VERSION := 22.12.0
# Isort 5.12.0 not supported with python 3.7.12
ISORT_VERSION := 5.12.0

build-testenv:
	docker build \
			-f $(TEST_DOCKERFILE) \
			-t $(TEST_IMAGE) \
			--build-arg PYTHON_VERSIONS=$(PYTHON_VERSIONS) \
			.

# Tests `PYTHON_VERSIONS`, provided they are also
# specified in setup.cfg `envlist`
test-all-testenv: clean build-testenv
	docker run --rm \
			--name $(TEST_CONTAINER) \
			-v .:$(WORKDIR) \
			-v build_artifacts:$(WORKDIR)/build \
			-v tox_artifacts:$(WORKDIR)/.tox \
			-v egg_artifacts:$(WORKDIR)/tsfresh.egg-info \
			$(TEST_IMAGE)

# Tests the python binaries installed
# on local machine, provided they are also
# specified in setup.cfg `envlist`
test-all-local: clean
	tox -r -p auto

# Tests for python version on local machine in
# current context (e.g. global or local version of
# python set by pyenv, or the python version in
# the active virtualenv).
test-local: clean
	pip install .[testing]
	pytest

clean:
	rm -rf .tox build/ dist/ *.egg-info

install-formatters:
	pip install black==$(BLACK_VERSION) isort==$(ISORT_VERSION)

format: install-formatters
	black --extend-exclude \.docs .
	isort --profile black --skip-glob="docs" .

.PHONY: clean test-all-local test-local test-all-testenv format install-formatters
