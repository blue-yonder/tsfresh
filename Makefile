WORKDIR := /tsfresh
TEST_IMAGE := tsfresh-test-image
TEST_DOCKERFILE := Dockerfile.testing
TEST_CONTAINER := tsfresh-test-container
PYTHON_VERSIONS := "3.9 3.10 3.11 3.12"

# Tests `PYTHON_VERSIONS`, provided they are also
# specified in setup.cfg `envlist`
test-all-testenv: build-docker-testenv run-docker-tests clean

build-docker-testenv:
	docker build \
			-f $(TEST_DOCKERFILE) \
			-t $(TEST_IMAGE) \
			--build-arg PYTHON_VERSIONS=$(PYTHON_VERSIONS) \
			.

run-docker-tests:
	docker run --rm \
			--name $(TEST_CONTAINER) \
			-v .:$(WORKDIR) \
			-v build_artifacts:$(WORKDIR)/build \
			-v tox_artifacts:$(WORKDIR)/.tox \
			-v egg_artifacts:$(WORKDIR)/tsfresh.egg-info \
			$(TEST_IMAGE)

clean:
	rm -rf .tox build/ dist/ *.egg-info

bisect:
	@if [ -z "$(GOOD_COMMIT)" ]; then \
		echo "Error: GOOD_COMMIT is required. Usage: make bisect GOOD_COMMIT=<commit_hash>."; \
		echo "Assumes that the current checked-out commit is a known bad commit, and bisects from there."; \
		exit 1; \
	fi
	git bisect start
	git bisect bad
	git bisect good $(GOOD_COMMIT)
	git bisect run pytest
	git bisect reset

.PHONY: build-docker-testenv clean run-docker-tests test-all-testenv bisect
