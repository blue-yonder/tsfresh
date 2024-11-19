WORKDIR := /tsfresh
TEST_IMAGE := tsfresh-test-image
TEST_DOCKERFILE := Dockerfile.testing
TEST_CONTAINER := tsfresh-test-container
PYTHON_VERSIONS := "3.7 3.8 3.9 3.10 3.11"

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


.PHONY: build-docker-testenv clean run-docker-tests test-all-testenv
