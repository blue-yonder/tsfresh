# Define builder and base image
FROM python:3.8-slim as base
FROM python:3.8 as builder

LABEL maintainer="nilslennartbraun@gmail.com"

# Install tsfresh from source into the builder image
ADD . /source
WORKDIR /source
RUN pip3 install --prefix=/install .

# Copy the installed sources to the base image
FROM base
COPY --from=builder /install /usr/local
