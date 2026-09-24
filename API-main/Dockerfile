# Dockerfile
################################################################################
## Python build stage
################################################################################

# We get Python via its official Debian-based image, and install dependencies.
# Further below, these will be copied into a fresh image layer.

FROM python:3.12-slim-bookworm AS python-builder

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL

WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Add the Mayo Clinic Cloud (MCC) SSL certificate, to allow the container to
# interact with internet sites from within GCP environments.
# In addition, install application dependencies (using pip):
#ADD ./CertEmulationCA.crt /usr/local/share/ca-certificates/CertEmulationCA.crt

# An upgraded version of setuptools is installed explicitly to address a GCP
# vulnerability scanner finding.
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y ca-certificates \
&& update-ca-certificates \
&& export SSL_CERT_DIR=/etc/ssl/certs \
&& export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
&& python3 -m pip config set global.cert /usr/local/share/ca-certificates/CertEmulationCA.crt \
&& pip3 install --upgrade pip \
&& pip3 install setuptools>=80.10.1 \
&& pip3 install --no-cache-dir --extra-index-url "$PIP_EXTRA_INDEX_URL" --trusted-host pkgs.dev.azure.com -r requirements.txt

################################################################################
## Final build stage
################################################################################

FROM ubuntu:25.04

# Install system dependencies required for psycopg2, Google Cloud SDK, etc.
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
  libpq-dev \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# "Install" python and the dependencies we installed above
COPY --from=python-builder /usr/local/bin/python3.12 /usr/bin/python3.12
COPY --from=python-builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=python-builder /usr/local/lib/libpython3.12.* /usr/local/lib/

# "Install" pip and other executables from the build stage
# This includes any console scripts installed by your dependencies
COPY --from=python-builder /usr/local/bin /usr/local/bin

RUN \
  ln -s /usr/bin/python3.12 /usr/bin/python \
  && ln -s /usr/bin/python3.12 /usr/bin/python3 \
  && ldconfig

# As above, add the Mayo Clinic Cloud (MCC) SSL certificate, to allow the
# container to interact with internet sites from within GCP environments.
#ADD ./CertEmulationCA.crt /usr/local/share/ca-certificates/CertEmulationCA.crt
RUN update-ca-certificates \
  && export SSL_CERT_DIR=/etc/ssl/certs \
  && export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

################################################################################
# From here on, you can add steps that are specific to your application
################################################################################

# Set the working directory for the application:
WORKDIR /app

# Copy application code into the container:
COPY app/ ./app/

# Copy gunicorn configuration:
COPY gunicorn_config.py /app/gunicorn_config.py

# Clean up unnecessary files and cache to reduce image size:
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Add a non-root user:
RUN useradd -ms /bin/bash appuser \
  && chown -R appuser:appuser /app
USER appuser

# Whatever port your application listens on:
EXPOSE 8000

# Start Gunicorn server with the specified configuration
CMD ["gunicorn", "app.main:app", "-c", "/app/gunicorn_config.py"]
