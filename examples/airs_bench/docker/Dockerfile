# AIRS-Bench agent sandbox image.
#
# Derived from MLGym's own agent image, with every AIRS-Bench task requirement
# baked in. Installing per episode would dominate runtime, because the same 14
# tasks run thousands of times during RL.
#
# NOTE(lhy): The Ubuntu package archive (archive.ubuntu.com) and conda channels
# (repo.anaconda.com, conda.anaconda.org) are unreachable from this cluster's
# docker build network. Workarounds:
#   - apt: sources.list rewritten to mirrors.tencentyun.com (accessible via HTTP).
#   - miniconda: installer pre-downloaded to build context by build_image.sh.
#   - mlgym_generic conda env: pre-created on the host with psrl conda 24.9.2
#     (which successfully contacts repo.anaconda.com through the proxy) and packed
#     with conda-pack. The tarball is COPY-ed in and unpacked offline.
#     build_image.sh regenerates the pack if it is missing.
ARG BASE_IMAGE=ubuntu:jammy
FROM ${BASE_IMAGE}

ARG TARGETARCH=amd64
ARG APT_MIRROR=mirrors.tencentyun.com
ARG http_proxy
ARG https_proxy
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i "s|http://archive.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g; \
            s|http://security.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g" \
      /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y bash gcc g++ git jq wget curl make file sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r mlgym \
    && useradd -r -g mlgym -m -d /home/agent -s /bin/bash agent \
    && echo "agent ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt" >> /etc/sudoers.d/agent \
    && chmod 440 /etc/sudoers.d/agent

ENV ROOT='/home/agent/'
ENV PS1="> "
WORKDIR /home/agent

ENV PATH="/home/agent/miniconda3/bin:/home/agent/.local/bin:${PATH}"

USER agent

# Miniconda, matching MLGym's expectation of a conda-managed agent environment.
# NOTE(lhy): The installer is pre-downloaded and copied from the build context
# because repo.anaconda.com is reachable only from the host. build_image.sh
# auto-downloads it if absent.
COPY --chown=agent:mlgym Miniconda3-latest-Linux-x86_64.sh /home/agent/miniconda.sh
RUN bash /home/agent/miniconda.sh -b -u -p /home/agent/miniconda3 \
    && rm -f /home/agent/miniconda.sh \
    && conda init bash

# The generic environment MLGym activates when a task sets use_generic_conda.
# NOTE(lhy): conda 26.x inside docker build cannot install Python 3.11 from any
# mirror because the proxy selectively blocks .conda package downloads during
# docker build (but not during docker run). The workaround: build_image.sh pre-creates
# this env on the host using psrl conda 24.9.2 (which reaches repo.anaconda.com
# directly), packs it with conda-pack, and COPYs the tarball here.
COPY --chown=agent:mlgym mlgym_generic.tar.gz /home/agent/mlgym_generic.tar.gz
RUN mkdir -p /home/agent/miniconda3/envs/mlgym_generic \
    && tar -xzf /home/agent/mlgym_generic.tar.gz -C /home/agent/miniconda3/envs/mlgym_generic \
    && rm -f /home/agent/mlgym_generic.tar.gz \
    && /home/agent/miniconda3/envs/mlgym_generic/bin/conda-unpack \
    && /home/agent/miniconda3/envs/mlgym_generic/bin/python --version

COPY --chown=agent:mlgym requirements_union.txt /home/agent/requirements_union.txt

# Bake every task requirement so episodes never pay install cost.
# NOTE(lhy): Use python -m pip because the pip shebang expects python3.11 on PATH,
# which is only available after conda activate. Calling python directly bypasses
# the shebang and works reliably in a non-interactive shell.
RUN /home/agent/miniconda3/envs/mlgym_generic/bin/python -m pip install --no-cache-dir \
      -r /home/agent/requirements_union.txt

# MLGym's flake8 linting hook expects flake8 on the base interpreter.
RUN /home/agent/miniconda3/bin/pip install --no-cache-dir flake8

RUN mkdir -p /home/agent/workspace

CMD ["/bin/bash", "-l"]
