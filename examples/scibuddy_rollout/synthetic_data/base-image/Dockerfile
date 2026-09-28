FROM debian:bookworm-slim

# scibuddy-synth-cpu:1.0 — base image for the synthetic tempstats smoke task.
# The synthetic task's environment/ and tests/ Dockerfiles both build FROM this
# tag, so it must exist on every Harbor node before a run. The task itself only
# needs python3; terminus-2 additionally installs recording tools (tmux,
# asciinema) at agent-setup, and a no-network task cannot apt-install them at
# runtime, so tmux/procps/curl are baked in here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        tmux procps curl ca-certificates bash coreutils && \
    rm -rf /var/lib/apt/lists/*

# The grader invokes `python3` explicitly; add a `python` alias for convenience.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python || true

WORKDIR /workspace
