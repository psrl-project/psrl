# Codex sandbox for the math RL recipe. The base image already ships Codex.
# p3b starts a sandbox with `sleep infinity`, so clear the Rethlas entrypoint.
FROM rethlas-codex:local

ENTRYPOINT []
CMD ["sleep", "infinity"]

ENV CODEX_HOME=/root/.codex

RUN python3 -m pip install --break-system-packages --no-cache-dir numpy scipy sympy

WORKDIR /workspace
