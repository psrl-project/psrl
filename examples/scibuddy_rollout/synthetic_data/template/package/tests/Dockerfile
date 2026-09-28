FROM scibuddy-synth-cpu:1.0

# Harbor runs `/tests/test.sh` inside this image, so the grader has to be copied
# in explicitly. The build context is the package's tests/ directory.
COPY grade.py /tests/grade.py
COPY test.sh /tests/test.sh

# The verifier runs in its own container after the agent's is gone, so it cannot
# rely on the agent's /data bind. Carry an identical copy of the series instead.
COPY series.csv /tests/series.csv

RUN chmod +x /tests/test.sh && mkdir -p /logs/verifier /logs/artifacts

WORKDIR /workspace
