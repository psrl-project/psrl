FROM scibuddy-synth-cpu:1.0

# Harbor mounts task data through Compose, not through `bindings.json` (which is
# pipeline metadata). Baking the read-only input keeps the task self-contained.
COPY series.csv /data/series.csv
RUN chmod 0444 /data/series.csv

WORKDIR /workspace
