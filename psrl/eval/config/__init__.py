"""Hydra config groups for `psrl.eval.serve`.

This is a package rather than a plain directory so the YAML is addressable as a
config module (`initialize_config_module("psrl.eval.config")`), which is how tests
compose it without depending on a filesystem path. `psrl/trainer/config/` does the
same.
"""
