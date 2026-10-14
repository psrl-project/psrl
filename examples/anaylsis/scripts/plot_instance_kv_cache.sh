python plot.py /jizhicfs/lhy/psrl_smg/examples/mini_swe/megatron_psrl_log/new_mig_async_GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1 \
--substring stats_r \
--xlabel "time(s)" \
--ylabel "kv cache usage" \
--out new_mig_async_kv_cache.png \
--mode subplot \
--processor instance_kv_cache_indexed_by_time_smg
