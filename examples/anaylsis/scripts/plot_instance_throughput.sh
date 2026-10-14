python plot.py /jizhicfs/lhy/psrl_smg/examples/mini_swe/megatron_psrl_log/new_mig_async_GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1 \
--substring stats_r \
--xlabel "time(s)" \
--ylabel "generation throughput" \
--out new_mig_async_throughput.png \
--mode subplot \
--processor instance_throughput_indexed_by_time_smg