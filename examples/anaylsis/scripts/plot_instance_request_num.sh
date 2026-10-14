'''
python plot.py ../paper_exp/e2e/logs/14b_staleness_1_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 14b_instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../paper_exp/e2e/logs/32b_staleness_1_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 32b_instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../paper_exp/e2e/logs/14b_staleness_1_ours \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 14b_instance_request_num_ours.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../paper_exp/e2e/logs/4+4_moe_staleness_1_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out moe_instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time
'''

'''
python plot.py /jizhicfs/lhy/verl/lhy_exp/ablation_new \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out ablation_verl.png \
--mode subplot \
--processor instance_request_num_indexed_by_time
'''

python plot.py /apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/dapo_trainer/megatron_psrl_log/qwen3_8b_megatron_dcp_preempt \
--substring stats_r \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out preempt_qwen3_8b_megatron_dcp_request_num.png \
--mode subplot \
--processor instance_request_num_indexed_by_time_smg