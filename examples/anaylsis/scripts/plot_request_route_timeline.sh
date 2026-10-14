python plot_request_route_timeline.py \
  /apdcephfs_zwfy10/share_303541817/lhy/psrl/examples/mini_swe/megatron_psrl_log/sticky_thunder_agent_kv_aware_GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1/route_trace.log \
  -n 64 \
  -d 10 \
  -o sticky_request_route_timeline.png

python plot_request_route_timeline.py \
  /apdcephfs_zwfy10/share_303541817/lhy/psrl/examples/mini_swe/megatron_psrl_log/group_sticky_thunder_agent_kv_aware_GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1/route_trace.log \
  -n 64 \
  -d 10 \
  -o group_sticky_request_route_timeline.png
