python -m psrl.bench.chunked_prefill.main_micro \
    model.path=/apdcephfs_zwfy10/share_303541817/lhy/models/SWE-agent-LM-7B \
    micro.experiment=e_chunked \
    "micro.e_chunked.total_lens=[8192]" \
    "micro.e_chunked.chunk_sizes=[512,2048]" \
    "micro.e_chunked.prefix_lens=[8192]" \
    micro.e_chunked.warmup=1 micro.e_chunked.iters=3 \
    rollout.max_num_batched_tokens=2048