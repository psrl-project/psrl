请设计并实现一个最小可复现实验，用于验证：

在 vLLM 作为 rollout inference engine、FSDP trainer 作为 logprob recompute backend 的情况下，对于同一条最终生成 sequence，

1. 直接拼接各个 decode segment 当时记录的 rollout logprob；
2. 在每次 reprefill 后，用 reprefill 对已有完整 prefix 重新计算得到的 token logprob，覆盖之前 decode 阶段记录的历史 logprob；

这两种 rollout logprob 构造方式中，哪一种与 FSDP trainer 对最终固定 sequence 重新 forward 得到的 logprob 更接近。

不要修改 vLLM、verl 或 trainer 的已有源码。优先通过独立测试脚本、已有公开 API 或最小 monkey-patch / hook 完成实验。

实验背景：

rollout 的执行方式为：

prefill
→ decode 一个 segment
→ interrupt
→ 使用当前完整 prefix 重新 prefill
→ decode 下一个 segment
→ interrupt
→ 再次 reprefill
→ ...

直到得到最终 response：

\[
Y=(y_1,y_2,\ldots,y_T)
\]

需要同时构造三套 per-token logprob。

第一套：

\[
L^{concat}_t
\]

定义为 token \(y_t\) 在最初实际 decode 生成它时，由 vLLM 返回的 logprob。

最终 sequence 的 rollout logprob 直接由各 segment 的 decode logprob 拼接得到：

\[
L^{concat}
=
[
L^{decode,1},
L^{decode,2},
\ldots
]
\]

这是 baseline。

第二套：

\[
L^{replace}_t
\]

每发生一次 reprefill 时，都使用当前完整 prefix 对已经生成的 token 重新计算 logprob。

如果某个历史 token 在后续 reprefill 中重新获得了 logprob，则使用最新一次 reprefill 计算结果覆盖原先 decode 阶段记录的 logprob。

最终得到：

\[
L^{replace}
\]

要求明确记录每个 token 最终的 logprob 来源：

- original decode；
- 第几次 reprefill replacement。

重点是验证这种“reprefill 后替换历史 logprob”的方式是否可以消除 segmented decode execution path 引入的异常值。

第三套：

\[
L^{trainer}_t
\]

最终 rollout 完成后，固定：

- prompt token ids；
- response token ids；

由 FSDP trainer 使用当前对应的 actor weight snapshot，对完整：

\[
[prompt, response]
\]

执行一次标准 teacher-forcing forward / recompute，得到每个 response token 的：

\[
\log P(y_t|x,y_{<t})
\]

作为 trainer reference logprob。

必须保证 vLLM rollout 和 FSDP recompute 使用完全相同的模型权重 snapshot。

本实验不研究训练 update，只比较 logprob consistency。

不要执行 optimizer step。

不要更新模型参数。

不要涉及 reward、advantage、DAPO loss 或 PPO loss。

核心问题：

比较：

\[
\Delta^{concat}_t
=
L^{trainer}_t-L^{concat}_t
\]

和：

\[
\Delta^{replace}_t
=
L^{trainer}_t-L^{replace}_t
\]

判断 replacement 是否使 mismatch：

1. mean 更接近 0；
2. median 更接近 0；
3. mean absolute error 更低；
4. p95 / p99 absolute error 更低；
5. max absolute error 更低；
6. extreme outlier token 数量减少；
7. logprob ratio 更集中在 1 附近。

同时重点判断：

> replacement 后的 mismatch 是否从明显的 segment / reprefill-boundary related outlier，回归到 rollout-vs-trainer 本身固有的平均 numerical mismatch 水平。

实验配置要求：

A. 模型和权重

固定一个模型。

vLLM 和 FSDP 必须加载完全相同的 checkpoint。

如果存在 weight synchronization 流程，必须确认在 rollout 开始到 trainer recompute 完成期间权重没有变化。

记录：

- model name；
- commit / checkpoint；
- model dtype；
- vocab size；
- tokenizer；
- rope / position 配置。

B. vLLM rollout 配置

固定并记录：

- vLLM version；
- dtype；
- KV cache dtype；
- Tensor Parallel size；
- attention backend；
- chunked prefill 是否开启；
- prefix caching 是否开启；
- CUDA graph 配置；
- max model length；
- max_num_seqs；
- GPU memory utilization；
- sampling temperature；
- top-p；
- top-k；
- logprob 返回配置。

第一阶段尽量使用简单稳定配置，例如：

- BF16 model；
- BF16 KV cache；
- TP=1，如果显存允许；
- batch size=1；
- 无其他并发 request；
- 尽量关闭 prefix caching；
- 尽量减少其他会改变 execution path 的变量。

如果模型无法 TP=1，再固定一个 TP 值。

C. FSDP trainer 配置

固定并记录：

- PyTorch version；
- transformers version；
- FSDP 配置；
- model dtype；
- mixed precision 配置；
- Tensor Parallel / Sequence Parallel，如果有；
- attention implementation；
- 是否使用 flash attention；
- logits 是否转 FP32 后做 log_softmax。

trainer recompute 必须：

- eval mode；
- no_grad；
- dropout disabled；
- 不做 optimizer update；
- 对最终固定 token sequence 做一次 teacher-forced forward。

D. segmented rollout

测试多个 segment length，例如：

- 32；
- 64；
- 128；
- 256。

对于每一种 segment length：

1. 初始 prompt prefill；
2. decode N 个 token；
3. interrupt；
4. 使用 prompt + 当前所有已生成 token 重新 prefill；
5. 获取当前完整 prefix 对历史 response token 的 logprob；
6. 用这些 reprefill logprob 更新 replacement buffer；
7. 再继续 decode N 个 token；
8. 重复直到达到固定总 response length。

建议固定 response length，例如：

- 512；
- 1024；

如果模型/context 允许，可增加 2048。

E. 必须保留的三个数组

最终每个 sample 至少保存：

```python
token_ids
logprob_concat
logprob_replace
logprob_trainer
```

另外保存：

```python
token_position
segment_id
reprefill_count_before_token
is_segment_boundary
distance_to_previous_boundary
distance_to_next_boundary
replacement_source
```

其中：

```python
replacement_source
```

至少能区分：

```text
decode
reprefill_1
reprefill_2
...
```

F. 对齐要求

这是实验最重要的部分。

必须验证：

\[
token\_ids^{rollout}
=
token\_ids^{trainer}
\]

所有 logprob 必须对应完全相同的 token：

\[
\log P(y_t|x,y_{<t})
\]

特别注意 causal shift。

例如 trainer logits 的 position \(i\) 对应的是下一个 token，而不是当前位置 token。

在正式统计前，打印前 10～20 个 token：

```text
position
token_id
token_text
concat_logprob
replace_logprob
trainer_logprob
```

人工确认不存在 off-by-one。

G. replacement 的定义

需要明确实现并报告 replacement 的具体语义。

假设已经生成：

\[
y_1,\ldots,y_{128}
\]

发生第一次 reprefill 后，如果能够获得：

\[
\log P(y_1|x),
\ldots,
\log P(y_{128}|x,y_{<128})
\]

则：

```python
logprob_replace[0:128] = reprefill_logprobs
```

之后生成：

\[
y_{129},\ldots,y_{256}
\]

第二次 reprefill 后，如果能够重新得到前 256 个 response token 的 logprob，则：

```python
logprob_replace[0:256] = second_reprefill_logprobs
```

也就是说，最终：

\[
L^{replace}_t
\]

采用“最后一次覆盖该 token 的 reprefill logprob”。

同时保留原始：

\[
L^{concat}_t
\]

绝对不能覆盖 baseline 数据。

H. 核心统计

分别对：

```text
trainer - concat
trainer - replace
```

计算：

- signed mean；
- signed median；
- mean absolute error；
- median absolute error；
- std；
- p90 absolute delta；
- p95 absolute delta；
- p99 absolute delta；
- p99.9 absolute delta；
- max absolute delta。

另外计算 importance ratio：

\[
r^{concat}_t
=
\exp(
L^{trainer}_t-L^{concat}_t
)
\]

和：

\[
r^{replace}_t
=
\exp(
L^{trainer}_t-L^{replace}_t
)
\]

统计：

- mean；
- median；
- p95；
- p99；
- max。

并统计假设 RS threshold 为：

\[
[0.5,2.0]
\]

时：

```text
concat hypothetical masked fraction
replace hypothetical masked fraction
```

即：

\[
r_t<0.5
\quad\text{or}\quad
r_t>2.0
\]

的 token 比例。

这只是离线统计，不真正运行 RS。

I. boundary analysis

重点验证 mismatch 是否集中在 reprefill boundary 附近。

对于每个 boundary \(b\)，统计：

\[
t-b
\]

处于：

```text
-8 ~ -1
0
+1 ~ +8
+9 ~ +32
far from boundary
```

时：

\[
|\Delta^{concat}|
\]

和：

\[
|\Delta^{replace}|
\]

的均值和 p95/p99。

尤其回答：

1. concat logprob 是否在 segment 开头 / reprefill 后几个 token 出现明显 spike；
2. replacement 后这些 spike 是否消失；
3. replacement 后是否只剩一个相对平稳的 vLLM-vs-FSDP baseline gap。

J. position analysis

按 absolute response position 分桶：

```text
0-127
128-255
256-383
384-511
...
```

分别比较：

```text
concat vs trainer
replace vs trainer
```

判断 mismatch 是否随 sequence length 增长。

K. 重复样本

不要只跑一条 sequence。

至少：

- 多个 prompt；
- 每个配置运行足够 token 数；

建议总 response token 数至少达到数万 token，避免只依赖个别 sequence。

如果测试成本较高，可以先做小规模 sanity check，再扩大统计。

L. 必须增加一个 continuous baseline

除了 segmented rollout，还需要增加一个：

```text
prefill
→ continuous decode
→ finish
```

baseline。

得到：

\[
L^{continuous}
\]

然后同样与 FSDP trainer 比：

\[
\Delta^{continuous}
=
L^{trainer}
-
L^{continuous}
\]

这样最终需要比较三种：

\[
L^{continuous}
\]

\[
L^{concat}
\]

\[
L^{replace}
\]

相对于：

\[
L^{trainer}
\]

的误差分布。

核心判断是：

如果：

\[
error(replace, trainer)
\approx
error(continuous, trainer)
\]

并且：

\[
error(concat, trainer)
\gg
error(continuous, trainer)
\]

则支持以下假设：

> segmented decode / reprefill execution path 给 rollout logprob 引入了额外 mismatch，而使用 reprefill logprob 回填历史 token 后，可以把这部分额外 mismatch 大幅消除，使 rollout-vs-trainer 差异回归到正常 baseline。

反之，如果：

\[
error(replace, trainer)
\]

没有明显改善，或者甚至更差，则说明：

- reprefill 并不是主要 mismatch 来源；
- 或 reprefill scoring path 本身也存在明显 numerical mismatch；
- 或主要问题来自 vLLM 与 FSDP trainer backend 差异。

M. 输出图表

至少输出：

1. 每 token：

```text
trainer - concat
trainer - replace
```

随 token position 的曲线。

2. absolute delta 的 histogram / ECDF。

3. concat 和 replace 的 p50/p95/p99/max 对比。

4. hypothetical RS masked fraction 对比。

5. mismatch 与 boundary distance 的关系。

6. 不同 segment length 下：

```text
continuous
concat
replace
```

三种方案的误差表。

N. 最终报告必须回答

1. segmented rollout 直接拼接 decode logprob，相比 continuous decode，是否产生额外的 rollout-vs-FSDP mismatch？

2. 使用 reprefill 后重新得到的历史 token logprob覆盖原始 decode logprob，是否显著降低：

\[
|L^{trainer}-L^{rollout}|
\]

3. replacement 后：

- mean；
- p95；
- p99；
- max；
- hypothetical RS masked fraction；

是否接近 continuous baseline？

4. 改善主要来自：

- 整体 mean shift 减少；
- 还是 extreme tail / outlier 减少？

重点区分这两者。

5. mismatch 是否集中在 segment boundary 或 reprefill 后的前几个 token？

6. replacement 能否把 boundary spike 消除？

7. replacement 后剩余 mismatch 是否更像稳定的 vLLM-vs-FSDP backend baseline？

8. 如果 replacement 明显有效，是否支持将其作为 DAPO rollout logprob canonicalization 的候选方案？

注意这里只给实验结论，不直接修改训练算法。

O. 最关键的验收标准

重点不是要求：

\[
L^{replace}=L^{trainer}
\]

完全一致。

真正需要验证的是：

\[
\boxed{
L^{replace}
\text{ 是否把 segmented execution 引入的额外误差消掉}
}
\]

也就是：

\[
\boxed{
Distribution(
L^{trainer}-L^{replace}
)
\approx
Distribution(
L^{trainer}-L^{continuous}
)
}
\]

而：

\[
Distribution(
L^{trainer}-L^{concat}
)
\]

明显更宽、tail 更重。

如果这一现象成立，就说明 reprefill replacement 有可能把 rollout logprob mismatch 恢复到正常的 vLLM-vs-FSDP baseline 水平，而不是继续携带 segmented decode / reprefill 造成的额外 numerical artifact。