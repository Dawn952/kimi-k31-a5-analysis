# 全局 TP>1 与细粒度 TP：当前支持结论

核对日期2026-11-11。A5 Ascend 源码仍为 `30572def8290cf08b300a2326b09f9c4abe8f30f`，上游 vLLM 为 `24420b7a341af37a9bc3186e6fc6062512e0aa58`；相关 tracked 源码没有修改。本文纠正此前“embedding/LM head 仅配置未禁止”的不充分表述。

## 结论

**当前 embedding 和 LM head 的细粒度 TP 实现支持与全局 TP>1 共存。** 这是已有的完整分组、权重加载和 forward 路径，不只是缺少一个配置拦截。Kimi 调用这些通用层，没有另外禁止全局 TP；其 SP 模型输出在进入 LM head 前已经恢复完整 token 行。

本次调用真实生产类做 CPU 验证：原有46项、新增36项，共 **82 passed**。新增检查覆盖 TP2/4/8、DP4/8，以及可整除 DP 的 fineTP2/4/8；使用V64/H4的FP32合成小权重，验证实际 loader 的切片、embedding lookup/mask/owner返回、LM head GEMM/A2A重排及 idle owner。通信以数学参考替换，**没有启动模型、HCCL通信或 ACL graph**，不能把这次结果表述为真实量化checkpoint或 K3.1 新拓扑的设备端验收。

| 模块 | 当前全局 TP>1 + fineTP>1 | 直接原因 |
|---|---|---|
| embedding | 支持 | 词表分片切换到 DP轴PG，输出已是完整 hidden，不再叠加原TP归约 |
| LM head | 支持，需满足采样约束 | 使用 fineTP hidden AG / GEMM / logits A2A，返回完整词表，不再做原TP logits gather |
| O | 配置明确拒绝 | Kimi 原 attention 输出已经按 native TP 切头；当前 fineTP O 交换按完整头维解释输入，布局与归约契约不兼容 |
| dense MLP | 配置明确拒绝 | 上游说明该组合未验证；Kimi SP 包装还要求 native TP partial输出，直接替换为完整 fineTP输出会产生重复归约风险 |
| QKV/beta | PR20574只支持TP1；Kimi新增方案尚未实现 | helper还回全局完整头，core/state却按原TP保留局部头，缺少两轴头序和loader接线 |
| full-rank G | A5尚无该开关/实现；设计第一版限制TP1 | 新增能力范围，不是现有硬件或数学禁止；尚未接通 native TP head map 与独立G localTP |

## Embedding / LM head 为什么能够共存

### 分组按 DP 轴，固定 native TP 下标

`distributed/parallel_state.py:202–218` 把 rank 视为 `[PP,DP,TP]`，在每个固定 `tp_idx` 上选 S 个 DP rank。

例如32卡 `TP4,DP8,S8`，PP1时：

```text
native TP rank0 的 fineTP组：0,4,8,12,16,20,24,28
native TP rank1 的 fineTP组：1,5,9,13,17,21,25,29
native TP rank2 的 fineTP组：2,6,10,14,18,22,26,30
native TP rank3 的 fineTP组：3,7,11,15,19,23,27,31
```

这里是逻辑 rank 列表，不保证物理同节点。同一个 DP owner 的四个 native TP rank，在这两个模块上持有相同 vocab shard，但各自沿自己的 fineTP组完成其他 DP owner 的词表贡献。

### 是替换词表分片，不是两轴相乘

在 `ops/vocab_parallel_embedding.py:93–159`，`_resolve_vocab_parallel_plan` 优先选择 `FINE_GRAINED`；`tp_rank/tp_size` 改为模块自己的 DP轴组 rank/size。后续实际权重分配与继承 loader 都使用这个 size。

因此每卡逻辑词表权重为：

```text
原 native TP：       [V / T, H]
开启模块 fineTP S：  [V / S, H]
不是：              [V / (T*S), H]
```

- embedding：固定容量 IDs AG → 当前 vocab shard lookup/mask → fineTP RS → owner完整 `[T_local,H]`，`output_reduce_group=None`，没有原TP all-reduce。
- LM head：固定容量 hidden AG → 当前 vocab shard GEMM → fineTP logits A2A → owner完整词表。`AscendLogitsProcessor._get_logits` 根据实际 `parallel_mode` 选专用分支，不再调用 native TP `_gather_logits`。
- Kimi SP：`models/kimi_k3/kimi_k3.py:1252–1267` 在模型出口 `sp_all_gather` 恢复完整 token 行，再交给 logits 路径；不能把 decoder内部的 SP 行分片直接当作 LM head 输入。

上游固定版本[使用文档](https://github.com/vllm-project/vllm-ascend/blob/66f79fede7980c1e1192796ef0af648fe9a13106/docs/source/user_guide/feature_guide/Fine_grained_TP.md#standard-tensor-parallelism-requirement)也明确支持 embedding/LM head 与标准TP共存；其 standalone 示例使用 TP4、DP4、两个模块fineTP4。

### 条件与显存含义

1. 模型为 MoE，fineTP size整除DP，词表padding支持该分片；建议核对的明确配置为 PP1/PCP1。只开 embedding/head 时不要求 O/MLP 的 PD decode/graph 条件。
2. PCP默认开启的 `PCP_X_TP` 词表分片与 fineTP 不可同时开；`_resolve_vocab_parallel_plan` 会明确报错。它们是两种不同词表方案。
3. **LM head fineTP 与 `enable_batch_sharded_sampling=true` 互斥**，`ascend_config.py:982–990` 明确拒绝；本基线上游该采样开关默认False。仅 embedding 不受这个 LM head 限制。
4. LM head fineTP 当前不支持 prompt logprobs / fixed-token prompt scoring；runner在这些请求出现时抛 `NotImplementedError`。所有DP peer仍须按固定/同步容量调用，包括 idle dummy logits。
5. DCP>1的runner保留完整 LM head容量，而非DCP1的缩小pad分支；本次CPU检查没有证明完整Kimi DCP集成。`skip_gather` 特殊调用也不能被当作已重建完整词表的普通采样路径。

对同一模块，细粒度开启前后每卡权重比为 `T/S`，**只有 S>T 才比原 native TP 词表权重更小**。例如：

| 32卡拓扑 | fineTP S | 相对原native TP词表权重 |
|---|---:|---:|
| TP4/DP8 | 8 | 1/2 |
| TP4/DP8 | 4 | 相同 |
| TP8/DP4 | 4 | 2倍 |
| TP8/DP4 | 8 | 非法：8不能整除DP4 |

这不表示模型其他层停止 native TP：只是这两个模块切换词表分片域。不能把 pureDP32/localTP8 的词表收益原样带到 TP8/DP4。

## O 的限制：有具体布局冲突

`ascend_config.py:1389–1396` 对 O 或 MLP fineTP 与 native TP>1 直接抛 `AssertionError`。

对Kimi，设全局 attention头维 `P=12288`、native TP为T，则core/UV的本 rank输出为 `[C,P/T]`。现有 `OProjRowParallelOp` 的权重切成 `[H,P/S]`，`ops/linear_op.py:368–382` 却把输入当作 `[C,P]` 重排为 `[S,C,P/S]`，再分配接收 `S*C*(P/S)=C*P` 个元素。

当 `T=4,S=8` 时，实际输入 `[C,3072]`，该协议需要 `[C,12288]`；元素数差4倍。可能在reshape报错，或在token数偶然能reshape时进入错误的collective计数。**只删除TP1校验不能修复它。**

此外，fineTP RS只合并固定native TP下标对应DP子组内的 O贡献，不会替代 native TP各自头片段之间的求和；Kimi SP路径还会在decoder继续执行TP reduce-scatter。

未来可选：在已有native TP头片段内再分S份，权重成为 `[H,P/(T*S)]`、最后保留native TP归约；或先恢复native TP完整头，再切换到替代式O分片。两种方案通信与归约都不同，均非现有开关直接支持。上游文档还记录DSV4分组O的 `n_local_groups` 与 OTP权重布局冲突，但不能把该模型的wo_a/wo_b原因冒充Kimi源码。

## Dense MLP 的限制：保守禁用，并有 SP 接线问题

现配置把MLP与O一起限制TP1，注释对MLP的理由就是 **untested**，并非已证明所有TP>1情形必然计算错误。

`MLPColumnParallelOp` / `MLPRowParallelOp` 用DP轴组替换中间维分片，做AG→本地gate/up→down→RS。若输入已复制、SP关闭，它在数学上可以像vocab一样替换native TP分片；当前仍被配置保护挡住，没有宣称支持。

Kimi SP有具体的待适配点：`AscendKimiMLP.forward` 在MLP外部执行 `sp_all_gather → MLP → sp_reduce_scatter`，原期望MLP返回native TP的部分和。fineTP down RS却已返回完整FFN结果；同DP owner的各native TP rank重复计算同一完整结果，再SP求和会重复计入。这是按当前调用链推导的冲突，不能靠复用一个 `tp_size` 字段解决。

要开放支持，需明确选择“native TP×fineTP两轴切分”或“fineTP替换native TP、SP末尾只分行不再求和”，再补量化slice、residual/token布局和graph验证。不能以“尚未验证”泛化成算法上不可能。

## QKV/beta 与 full-rank G 的限制不同

PR20574配置明确检查 native TP=PCP=PP=1。其 helper根据**全局** `num_heads/S` 投影并返回全部 `num_heads`，而layer的 `local_num_heads`、core输出与state按 `num_heads/T` 分配，F/G上投影也仍在原head域。TP1使两者刚好相同；TP>1则没有完成头片段映射，直接去掉检查会产生shape/语义冲突。

Kimi QKV适配尚未接入A5，更不能称为当前已支持TP1以外的版本。要做两轴投影，应在每个native TP rank内使用 `N/T` 个core头，localTP只再切这部分，返回owner的 `N/T` 头，连同MXFP8 scale、beta、conv/F参数head索引一起对齐；state仍按原native TP分区。

**full-rank G则是新方案新增能力，当前A5连独立开关都没有。** 本设计第一版限制TP1是交付边界。未来G可在native TP已有头片段内切分再还owner，或与匹配的O片段融合；不存在“G数学上必须TP1”的结论。

## 可复现检查与证据范围

[test_tp_gt1_runtime.py](test_tp_gt1_runtime.py)运行于上述A5实际Python环境，导入实际配置类、embedding/head类和logits processor。仅替换拓扑及集体通信，真实权重加载、lookup/mask、GEMM、A2A reshape均执行CPU路径。

- 15种 `(TP,DP,S)` × embedding/LM head =30项。
- TP2/4/8 × O/MLP启动拒绝 =6项。
- 既有词表、A2A、LM head容量UT =46项。
- 合计82项通过；metadata汇总见 [tp-compatibility.json](../data/tp-compatibility.json)。

示例命令（在源码仓库，`/path/to/`替换为脚本的实际位置）：

```bash
python -m pytest -q \
  tests/ut/ops/test_vocab_parallel_embedding.py \
  tests/ut/ops/test_lmhead_all_to_all.py \
  tests/ut/worker/v2/test_lmhead_tp_utils.py \
  /path/to/test_tp_gt1_runtime.py
```

检查未修改推理源码；未测试真实HCCL、模型精度、DCP整链或设备多流。这里的明确结论是“当前模块实现支持TP>1，且CPU验证通过”，不是“整个部署已完成设备端验收”。
