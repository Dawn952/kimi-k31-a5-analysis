# PR20574 公共实现快照

来源：[vllm-project/vllm-ascend PR20574](https://github.com/vllm-project/vllm-ascend/pull/20574/files)，Apache-2.0；保留各文件原有版权声明，许可证见 [LICENSE](LICENSE)。

- head：`66f79fede7980c1e1192796ef0af648fe9a13106`
- base：`dcedb53a34f3850dfafdbf01263cfcc8fb82a932`
- `head/`：该 PR 的14个变更文件在指定 head 的完整公开源码，逐字节复制，未改写。
- [upstream.diff](upstream.diff)：原始 PR diff，便于只移植公共增量。
- [manifest.json](manifest.json)：固定来源及文件清单。

这是一份供移植与审查的源码快照，未接入推理服务。它含 GLM 专用实现；Kimi 接线和差异见[软件设计](../../design/README.md)。不包含 A5 私有运行源码。

优先阅读：

1. [配置](head/vllm_ascend/ascend_config.py)、[通信组](head/vllm_ascend/distributed/parallel_state.py)、[DP 同步条件](head/vllm_ascend/utils.py)。
2. [AG → projection → async A2A → owner restore](head/vllm_ascend/models/glm5next/kda_input_projection_tp.py)。
3. [GLM KDA core / norm / output 分离](head/vllm_ascend/models/glm5next/kda.py)。
4. [跨 rank 数值参考测试](head/tests/ut/models/test_glm5next_kda_input_projection_tp.py)及[配置测试](head/tests/ut/test_ascend_config.py)。
