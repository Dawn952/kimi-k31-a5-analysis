# Kimi K3.1 · A5 localTP 分析

[打开交互网页](https://dawn952.github.io/kimi-k31-a5-analysis/#lp-flow) · [1M 显存计算器 · A3 / A5](https://dawn952.github.io/kimi-k31-a5-analysis/memory.html)

网页维护最终的分析内容。计算与通信依赖流采用本地实际 A5 localTP profiling；权重、payload 和时间预算提供独立的参数估算。

## 计算与通信依赖流

采样部署为 Ascend `3634a7eae94897d1a7b14d00807323e5af6b4a04`、vLLM `4ba787ec9959e566d3b587a796dff07673324da5`。真实 P/D，D 为 DP32/TP1/localTP8；全局并发128，输入131072 tokens、输出1024 tokens，DSpark3/固定采信2.7。采样日期2026-11-13，单rank、warmup阶段的10次worker step。

| 模块 | 实际执行关系 | 匹配样本 |
|---|---|---:|
| KDA | input AG 与 F；先 G GEMM/返回，再 QKV；QKV/beta 合并返回后，主流重排和 beta 后处理，再 Conv/Recurrent、gated norm、O A2A→GEMM→RS | 690 |
| DSA | G input AG 与 MLA prolog 相交；G 投影/返回靠前；Indexer、SFA、UV、owner gate、O 输出链路 | 240 |
| Shared expert | shared/routed 保留真实执行顺序；所列 AG/latent-down、RS/latent-up、gate-up/GMM1、down/combine 交集均为0 | 920 |
| Dense MLP | 双 chunk：AG0→AG1→RS0→RS1；AG1/up0 和 RS0/up1 有实际交集 | 10 |
| LM head | 双 chunk：两次 hidden AG、两次 GEMM、两次 logits A2A，owner 重排与 Concat 后进入 sampler | 10 |
| Embedding | 双 chunk IDs AG→lookup/mask→vectors RS；本 capture 的 RS 标记为 AicpuKernel | 20 |

Embedding 按512-row lookup双chunk形状匹配，CSV本身未区分target/draft调用者。图中计算节点标注真实kernel类型和stream；点击节点可查看首例CSV行号与相对时间。多笔kernel的顺序组不冒充融合算子。

彩框表示全部匹配样本的设备区间交集中位数；节点时间来自同一个首例调用。依赖布局不是按比例的时间线，交集也不是端到端收益。配置开启多流不等于设备上已出现所选窗口的并发。

- 去标识证据：[localtp-profile-20261113.json](data/localtp-profile-20261113.json)。
- 静态图：[KDA](docs/images/a5-kda-localtp.png)、[DSA](docs/images/a5-dsa-localtp.png)、[Shared](docs/images/a5-shared-localtp.png)、[Dense](docs/images/a5-dense-localtp.png)、[Head](docs/images/a5-head-localtp.png)、[Embedding](docs/images/a5-embedding-localtp.png)。每张图有同名SVG可下载。
- 图模型：[localtp-profile.js](src/localtp-profile.js)；交互视图：[localtp-view.js](src/localtp-view.js)。
- 证据提取：[extract-localtp-profile.py](scripts/extract-localtp-profile.py)。只发布精简设备窗口、相对时间和必要统计，不包含原始trace、权重、机器地址或运行环境私有源码。

## 权重、通信和显存口径

localTP沿DP轴组成子组，投影权重按通道切分；KV/recurrent状态仍属于原请求所在rank。负载均衡、无额外padding时，每卡以1/P权重处理P个owner的token，投影FLOPs基本不变。

参数计算器默认对齐本capture：localTP8，每owner `T=C=16`，embedding pad `E=128`，head `L=16`。改变参数只改变公式估算，不改写实测依赖图。每卡发送量与接收量分别计量；带宽、launch和HBM参数不是从通信envelope反推的实测带宽。

G和O的checkpoint逻辑矩阵分别为 `[12288,7168]`、`[7168,12288]`，均BF16，每层各168MiB，localTP8后各21MiB/card。数值独立、不共享权重。[权重核对](data/weight-shapes.json)。

共享专家源码学习基线为 [PR21624](https://github.com/vllm-project/vllm-ascend/pull/21624) 的 `976ea36e9`，与profiling部署版本分别标注；源码可用能力不等于该capture已测效果。显存计算器使用其页面所列物理池、cache布局与运行时校准证据，权重估算不等于峰值HBM。

## 构建和校验

网页为无运行依赖的单文件 `docs/index.html`，可离线打开。

```bash
node scripts/build.mjs
node scripts/export-profile-diagrams.mjs
python scripts/render-profile-diagrams.py
node --test tests/*.cjs
```

PNG渲染使用Pillow与系统中文字体。重新提取证据时传入本地CSV路径：

```bash
python scripts/extract-localtp-profile.py /path/to/kernel_details.csv
```
