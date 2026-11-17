/* Source-aligned lesson; independent from the historical communication ledger. */
(function(root){
  const head='976ea36e9720028435d5ddc7d23600f6176b7f62';
  const source=(file,line)=>`https://github.com/Dawn952/vllm-ascend/blob/${head}/vllm_ascend/${file}#L${line}`;
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1380 615" role="img" aria-labelledby="shared-title shared-desc">
  <title id="shared-title">Shared expert localTP：权重分片与 owner 行返回</title>
  <desc id="shared-desc">P 个 DP owner 各持 C 行；all-gather 后每卡处理 P C 行，gate 和 up 沿中间维切 P 份，down 沿输入维匹配切片。reduce-scatter 求和并返回原 owner。通信属于 runner，Linear 不另做 collective。</desc>
  <defs><marker id="shared-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#73e4cb"/></marker></defs>
  <rect width="1380" height="615" rx="20" fill="#0d1928"/>
  <g font-family="Segoe UI,Microsoft YaHei,sans-serif" fill="#e5eff8">
  <text x="30" y="40" font-size="23" font-weight="700">01 / Shared expert localTP · 当前生产代码</text>
  <text x="30" y="69" font-size="16" fill="#a9bfce">PR21624 · 976ea36e9 · P = shared TP size；C = 每 owner 统一容量；H = hidden；M = intermediate</text>
  <g fill="#162b3e" stroke="#355367" stroke-width="1.5">
    <rect x="30" y="105" width="175" height="155" rx="12"/>
    <rect x="245" y="105" width="175" height="155" rx="12"/>
    <rect x="460" y="105" width="210" height="155" rx="12"/>
    <rect x="710" y="105" width="185" height="155" rx="12"/>
    <rect x="935" y="105" width="175" height="155" rx="12"/>
    <rect x="1150" y="105" width="200" height="155" rx="12"/>
  </g>
  <g font-size="18" font-weight="600">
    <text x="46" y="138">DP owner p</text><text x="261" y="138">Input AG</text>
    <text x="476" y="138">Gate + Up shard</text><text x="726" y="138">Activation</text>
    <text x="951" y="138">Down shard</text><text x="1166" y="138">Output RS</text>
  </g>
  <g font-size="16" fill="#b9d0de">
    <text x="46" y="175">X_p: [C,H]</text><text x="46" y="204">各 owner 不同请求</text><text x="46" y="235">padding 已统一</text>
    <text x="261" y="175">X: [P·C,H]</text><text x="261" y="204">按 owner 顺序拼行</text><text x="261" y="235">由 runner 发起</text>
    <text x="476" y="175">Wg/Wu: [M/P,H]</text><text x="476" y="204">各得 [P·C,M/P]</text><text x="476" y="235">同一中间维切片</text>
    <text x="726" y="175">配置的激活函数</text><text x="726" y="204">Z: [P·C,M/P]</text><text x="726" y="235">量化由 scheme 决定</text>
    <text x="951" y="175">Wd: [H,M/P]</text><text x="951" y="204">partial [P·C,H]</text><text x="951" y="235">Linear 不归约</text>
    <text x="1166" y="175">先求和，再分行</text><text x="1166" y="204">owner p: [C,H]</text><text x="1166" y="235">与 routed 输出相加</text>
  </g>
  <g stroke="#73e4cb" stroke-width="2" marker-end="url(#shared-arrow)"><path d="M205 181H239"/><path d="M420 181H454"/><path d="M670 181H704"/><path d="M895 181H929"/><path d="M1110 181H1144"/></g>
  <rect x="30" y="288" width="1320" height="80" rx="12" fill="#143631"/>
  <text x="50" y="319" font-size="19" fill="#91f3d4">设计目的：每卡投影权重约为完整权重的 1/P；均衡、无额外 padding 时，投影 FLOPs 基本不变。</text>
  <text x="50" y="348" font-size="17" fill="#c3d9d4">每卡从 C 行 × 完整 M，变成 P·C 行 × M/P；新增 AG/RS、输入量化和 padding 成本须另计；activation 元素数不因 P 增大，不能据此宣称加速。</text>
  <g font-size="17">
  <text x="30" y="412" fill="#73e4cb">公共 localTP：配置与组管理；相同 size 复用 group；DP 请求容量 / graph 协调。</text>
  <text x="30" y="451" fill="#8dbdff">Shared 域公共能力：配对切 gate/up/down，AG → MLP → RS；GLM 等可复用，须满足布局与量化契约。</text>
  <text x="30" y="490" fill="#e9bc79">Kimi 可用窗口：开启多流且有 latent 投影：AG ↔ routed down；RS ↔ routed up；基本路径不依赖这些投影。</text>
  <text x="30" y="529" fill="#b9d0de">A3 INT8 与 A5 MXFP：权重布局 / AG / RS 相同；动态量化、scale、激活融合与 matmul 的 scheme 不同。</text>
  <text x="30" y="568" fill="#b9d0de">按 shared Linear 的实际 scheme 选路；W8A8MXFP 保留 Linear；真实 INT4 / FP8 保留自身 wrapper。</text>
  <text x="30" y="597" font-size="14" fill="#93adbf">逻辑 shape；量化后的物理转置、packing、scale 布局由 quant scheme 管理。本图为源码事实，非时序实测。</text>
  </g></svg>`;
  const link=(label,file,line)=>`<a href="${source(file,line)}">${label}</a>`;
  function render(){return `<section id="shared-current" class="panel shared-current" style="margin-bottom:28px">
    <style>.shared-current svg{display:block;width:100%;height:auto;margin:20px 0}.shared-current p,.shared-current li{line-height:1.9}.shared-current code{overflow-wrap:anywhere}.shared-current .shared-scroll{overflow-x:auto}.shared-current table{min-width:640px}</style>
    <div class="eyebrow">2026-11-17 / 当前实现 · 学习顺序 01</div><h2>共享专家：减少每卡权重，按 owner 返回结果</h2>
    <p>源码基线 <code>${head}</code>，已整合 main <code>8cd3c3762</code>。<a href="https://github.com/vllm-project/vllm-ascend/pull/21624">Open PR21624</a>。这是当前 shared 域；下方早期通信账本与 11 月 13 日 KDA 实测保留各自版本，不计入本图。</p>
    ${svg}
    <div class="shared-scroll"><table class="data-table"><thead><tr><th>逐行入口</th><th>链路职责 / 必要性</th><th>能力归属</th></tr></thead><tbody>
    <tr><td>${link('ascend_config.py:1440 / 1453','ascend_config.py',1440)}</td><td>独立 shared size；进入 graph 契约，不随 dense MLP 开关切换。拓扑与运行限制须按具体链路验证。</td><td>localTP 公共配置</td></tr>
    <tr><td>${link('parallel_state.py:213 / 248','distributed/parallel_state.py',213)}</td><td>沿 DP 维取连续 P 个 rank；同尺寸域复用 coordinator。共享组不代表共享权重或共享流。</td><td>所有 localTP 域公共</td></tr>
    <tr><td>${link('linear_op.py:457–461','ops/linear_op.py',457)}</td><td>shared 前缀与显式开关选独立 op，返回该组的 rank/size 给权重加载。优先于通用 disable_tp，避免 SP 标志误取消 shared 分片。</td><td>Shared 域公共</td></tr>
    <tr><td>${link('linear_op.py:281–304','ops/linear_op.py',281)}</td><td>Column/Row 只运行局部 quant method；Row 的 bias 仅 rank 0 加一次。collective 交 runner，不能恢复基类自动归约。</td><td>Shared 域公共</td></tr>
    <tr><td>${link('shared_experts.py:686–712','ops/fused_moe/shared_experts.py',686)}</td><td>AG 的行属于不同 DP 请求，不能按单个 owner 的 num_tokens 截断；RS 同时求和并归还 owner 行。</td><td>Shared 域公共</td></tr>
    <tr><td>${link('shared_experts.py:1139–1171','ops/fused_moe/shared_experts.py',1139)}</td><td>从两层 shared Linear 的实际 scheme 读取类型；类型一致才尝试专用流水，未知/不同类型保留 wrapper。W8A8MXFP 再检查实际 Down metadata。</td><td>量化 scheme 公共能力</td></tr>
    <tr><td>${link('fused_moe.py:413–435','ops/fused_moe/fused_moe.py',413)}</td><td>开启 shared 多流且存在对应 transform 时，AG 与输入投影重叠、RS 与输出投影重叠；事件保证次序。</td><td>公共接线，Kimi latent 投影受益</td></tr>
    <tr><td>${link('fused_moe.py:367–376','ops/fused_moe/fused_moe.py',367)}</td><td>localTP 的 RS 已求和；此处跳过 shared 分量的标准 TP 规约。最终输出另有 TP 收口，本模式全局 TP=1。</td><td>MoE 公共输出收口</td></tr>
    </tbody></table></div>
    <p><b>A3 / A5 的差异边界：</b>A3 已采样的 shared checkpoint 是 <code>W8A8_DYNAMIC</code>，对应 INT8 动态量化路径。shared MLP 从 <code>layer.*_proj.quant_method.quant_method</code> 读取实际 scheme 类型，不能由整模型 W4A8 名称推断 shared 的位宽。实际 <code>W8A8MXFP</code> 保留两层 Linear，只在 Down metadata 满足条件时融合激活量化；实际 <code>W4A8MXFP</code> 使用显式 MXFP 流水；真实 INT4 / 普通 FP8 走自身 wrapper，保留 GMM 补偿与原量化数学。<code>self.quant_type</code> 仍是 routed 类型，仅保留其 routed/MegaMoe 用途。这里未做新的 A3/A5 整模性能测量。</p>
    <p><b>必要性审视：</b>权重切分、匹配的 AG/RS、owner 顺序与退出等待是正确性条件；多流和激活量化融合属于性能优化。现有 <code>_gather_sp_input</code> 名称/开头注释偏 SP，实际已含 DP-owner 路径，学习时须按分支理解；它不表示 DP owner 是同一请求的 sequence slice。</p>
  </section>`;}
  const api={head,source,svg,render};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.KimiSharedTPCurrent=api;
})(typeof globalThis!=='undefined'?globalThis:this);
