(function(){
  'use strict';
  const M=window.KimiMemory,E=window.KIMI_MEMORY_EVIDENCE;
  const $=id=>document.getElementById(id), G=x=>(x/M.GiB).toFixed(6), B=x=>Math.round(x).toLocaleString('en-US');
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let p=M.defaults('a3',E),current=null,selectedLayer=0;
  const numeric=['context','output','globalConcurrency','maxSeqs','localTP','utilization','extraFullBlocks','extraStateBlocks','runtimeGiB','extraGiB','scaleBytesPerLayer'];
  function controls(){
    $('mem-controls').innerHTML=`<h2>计算输入</h2>
      <label>平台 / owner 拓扑<select id="platform"><option value="a3">A3 · DP64 / TP1 / EP64</option><option value="a5">A5 · DP32 / TP1 / EP32</option></select></label>
      <label>localTP 投影切分<select id="localTP">${[1,2,4,8,16,32].map(x=>`<option value="${x}">${x}</option>`).join('')}</select></label>
      <label>每请求已驻留上下文 tokens<input id="context" type="number" min="1" step="1"></label>
      <label>另加输出 tokens<input id="output" type="number" min="0" step="1"></label>
      <label>全局活跃请求数<input id="globalConcurrency" type="number" min="1" step="1"></label>
      <label>max_num_seqs / rank<input id="maxSeqs" type="number" min="1" step="1"></label>
      <label>gpu_memory_utilization<input id="utilization" type="number" min="0.01" max="1" step="0.01"></label>
      <label class="mem-check"><input id="align" type="checkbox">prefix cache · align（开=2个基础 state）</label>
      <label class="mem-check"><input id="c8" type="checkbox">DSpark C8 · 576 B/token</label>
      <label class="mem-check"><input id="recover" type="checkbox">RecoverSSM · PR21549</label>
      <label class="mem-check"><input id="shared4" type="checkbox">shared expert W4A8</label>
      <small>1M = 1,048,576 tokens；默认每个 DP rank 一条请求。localTP 只切投影；这里 TP=PCP=DCP=PP=1，KV / SSM 不除 localTP。</small>
      <details><summary>运行时 / 池预留</summary>
        <label>非权重、非缓存总残差 GiB<input id="runtimeGiB" type="number" min="0" step="0.01"></label>
        <small>默认为 A3 实测 HBM 上限 − 权重 − 物理缓存；覆盖通信、allocator、graph、其他驻留与设备侧开销。A5 使用这一个显式迁移假设，非 A5 实测。</small>
        <label>额外峰值 / 预留 GiB<input id="extraGiB" type="number" min="0" step="0.1"></label>
        <label>额外 retained history blocks / rank<input id="extraFullBlocks" type="number" min="0" step="1"></label>
        <label>额外 KDA state blocks / 每组<input id="extraStateBlocks" type="number" min="0" step="1"></label>
        <label>C8 独立 scale/量化参数 bytes / draft层<input id="scaleBytesPerLayer" type="number" min="0" step="1"></label>
      </details><button id="mem-reset">恢复该平台默认值</button>
      <div class="mem-side-links"><a href="#mem-ledger">物理账本</a><a href="#mem-layers">逐层排布</a><a href="#mem-scenarios">三项增量</a><a href="#mem-runtime">实测校准</a></div><div id="mem-error" class="mem-error" role="alert"></div>`;
    setControls();
    $('platform').addEventListener('change',()=>{p=M.defaults($('platform').value,E);setControls();draw();});
    numeric.forEach(k=>$(k).addEventListener('change',readControls));
    ['align','c8','recover','shared4'].forEach(k=>$(k).addEventListener('change',readControls));
    $('mem-reset').addEventListener('click',()=>{p=M.defaults(p.platform,E);setControls();draw();});
  }
  function setControls(){numeric.forEach(k=>$(k).value=p[k]);$('platform').value=p.platform;['align','c8','recover','shared4'].forEach(k=>$(k).checked=p[k]);}
  function readControls(){const next={...p};numeric.forEach(k=>next[k]=Number($(k).value));['align','c8','recover','shared4'].forEach(k=>next[k]=$(k).checked);try{M.calculate(next,E);p=next;draw();$('mem-error').textContent='';}catch(e){$('mem-error').textContent=e.message;}}
  const table=(heads,rows)=>`<div class="mem-table-wrap"><table class="mem-table"><thead><tr>${heads.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  const row=(name,bytes,formula,cls='')=>`<tr class="${cls}"><td>${name}</td><td class="num">${G(bytes)}</td><td><code>${formula}</code></td></tr>`;
  function ledger(r){
    const tokenBytes=M.ND*(M.MAIN+r.indexBytes)+M.DRAFT*r.draftBytes;
    const rows=[row('主模型 DSA · MLA 有效历史',r.contentMain,`${r.requests} × ${r.tokens} × 24 × 656`),
      row('主模型 DSA · Indexer 有效历史',r.contentIndexer,`${r.requests} × ${r.tokens} × 24 × ${r.indexBytes}`),
      row('DSpark 5 层有效历史',r.contentDraft,`${r.requests} × ${r.tokens} × 5 × ${r.draftBytes}`),
      row('KDA 必需 state 内容（含当前策略的所有副本）',r.contentState,`69 × ${r.statePerGroup} × 6,733,824`),
      row('3 token lookahead 容量',r.lookaheadBytes,`${r.requests} × 3 × ${tokenBytes}`),
      row('历史尾块空位（已排除 lookahead）',r.fullTail,`(请求数 × ${r.fullPerRequest} × 12288 − 有效 tokens − lookahead) × ${tokenBytes}`),
      row('KDA 每页 padding',r.statePadding,`69 × ${r.statePerGroup} × (8,060,928 − 6,733,824)`),
      row('KDA 组内第 24 个 MLA slot 空置',r.holes,`${r.stateBlocks} × (24 − 23) × 8,060,928`),
      row('KDA ID 下空置的 Indexer + DSpark sidecar',r.sidecars,`${r.stateBlocks} × (24×12288×${r.indexBytes} + 5×12288×${r.draftBytes})`),
      row('额外 retained history 池占位',r.retainedHistory,`${p.extraFullBlocks} × ${r.poolBytesPerID}`),
      row('全池共用 null block（必要保留，不是每层再加一块）',r.nullBytes,`1 × ${r.poolBytesPerID}`),
      row('物理缓存池合计',r.poolBytes,`${r.blocks} × ${r.poolBytesPerID}`,'total')];
    const segs=[['main',r.contentMain+r.contentIndexer,'DSA'],['draft',r.contentDraft,'DSpark'],['state',r.contentState,'KDA state'],['waste',r.waste,'排布浪费'],['null',r.nullBytes+r.lookaheadBytes+r.retainedHistory,'保留/预留']];
    return `<section id="mem-ledger" class="mem-section"><h2>01 / 物理缓存账本 · 每卡</h2>
      <p>这是能容纳当前请求的最小 physical pool；生产配置按预算预分配时可能更大。<b>${r.fullBlocks} 个 F 历史 ID + 3×${r.statePerGroup} 个 KDA ID + 1 个 null = ${r.blocks} 个 ID</b>。所有组从同一池拿不同 ID。以下“每ID成本”用于容量会计，同一ID的各层页分散在各自region内；真实地址stride见逐层表。</p>
      <div class="mem-code">每 pool ID 的总物理成本（不是连续地址 stride） = 24 × (12288 × 656) + 24 × (12288 × ${r.indexBytes}) + 5 × (12288 × ${r.draftBytes})
= ${B(r.poolBytesPerID)} bytes = ${(r.poolBytesPerID/M.MiB).toFixed(4)} MiB / pool ID
全局 ${r.dp} 个 rank 按最繁忙rank容量统一预分配：${G(r.clusterPoolBytes)} GiB 缓存；每条请求只驻留在其 owner，不能再乘 localTP。</div>
      <div class="mem-stacked">${segs.map(([cl,n,label])=>`<span class="${cl}" style="width:${n/r.poolBytes*100}%" title="${label}: ${G(n)} GiB"></span>`).join('')}</div>
      <div class="mem-legend">${segs.map(([cl,n,label])=>`<span><i style="background:${{main:'#739cd7',draft:'#a487c7',state:'#70ba98',waste:'#c99e66',null:'#69746e'}[cl]}"></i>${label} ${G(n)}</span>`).join('')}</div>
      ${table(['分项','GiB / rank','整数 bytes 计算式'],rows)}
      <p><b>真正的排布空耗 ${G(r.waste)} GiB</b>：尾块、state page padding、组内空 slot 与 sidecar；另外 ${G(r.nullBytes)} GiB 是必要 null 保留。KDA 的 ${r.stateCopies} 个副本不是这里的 padding：一个状态内容共 6.421875 MiB；全层单副本 ${G(r.canonicalState)} GiB，当前策略多副本另占 ${G(r.policyStateOverhead)} GiB。</p></section>`;
  }
  function layout(r){
    return `<section id="mem-layers" class="mem-section"><h2>02 / 98 个计算层，122 个 cache spec</h2>
      <p><b>主模型 93 层 = 69 KDA + 24 DSA</b>；24 个 DSA 各有 MLA 与 Indexer 两个 spec；draft 另有 5 个普通 MLA 层。因此是 98 个计算层、69+24+24+5=122 个 cache spec。以下 layer ID 全为 <b>0-based</b>；checkpoint 的配置列表用 1-based。</p>
      <div class="mem-code">KDA：0,1,2 / 4,5,6 / … / 88,89,90
DSA：3,7,11,…,91，另加第92层（最后连续两层 DSA）
DSpark：物理命名 93–97，draft 内部 0–4；不与 target feature layer 75/79/83/87/91 共用 KV
M1：0,4,…,88；M2：1,5,…,89；M3：2,6,…,90。每组23层，复用24个主MLA slot中的前23个。</div>
      <div class="mem-layer-strip">${r.layers.map(x=>`<button class="${x.type}${x.layer===selectedLayer?' chosen':''}" data-layer="${x.layer}" aria-label="查看第 ${x.layer} 层 ${x.type}" title="L${x.layer} · ${x.type} · ${x.group}">${x.layer}</button>`).join('')}</div>
      <div id="mem-layer-detail" class="mem-layer-detail"></div>
      <p>下表的 allocated 是请求及显式预留用到的 layer view；空 sidecar、空 slot、null 在上一节统一计费。逐层CSV另外列出lookahead与retained bytes，不混入padding。所有 view 最终来自<b>同一块 ${G(r.poolBytes)} GiB backing</b>；不能把每个 descriptor 的 <code>size=backing_size</code> 重复相加。</p>
      <details><summary>展开98层逐层账本 / CSV</summary>
      <div class="mem-toolbar"><select id="mem-filter" aria-label="筛选层类型"><option value="all">全部 98 层</option><option value="KDA">69 KDA</option><option value="DSA">24 DSA</option><option value="DSpark">5 DSpark</option></select><button id="mem-layer-csv">导出逐层 CSV</button></div>
      <div id="mem-layer-table"></div></details>
      <details><summary>展开 dtype 与每页物理 shape</summary>
        ${table(['类型','真实内容 / 层','页 / slot','排布规则'],[
          `<tr><td>DSA MLA</td><td>512×${p.platform==='a3'?'INT8':'FP8'} + 64×BF16 + 4×FP32 = 656 B/token</td><td>[12288,1,656] byte = 8,060,928 B</td><td>RoPE 已含在 656，不能另加 128 B</td></tr>`,
          `<tr><td>Indexer</td><td>128×${p.platform==='a3'?'INT8':'FP8'} + 1×${p.platform==='a3'?'FP16':'FP32'} = ${r.indexBytes} B/token</td><td>[12288,128] C8 + [12288,1] scale</td><td>A3 130 B；A5 132 B，不按 query head 数乘96</td></tr>`,
          '<tr><td>KDA recurrent</td><td>[96,128,128] FP32 = 6,291,456 B</td><td>与 conv 放同一 slot</td><td>96 头完整 owner state；localTP 不改 shape</td></tr>',
          '<tr><td>KDA conv</td><td>[6,36864] BF16 = 442,368 B</td><td>6 = kernel4−1 + speculative3</td><td>合计6,733,824 B，pad至8,060,928 B</td></tr>',
          `<tr><td>DSpark</td><td>[512+64] × ${p.c8?'C8':'BF16'} = ${r.draftBytes} B/token</td><td>[12288,1,576] × ${p.c8?1:2} B</td><td>5层各有独立region；C8 static scales另计</td></tr>`])}
      <h3>Manager 页与 kernel 页不是同一个 block_size</h3>
      <div class="mem-code">manager logical page = 12288 tokens；backend kernel block = 128 tokens
每 manager page = 96 个 kernel blocks；N=${r.blocks} 时，kernel cache第一维 = N×96 = ${r.blocks*96}
DSA packed view：[${r.blocks*96},128,1,656]；每token byte[0:512]为C8 latent，byte[512:640]为64个BF16 RoPE，byte[640:656]为4个FP32 scales（group128）
Indexer view：K [${r.blocks*96},128,1,128] C8；scale [${r.blocks*96},128,1] ${p.platform==='a3'?'FP16':'FP32'}
每个kernel slot先放128×128个key bytes，再放128个scale；并非把130/132 B逐token交错存放
DSpark view：latent [${r.blocks*96},128,1,512] + RoPE [${r.blocks*96},128,1,64]，均为${p.c8?'C8':'BF16'}，每个kernel slot两个组件连续；独立scale另存
KDA 每slot：conv前442,368 B + recurrent 6,291,456 B + padding 1,327,104 B</div>
      <p>A3原62块的profiling实际输入为主KV [5952,128,1,656]、Indexer [5952,128,1,128] 与FP16 scale [5952,128,1]，与62×96=5952一致。A5相同kernel block大小由固定版本backend源码给出；A5 bound tensor未实测。A5的packed容器可标成FP8，但RoPE/scale子区域保存的仍是BF16/FP32原始字节。</p>
      </details></section>`;
  }
  function drawLayers(){
    const r=current,x=r.layers.find(x=>x.layer===selectedLayer)||r.layers[0];
    $('mem-layer-detail').innerHTML=`<b>L${x.layer} · ${x.type} · group ${x.group} · slot ${x.slot}</b><br>
      ${esc(x.headShape)}<br>${esc(x.convShape)}<br>
      view offset = <code>${B(x.offsetBytes)} B</code>；layer stride = <code>${B(x.layerStride)} B</code>；block stride = <code>${B(x.blockStride)} B</code>
      ${x.indexOffset!==null?`<br>Indexer view offset = <code>${B(x.indexOffset)} B</code>；block stride = <code>${B(M.BLOCK*r.indexBytes)} B</code>`:''}<br>
      有效内容 ${G(x.contentBytes)} GiB；请求所用 view ${G(x.allocatedBytes)} GiB。${esc(x.layout)}<br>
      <span class="mem-tag">offset 是本计算输入下的 backing 相对偏移，非设备物理地址；具体 ID 分配由 scheduler 决定。</span>`;
    const filter=$('mem-filter').value;
    $('mem-layer-table').innerHTML=table(['layer / type','group / slot','dtype / 内容 shape','有效 GiB','所用 view GiB','页/块空位 GiB'],r.layers.filter(x=>filter==='all'||x.type===filter).map(x=>`<tr><td><button data-layer="${x.layer}">L${x.layer} · ${x.type}</button></td><td>${x.group} / ${x.slot}</td><td>${esc(x.headShape)}<br><span class="mem-tag">${esc(x.convShape)}</span></td><td class="num">${G(x.contentBytes)}</td><td class="num">${G(x.allocatedBytes)}</td><td class="num">${G(x.paddingBytes)}</td></tr>`));
  }
  function scenarios(r){
    const ss=M.scenarios(p,E),base=ss[0].result;
    return `<section id="mem-scenarios" class="mem-section"><h2>03 / 三项增量 · 每项独立与组合重算</h2>
      <p>各行相对同一基线，单位 GiB/卡。<b>“净减少”=物理池减少 + 权重减少 − 新增常驻</b>；新增常驻包含 C8 scale、RecoverSSM 记录和 commit metadata。没有把独立收益直接相加。</p>
      ${table(['配置','物理 KV 池','池减少','权重减少','新增常驻','净减少','含残差总预算'],ss.map(s=>`<tr class="${s.c8===p.c8&&s.recover===p.recover&&s.shared4===p.shared4?'selected':''}"><td>${s.name}</td><td class="num">${G(s.result.poolBytes)}</td><td class="num">${G(s.poolSaved)}</td><td class="num">${G(s.weightSaved)}</td><td class="num">${G(s.extra)}</td><td class="num">${G(s.saved)}</td><td class="num">${G(s.result.totalBytes)}</td></tr>`))}
      <h3>DSpark C8：1152 → 576 B/token，static scale 独立计</h3>
      <p>本需求选用普通 MLA 全量8bit，<b>不使用656 B的packed SFA格式</b>。在1M有效历史上，5层内容少 ${G(r.requests*r.tokens*5*576)} GiB；物理池还会少掉 KDA ID/null/尾块中对应的 draft 预留区。按BF16初始化FAKQuant，默认独立量化参数 ${p.scaleBytesPerLayer} B/层 ×5，包含现有 FAKQuant 的 FP32 512项 scale展开及 scale/offset/reciprocal 副本，可在左侧按最终实现覆盖。不是每token另加scale，也不是零开销。</p>
      <p>当前实测 draft 为 <code>auto/BF16</code>。现有普通 MLA 会在 <code>fa_quant_layer=False</code> 时回退模型 dtype；因此仅改 cache dtype 参数不能视为已完成 C8。这里给的是所选布局的容量测算，需配套量化参数与 kernel 验证。</p>
      <h3>RecoverSSM：省 state 副本，新增小型记录</h3>
      <div class="mem-code">当前 align：2 + 3 = 5 state pages / KDA层 / 请求
RecoverSSM align：2 + 0 = 2；none 模式为4→1
correction = [max_num_seqs,96,4,128] FP32 ×69 = ${B(r.rmem.correction)} B
key_gate   = [max_num_seqs,96,4,256] BF16 ×69 = ${B(r.rmem.keyGate)} B
commit地址/stride、A_log/dt_bias stacks、6个INT32请求数组 = ${B(r.rmem.metadata)} B
新增合计 = ${G(r.rmem.total)} GiB；记录按 max_num_seqs，不按 pool pages。</div>
      <p>69层FP32 state的3个额外副本被去掉，不是将 DSA 的1M KV变小。当前保留其他已测 runtime 残差，只扣明确可定位的state/池变化；没有虚构 state-copy workspace 的额外收益。<b>固定cache预算时总HBM可能不降，收益变为更多可用block</b>。</p>
      <div class="mem-note">PR21549 @ ${E.recoverHead.slice(0,10)} 当前拒绝 KV transfer。本 A3 实测基线带 P/D connector，故这是容量反事实，不能在该部署上直接打开；需要先有兼容实现或改为受支持的共置配置。</div>
      <h3>Shared expert：只改92层 shared，不动 routed、latent 或首层 dense</h3>
      <div class="mem-code">每层 gate/up [6144,7168]，down [7168,6144]；localTP=${p.localTP}
旧权重payload ${G(r.shared.oldPayload)} + metadata ${G(r.shared.oldMetadata)} GiB
新权重payload ${G(r.shared.newPayload)} + metadata ${G(r.shared.newMetadata)} GiB
净减少 ${G(r.shared.saved)} GiB</div>
      <p>${p.platform==='a3'?'A3 W8A8→W4A8：INT4打包减半；列投影输出6144/S，down输出7168仍复制。每输出 metadata 从8 B变为20 B：INT64编码scale8＋FP32 scale副本4＋offset4＋folded scale_bias4。因此净收益小于权重payload的一半。':'A5 MXFP8→MXFP4/W4A8：payload减半，group32的U8 scale不变。6144/S和7168均按现有NZ/scale布局校验；这是当前格式下的容量测算。'}</p>
      <p>组合中 C8 与 RecoverSSM 共省同一批 KDA ID 的 draft region；它们有交互项。以当前输入为例，不能将两项独立净收益相加作为组合值。</p></section>`;
  }
  function runtime(r){
    const m=E.measured,others=m.torch_allocated_after_graph_gib-m.weights_main_plus_draft_gib_measured-m.kv_pool_physical_gib;
    const rawPool=E.pool.physicalBytes;
    return `<section id="mem-runtime" class="mem-section"><h2>04 / 所有显存：加载、缓存与运行时</h2>
      ${table(['可加总主账','GiB / rank','口径'],[
        row('主模型＋DSpark加载权重',r.weightBytes,p.platform==='a3'?'A3 measured 37.700644 GiB 锚定；变更项逐项扣除':'实际checkpoint header＋loader/EP32/localTP8推导；非运行实测'),
        row('物理缓存池',r.poolBytes,'包含主模型、draft、SSM、所有布局空耗与null'),
        row('新增C8 / RecoverSSM常驻',r.addedRuntime,'独立scale、records、commit metadata；不在原残差内'),
        row('非权重、非缓存总残差',p.runtimeGiB*M.GiB,p.platform==='a3'?'A3 sampled HBM max − measured weights − physical pool':'显式迁移A3残差作参考；必须用A5 localTP实测替换'),
        row('用户额外峰值预留',p.extraGiB*M.GiB,'默认0是规划输入；1M prefill / 更多capture / MM等新增开销需另填'),
        row('当前 decode 参考总预算',r.totalBytes,'以上五项之和','total')])}
      <p><b>${p.platform.toUpperCase()} ${r.physicalCapacityGiB} GiB physical HBM</b>；当前参考总预算 ${G(r.totalBytes)} GiB，差额 ${(r.physicalCapacityGiB-r.totalBytes/M.GiB).toFixed(6)} GiB。权重＋物理池＋新增记录的已核算部分为 ${G(r.knownBytes)} GiB。${p.platform==='a5'?'<b>A5 总预算含明确的跨硬件残差假设，不能据此宣布部署通过。</b>':'A3 是用128K decode样本迁移到1M的预算，不是1M真实运行峰值。'}</p>
      <div class="mem-note">框架可见 ${r.frameworkCapacityGiB.toFixed(6)} GiB × utilization ${p.utilization} = <b>${G(r.requestedBudgetBytes)} GiB</b> 的请求预算。${r.knownBytes>r.requestedBudgetBytes?`仅已核算权重＋缓存＋新增记录就超预算 ${G(r.knownBytes-r.requestedBudgetBytes)} GiB，<b>不能满足该 utilization</b>。`:'已核算部分在请求预算内，仍需为non-Torch、激活、graph和分配器开销留足空间。'}physical容量有余量不等于框架预算通过；物理残差也不能未经转换再从框架预算重复扣一次。</div>
      <div class="mem-note">本页算的是1M已驻留后的decode。实测跳过MM profiling、使用模拟KV；没有实际1M prefill峰值、精度或prefix命中率验证。全模型权重已含ViT/MM参数；未测的图像激活、1M prefill工作区、不同并发/graph容量变化不能宣称为0，需用上面的额外峰值输入预留并实测。</div>
      <h3>A3 实测校准：分阶段读数，以下不再加到主账</h3>
      ${table(['实测指标','GiB','关系 / 时点'],[
        ['加载权重',m.weights_main_plus_draft_gib_measured,'已在 Torch allocated 内'],
        ['物理 KV backing',rawPool/M.GiB,'62 × 302,579,712 B；同一 arena，只计一次'],
        ['profile KV 可分预算',m.kv_pool_budget_gib,'预算，不是allocation；小于一整个pool ID的余量不分配'],
        ['激活峰值',m.activation_peak_gib_measured,'profile阶段 peak，非长期独立常驻'],
        ['graph前 non-Torch增加',m.non_torch_increase_before_graph_gib,'含通信等；不能与全部HCCL再加一次'],
        ['HCCL allocation',m.hccl_buffer_allocation_gib,'已包含于non-Torch / graph会计'],
        ['capture物理free delta',m.graph_capture_physical_free_memory_delta_gib,'跨capture测量，部分已在Torch内'],
        ['graph后 Torch allocated',m.torch_allocated_after_graph_gib,'含权重＋KV＋其他Torch张量'],
        ['其中其他Torch',others,'allocated − weights − KV，仅该时点'],
        ['graph后 reserved unused',m.torch_reserved_unused_after_graph_gib,'reserved − allocated'],
        ['graph后 Torch reserved',m.torch_reserved_after_graph_gib,'包含 allocated，不能叠加'],
        ['benchmark物理HBM最小',E.physicalSnapshot.minGiB,'64设备运行采样'],
        ['benchmark物理HBM最大',E.physicalSnapshot.maxGiB,'采样上限，不是instrumented peak']
      ].map(([name,n,note])=>`<tr><td>${name}</td><td class="num">${n.toFixed(6)}</td><td>${note}</td></tr>`))}
      <p>复算62块：${G(rawPool)} GiB，精确匹配账本。当前1M配置需要 ${r.blocks} 块；旧62块没有实际承载1M。profile返回预算减已分配池 ${ (m.kv_pool_budget_gib-m.kv_pool_physical_gib).toFixed(6)} GiB 是budget rounding余量，不是又一份allocated。</p>
      <details><summary>展开当前权重组成与加载校准</summary>
      ${table(['主模型模块','GiB / rank'],r.weights.rows.map(x=>`<tr><td>${esc(x.name)}</td><td class="num">${G(x.bytes)}</td></tr>`).concat([
        `<tr><td>主模型估算合计</td><td class="num">${G(r.weights.main)}</td></tr>`,
        `<tr><td>DSpark preview独立权重（不重复共享target权重）</td><td class="num">${G(r.weights.draft)}</td></tr>`,
        `<tr><td>A3加载测量−header/loader估算校准</td><td class="num">${G(r.weights.adjustment)}</td></tr>`]))}
      <p>Routed experts按EP分片（896专家，A3每rank14、A5每rank28）；latent/router等仍复制。QKV/G/O/vocab/dense/shared按各自localTP规则；取消full-F。A3测量与header估算的余项保留为加载校准；A5没有该拓扑的对应校准。开启shared W4A8后，在主账扣除上节的净节省，展开表仍显示基线。</p></details></section>`;
  }
  function source(){return `<section class="mem-section"><h2>05 / 来源与复现</h2><ul class="mem-sources">
    <li>A3：2026-11-12 DP64/TP1/EP64/localTP16、seqs1、HCCL256/local64、DSpark3；head <code>${E.sourceHead}</code>。公开版仅含去标识后的账本、cache spec、权重分类汇总，<a href="https://github.com/Dawn952/kimi-k31-a5-analysis/blob/main/data/memory-evidence.json">查看精简证据</a>。</li>
    <li>localTP与物理布局：<a href="https://github.com/vllm-project/vllm-ascend/pull/21624">PR21624</a>；<a href="https://github.com/Dawn952/vllm-ascend/blob/${E.sourceHead}/vllm_ascend/models/kimi_k3/paired_sfa_cache.py">固定版本paired_sfa_cache.py</a>：同一arena、main/indexer/draft分区，KDA只overlay main区。</li>
    <li>RecoverSSM：<a href="https://github.com/vllm-project/vllm-ascend/pull/21549">PR21549</a> @ <code>${E.recoverHead}</code>；<a href="https://github.com/vllm-project/vllm-ascend/blob/${E.recoverHead}/vllm_ascend/ops/triton/kda/recoverssm.py">commit metadata与records调用</a>。上游vLLM @ <code>${E.upstreamHead}</code> 的 <a href="https://github.com/vllm-project/vllm/blob/${E.upstreamHead}/vllm/models/kimi_k3/nvidia/ops/recoverssm.py">KDARecoverSSMRecords</a> 给出确切shape/dtype。</li>
    <li>W4A8：<a href="https://github.com/Dawn952/vllm-ascend/blob/${E.sourceHead}/vllm_ascend/quantization/methods/w4a8/w4a8.py">A3 INT4 linear</a>、<a href="https://github.com/Dawn952/vllm-ascend/blob/${E.sourceHead}/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py">A5 MXFP4 linear</a>；C8独立参数参照同版本 <a href="https://github.com/Dawn952/vllm-ascend/blob/${E.sourceHead}/vllm_ascend/quantization/methods/kv_cache/kv_c8.py">FAKQuant</a>。场景是容量规划，不代表该checkpoint已转量化或完成精度验收。</li>
    </ul><div class="mem-toolbar"><button id="mem-json">导出本次完整计算 JSON</button><button id="mem-scenario-csv">导出场景 CSV</button><a href="https://github.com/Dawn952/kimi-k31-a5-analysis/blob/main/src/memory-model.js">审阅全部公式</a></div></section>`;}
  function draw(){
    const r=M.calculate(p,E);current=r;
    $('mem-results').innerHTML=`<div class="mem-kpis">
      <div class="mem-kpi"><span>${p.platform.toUpperCase()} · 1 rank最小物理缓存池</span><strong>${G(r.poolBytes)} <small>GiB</small></strong><small>${r.blocks} pool IDs；每rank ${r.requests} 请求</small></div>
      <div class="mem-kpi"><span>排布空耗（不含必要null）</span><strong>${G(r.waste)} <small>GiB</small></strong><small>另有 null ${G(r.nullBytes)} GiB</small></div>
      <div class="mem-kpi"><span>含所有实测残差的decode参考预算</span><strong>${G(r.totalBytes)} <small>GiB</small></strong><small>${p.platform==='a5'?'A3残差迁移假设；A5非实测':'A3 128K测量外推，非1M实测峰值'}</small></div></div>
      ${p.platform==='a5'?'<div class="mem-note">A5：DP32/TP1/localTP8是源码与checkpoint估算。缓存分组按PR21624同型paired布局；运行时残差默认迁移A3上限，可修改，不能作为A5峰值验收。</div>':''}
      ${ledger(r)}${layout(r)}${scenarios(r)}${runtime(r)}${source()}`;
    drawLayers();$('mem-filter').addEventListener('change',drawLayers);
    $('mem-layer-csv').addEventListener('click',()=>csv('layers',r.layers));
    $('mem-scenario-csv').addEventListener('click',()=>csv('scenarios',M.scenarios(p,E).map(s=>({name:s.name,poolGiB:s.result.poolBytes/M.GiB,poolSavedGiB:s.poolSaved/M.GiB,weightSavedGiB:s.weightSaved/M.GiB,newResidentGiB:s.extra/M.GiB,netSavedGiB:s.saved/M.GiB,totalWithRuntimeAssumptionGiB:s.result.totalBytes/M.GiB,runtimeAssumptionGiB:p.runtimeGiB,context:p.context,platform:p.platform}))));
    $('mem-json').addEventListener('click',()=>download('kimi-1m-'+p.platform+'.json',JSON.stringify({evidence:E.sourceHead,recover:E.recoverHead,result:r,scenarios:M.scenarios(p,E),scope:E.scope},null,2),'application/json'));
  }
  function csv(name,rows){const heads=Object.keys(rows[0]),quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';download(`kimi-1m-${p.platform}-${name}.csv`,'\uFEFF'+[heads.map(quote).join(','),...rows.map(x=>heads.map(k=>quote(x[k])).join(','))].join('\r\n'),'text/csv;charset=utf-8');}
  function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
  controls();
  $('mem-results').addEventListener('click',event=>{
    const el=event.target.closest('[data-layer]');if(!el)return;
    selectedLayer=Number(el.dataset.layer);drawLayers();
    document.querySelectorAll('.mem-layer-strip [data-layer]').forEach(b=>b.classList.toggle('chosen',Number(b.dataset.layer)===selectedLayer));
  });
  draw();
})();
