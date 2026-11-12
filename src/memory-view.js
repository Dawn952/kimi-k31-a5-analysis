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
  const trace=text=>`<div class="mem-formula-note">${esc(text).replaceAll('\n','<br>')}</div>`;
  const row=(name,bytes,formula,cls='',explanation='')=>`<tr class="${cls}"><td>${name}</td><td class="num">${G(bytes)}</td><td><code>${esc(formula)}</code>${explanation?trace(explanation):''}</td></tr>`;
  function ledger(r){
    const tokenBytes=M.ND*(M.MAIN+r.indexBytes)+M.DRAFT*r.draftBytes;
    const owners=`${r.requests} 请求/rank = ceil(${p.globalConcurrency} 全局请求 ÷ ${r.dp} 个DP rank)，按最繁忙rank核算。`;
    const context=`${B(r.tokens)} token/请求 = ${B(p.context)} 已驻留上下文 + ${B(p.output)} 另加输出。`;
    const mainBytes='656 B/(token·层) = 512 个latent元素×1 B/C8 + 64 个RoPE元素×2 B/BF16 + 4 个scale×4 B/FP32。';
    const indexBytes=`${r.indexBytes} B/(token·层) = 128 个key元素×1 B/C8 + 1 个scale×${p.platform==='a3'?2:4} B/${p.platform==='a3'?'FP16':'FP32'}。`;
    const draftBytes=`${r.draftBytes} B/(token·层) = (512 个latent元素 + 64 个RoPE元素) × ${p.c8?1:2} B/${p.c8?'C8':'BF16'}。${p.c8?'独立static scale在新增常驻另计。':''}`;
    const stateBytes='6,733,824 B/state页 = 96 头×128 个value维度×128 个key维度×4 B/FP32 + 6 个历史位置×36,864 个conv通道×2 B/BF16。';
    const pageBytes='8,060,928 B/物理页 = 12,288 token/manager页 × 656 B/(token·DSA层)。KDA复用同样大小的物理页。';
    const fullBytes=`${B(tokenBytes)} B/token = 24 个DSA层×(656 B主KV + ${r.indexBytes} B索引) + 5 个draft层×${r.draftBytes} B。`;
    const stateCount=`每组 ${r.statePerGroup} 页 = ${r.requests} 请求×${r.stateCopies} state页/请求 + ${p.extraStateBlocks} 额外预留页；69个KDA层均各占这些页。`;
    const rows=[row('主模型 DSA · MLA 有效历史',r.contentMain,`${r.requests} 请求/rank × ${B(r.tokens)} token/请求 × 24 层 × 656 B/(token·层)`,'',`${owners}\n${context}\n24 是主模型DSA层数（每层有独立KV），不是TP卡数。${mainBytes}`),
      row('主模型 DSA · Indexer 有效历史',r.contentIndexer,`${r.requests} 请求/rank × ${B(r.tokens)} token/请求 × 24 层 × ${r.indexBytes} B/(token·层)`,'',`${owners}\n${context}\n24个DSA层各自保存Indexer cache。${indexBytes}`),
      row('DSpark 5 层有效历史',r.contentDraft,`${r.requests} 请求/rank × ${B(r.tokens)} token/请求 × 5 层 × ${r.draftBytes} B/(token·层)`,'',`${owners}\n${context}\n5 是draft模型层数，与3个推测token无关。${draftBytes}`),
      row('KDA 必需 state 内容（含当前策略的所有副本）',r.contentState,`69 层 × ${r.statePerGroup} state页/层 × 6,733,824 B/state页`,'',`${stateCount}\n${stateBytes}\n6 = conv kernel长度4 − 1 + 3个推测token；36,864 = Q/K/V三路×96头×128维。`),
      row('3 token lookahead 容量',r.lookaheadBytes,`${r.requests} 请求/rank × 3 token/请求 × ${B(tokenBytes)} B/token`,'',`3 是预留的推测token数，不是draft层数。${fullBytes}`),
      row('历史尾块空位（已排除 lookahead）',r.fullTail,`(${r.requests} 请求 × ${r.fullPerRequest} 页/请求 × 12,288 token/页 − ${B(r.requests*r.tokens)} 有效token − ${r.requests*3} lookahead token) × ${B(tokenBytes)} B/token`,'',`${r.fullPerRequest} 页/请求 = ceil((${B(r.tokens)} 历史token + 3 lookahead token) ÷ 12,288 token/页)。按整页分配后剩余的位置才是尾部空位。${fullBytes}`),
      row('KDA 每页 padding',r.statePadding,`69 层 × ${r.statePerGroup} state页/层 × (8,060,928 − 6,733,824) B/页`,'',`${pageBytes}\n${stateBytes}\n每页padding = 8,060,928 − 6,733,824 = 1,327,104 B；这部分没有有效state。`),
      row('KDA 组内第 24 个 MLA slot 空置',r.holes,`${r.stateBlocks} 个KDA池ID × (24 − 23) 个空slot/ID × 8,060,928 B/slot`,'',`${r.stateBlocks} 个ID = 3个KDA组×${r.statePerGroup} 页/组。69个KDA层拆成3组，每组23层；主MLA区固定24个slot，因此每个KDA ID闲置1个slot。\n${pageBytes}`),
      row('KDA ID 下空置的 Indexer + DSpark sidecar',r.sidecars,`${r.stateBlocks} 个KDA池ID × (24 层×12,288 token/页×${r.indexBytes} B/(token·层) + 5 层×12,288 token/页×${r.draftBytes} B/(token·层))`,'',`每个KDA ID只写main区里的state，但池仍给这个ID预留24份Indexer页和5份draft页。\nIndexer空区 ${B(r.indexRegion)} B/ID；draft空区 ${B(r.draftRegion)} B/ID。${indexBytes} ${draftBytes}`),
      row('额外 retained history 池占位',r.retainedHistory,`${p.extraFullBlocks} 额外历史池ID × ${B(r.poolBytesPerID)} B/ID`,'','额外ID数量由左侧输入；每ID字节数由本节上方main、Indexer、draft三个region相加得出，不是单个tensor的stride。'),
      row('全池共用 null block（必要保留，不是每层再加一块）',r.nullBytes,`1 个null池ID × ${B(r.poolBytesPerID)} B/ID`,'','1 是全池共享的null ID数量；它仍占一整份main＋Indexer＋draft容量。每ID成本的来源见本节上方分区计算。'),
      row('物理缓存池合计',r.poolBytes,`${r.blocks} 个池ID × ${B(r.poolBytesPerID)} B/ID`,'total',`${r.blocks} = ${r.fullBlocks} 历史ID + ${r.stateBlocks} KDA ID + 1 null ID；${r.fullBlocks} = ${r.requests} 请求×${r.fullPerRequest} 历史页/请求 + ${p.extraFullBlocks} 额外历史页。`)];
    const segs=[['main',r.contentMain+r.contentIndexer,'DSA'],['draft',r.contentDraft,'DSpark'],['state',r.contentState,'KDA state'],['waste',r.waste,'排布浪费'],['null',r.nullBytes+r.lookaheadBytes+r.retainedHistory,'保留/预留']];
    return `<section id="mem-ledger" class="mem-section"><h2>01 / 物理缓存账本 · 每卡</h2>
      <p>这是能容纳当前请求的最小 physical pool；生产配置按预算预分配时可能更大。<b>${r.fullBlocks} 个 F 历史 ID + 3×${r.statePerGroup} 个 KDA ID + 1 个 null = ${r.blocks} 个 ID</b>。所有组从同一池拿不同 ID。以下“每ID成本”用于容量会计，同一ID的各层页分散在各自region内；真实地址stride见逐层表。</p>
      <div class="mem-code">每 pool ID 的总物理成本（不是连续地址 stride）：
主MLA区 = 24 个DSA层 × 12,288 token/页 × 656 B/(token·层) = ${B(r.mainRegion)} B/ID
Indexer区 = 24 个DSA层 × 12,288 token/页 × ${r.indexBytes} B/(token·层) = ${B(r.indexRegion)} B/ID
DSpark区 = 5 个draft层 × 12,288 token/页 × ${r.draftBytes} B/(token·层) = ${B(r.draftRegion)} B/ID
相加 = ${B(r.mainRegion)} + ${B(r.indexRegion)} + ${B(r.draftRegion)} = ${B(r.poolBytesPerID)} B/ID
单位换算：1 MiB = 1024×1024 B；所以每ID成本 = ${(r.poolBytesPerID/M.MiB).toFixed(4)} MiB
全局 ${r.dp} 个 rank 按最繁忙rank容量统一预分配：${G(r.clusterPoolBytes)} GiB 缓存；每条请求只驻留在其 owner，不能再乘 localTP。</div>
      ${trace(`12,288 是每个manager页的token容量（不是每请求长度）；656 = 512×1 + 64×2 + 4×4 B；Indexer ${r.indexBytes} = 128×1 + 1×${p.platform==='a3'?2:4} B；draft ${r.draftBytes} = (512+64)×${p.c8?1:2} B。下面每行继续解释各个乘数。`)}
      <div class="mem-stacked">${segs.map(([cl,n,label])=>`<span class="${cl}" style="width:${n/r.poolBytes*100}%" title="${label}: ${G(n)} GiB"></span>`).join('')}</div>
      <div class="mem-legend">${segs.map(([cl,n,label])=>`<span><i style="background:${{main:'#739cd7',draft:'#a487c7',state:'#70ba98',waste:'#c99e66',null:'#69746e'}[cl]}"></i>${label} ${G(n)}</span>`).join('')}</div>
      <p>以下计算式的结果都是 <b>bytes（B）</b>；再除以 <b>1,073,741,824 B/GiB = 1024×1024×1024 B/GiB</b> 得到第二列。请求、token、层和字节宽度分别标注，不能把数值相同的不同单位混用。</p>
      ${table(['分项','GiB / rank','计算式 · 每项含义与字节数来源'],rows)}
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
          '<tr><td>KDA recurrent</td><td>96头 × 128个value维度 × 128个key维度 × 4 B/FP32 = 6,291,456 B</td><td>与 conv 放同一 slot</td><td>96头完整owner state；localTP不改shape</td></tr>',
          '<tr><td>KDA conv</td><td>6个历史位置 × 36,864通道 × 2 B/BF16 = 442,368 B</td><td>6 = kernel长度4−1 + 推测token数3；36,864 = Q/K/V三路×96头×128维</td><td>6,291,456 + 442,368 = 6,733,824 B有效内容；物理页12,288×656 = 8,060,928 B</td></tr>',
          `<tr><td>DSpark</td><td>[512+64] × ${p.c8?'C8':'BF16'} = ${r.draftBytes} B/token</td><td>[12288,1,576] × ${p.c8?1:2} B</td><td>5层各有独立region；C8 static scales另计</td></tr>`])}
      <h3>Manager 页与 kernel 页不是同一个 block_size</h3>
      <div class="mem-code">manager logical page = 12288 tokens；backend kernel block = 128 tokens
每 manager page = 12,288 token ÷ 128 token/kernel块 = 96 个 kernel blocks（这里96是块数，不是96个attention头）
N=${r.blocks} 是pool ID数；kernel cache第一维 = ${r.blocks} 个manager页×96 个kernel块/页 = ${r.blocks*96} 个kernel块
DSA packed view：[${r.blocks*96},128,1,656]；每token byte[0:512]为C8 latent，byte[512:640]为64个BF16 RoPE，byte[640:656]为4个FP32 scales（group128）
Indexer view：K [${r.blocks*96},128,1,128] C8；scale [${r.blocks*96},128,1] ${p.platform==='a3'?'FP16':'FP32'}
每个kernel slot先放128×128个key bytes，再放128个scale；并非把130/132 B逐token交错存放
DSpark view：latent [${r.blocks*96},128,1,512] + RoPE [${r.blocks*96},128,1,64]，均为${p.c8?'C8':'BF16'}，每个kernel slot两个组件连续；独立scale另存
KDA 每slot有效内容 = conv 442,368 B + recurrent 6,291,456 B = 6,733,824 B
KDA 每slot padding = 物理页8,060,928 B − 有效内容6,733,824 B = 1,327,104 B</div>
      ${trace('Shape中的轴：第1维是kernel块数，第2维128是每kernel块的token数，第3维1是一个压缩KV槽（不是96个query head），最后一维是每token的feature数或packed字节宽度。512是latent维度，64是RoPE维度；656 B中的4个FP32 scale来自512 latent维÷128元素/量化组。1 B/C8、2 B/BF16或FP16、4 B/FP32。')}
      <p>A3原62块的profiling实际输入为主KV [5952,128,1,656]、Indexer [5952,128,1,128] 与FP16 scale [5952,128,1]，与62×96=5952一致。A5相同kernel block大小由固定版本backend源码给出；A5 bound tensor未实测。A5的packed容器可标成FP8，但RoPE/scale子区域保存的仍是BF16/FP32原始字节。</p>
      </details></section>`;
  }
  function drawLayers(){
    const r=current,x=r.layers.find(x=>x.layer===selectedLayer)||r.layers[0];
    const indexPage=M.BLOCK*r.indexBytes,draftPage=M.BLOCK*r.draftBytes;
    const address=x.type==='DSpark'
      ?`view offset = ${r.blocks} 池ID×(${B(r.mainRegion)} B主MLA区/ID + ${B(r.indexRegion)} B索引区/ID) + ${x.slot} draft槽号×${r.blocks} 池ID×${B(draftPage)} B/draft页 = ${B(x.offsetBytes)} B\nlayer stride = ${r.blocks} 池ID×${B(draftPage)} B/draft页 = ${B(x.layerStride)} B\nblock stride = 12,288 token/页×${r.draftBytes} B/token = ${B(x.blockStride)} B`
      :`view offset = ${x.slot} main槽号×${r.blocks} 池ID×8,060,928 B/main页 = ${B(x.offsetBytes)} B\nlayer stride = ${r.blocks} 池ID×8,060,928 B/main页 = ${B(x.layerStride)} B\nblock stride = 12,288 token/页×656 B/token = 8,060,928 B`;
    const shape=x.type==='KDA'
      ?'State的96是head数，两个128分别是value和key维度，每元素FP32占4 B；conv的6是历史位置数（kernel4−1+推测3），36,864是Q/K/V三路×96头×128维，每元素BF16占2 B。'
      :x.type==='DSA'
        ?`512是latent维度（C8每元素1 B），64是RoPE维度（BF16每元素2 B），4是latent按128维分组的scale数（FP32每个4 B），相加656 B/token。Indexer另有128个C8 key元素和1个${p.platform==='a3'?'FP16（2 B）':'FP32（4 B）'} scale。`
        :`512是latent维度、64是RoPE维度，合计576个元素/token；当前每元素${p.c8?'C8占1 B，所以576 B/token':'BF16占2 B，所以1152 B/token'}。`;
    $('mem-layer-detail').innerHTML=`<b>L${x.layer} · ${x.type} · group ${x.group} · slot ${x.slot}</b><br>
      ${esc(x.headShape)}<br>${esc(x.convShape)}<br>
      ${trace(shape)}
      ${trace(address)}
      ${x.indexOffset!==null?trace(`Indexer view offset = 24 个main槽×${r.blocks} 池ID×8,060,928 B/main页 + ${x.slot} index槽号×${r.blocks} 池ID×${B(indexPage)} B/index页 = ${B(x.indexOffset)} B\nIndexer block stride = 12,288 token/页×${r.indexBytes} B/token = ${B(indexPage)} B`):''}
      ${trace('槽号slot从0开始，是该region内的层顺序；offset是从backing起点跳过多少字节，layer stride是相邻层起点的字节间隔，block stride是同层相邻manager页起点的字节间隔。main、Indexer、draft三个region的大小由本页第一节的乘式推导。')}
      有效内容 ${G(x.contentBytes)} GiB；请求所用 view ${G(x.allocatedBytes)} GiB。${esc(x.layout)}<br>
      <span class="mem-tag">offset 是本计算输入下的 backing 相对偏移，非设备物理地址；具体 ID 分配由 scheduler 决定。</span>`;
    const filter=$('mem-filter').value;
    $('mem-layer-table').innerHTML=table(['layer / type','group / slot','dtype / 内容 shape','有效 GiB','所用 view GiB','页/块空位 GiB'],r.layers.filter(x=>filter==='all'||x.type===filter).map(x=>`<tr><td><button data-layer="${x.layer}">L${x.layer} · ${x.type}</button></td><td>${x.group} / ${x.slot}</td><td>${esc(x.headShape)}<br><span class="mem-tag">${esc(x.convShape)}</span></td><td class="num">${G(x.contentBytes)}</td><td class="num">${G(x.allocatedBytes)}</td><td class="num">${G(x.paddingBytes)}</td></tr>`));
  }
  function scenarios(r){
    const ss=M.scenarios(p,E),base=ss[0].result;
    const defaultScale=p.platform==='a3'?2350:2352,reciprocalBytes=p.platform==='a3'?2:4;
    const sharedOutputs=92*(2*M.I/p.localTP+M.H),sharedElements=92*3*M.H*M.I/p.localTP;
    return `<section id="mem-scenarios" class="mem-section"><h2>03 / 三项增量 · 每项独立与组合重算</h2>
      <p>各行相对同一基线，单位 GiB/卡。<b>“净减少”=物理池减少 + 权重减少 − 新增常驻</b>；新增常驻包含 C8 scale、RecoverSSM 记录和 commit metadata。没有把独立收益直接相加。</p>
      ${table(['配置','物理 KV 池','池减少','权重减少','新增常驻','净减少','含残差总预算'],ss.map(s=>`<tr class="${s.c8===p.c8&&s.recover===p.recover&&s.shared4===p.shared4?'selected':''}"><td>${s.name}</td><td class="num">${G(s.result.poolBytes)}</td><td class="num">${G(s.poolSaved)}</td><td class="num">${G(s.weightSaved)}</td><td class="num">${G(s.extra)}</td><td class="num">${G(s.saved)}</td><td class="num">${G(s.result.totalBytes)}</td></tr>`))}
      <h3>DSpark C8：1152 → 576 B/token，static scale 独立计</h3>
      <p>本需求选用普通 MLA 全量8bit，<b>不使用656 B的packed SFA格式</b>。在1M有效历史上，5层内容少 ${G(r.requests*r.tokens*5*576)} GiB；物理池还会少掉 KDA ID/null/尾块中对应的 draft 预留区。当前独立量化参数 ${p.scaleBytesPerLayer} B/层 ×5，按BF16初始化FAKQuant的默认推导如下，包含现有 FAKQuant 的 FP32 512项 scale展开及 scale/offset/reciprocal 副本，可在左侧按最终实现覆盖。不是每token另加scale，也不是零开销。</p>
      <div class="mem-code">1152 B/token = (512 latent维 + 64 RoPE维) × 2 B/BF16
576 B/token = (512 latent维 + 64 RoPE维) × 1 B/C8
每token减少 = 1152 − 576 = 576 B（数值恰好也为576，但这里表示减少量）
默认独立量化参数 ${defaultScale} B/层的来源：
(96个query通道 + 1个key通道 + 1个value通道) × (2 B/scale + 1 B/INT8 offset) = 294 B
+ 1个FP32 key scale副本×4 B = 4 B
+ 1个scale倒数×${reciprocalBytes} B = ${reciprocalBytes} B
+ 1个BF16 offset副本×2 B = 2 B
+ 512个展开的latent量化scale×4 B/FP32 = 2048 B
相加：294 + 4 + ${reciprocalBytes} + 2 + 2048 = ${defaultScale} B/层
当前采用 ${p.scaleBytesPerLayer} B/层 × 5个draft层 = ${B(5*p.scaleBytesPerLayer)} B ${p.scaleBytesPerLayer===defaultScale?'（默认FAKQuant规划）':'（用户已覆盖默认值）'}</div>
      <p>当前实测 draft 为 <code>auto/BF16</code>。现有普通 MLA 会在 <code>fa_quant_layer=False</code> 时回退模型 dtype；因此仅改 cache dtype 参数不能视为已完成 C8。这里给的是所选布局的容量测算，需配套量化参数与 kernel 验证。</p>
      <h3>RecoverSSM：省 state 副本，新增小型记录</h3>
      <div class="mem-code">align基础2页 = 1个当前state + 1个对齐checkpoint；原来另有3个推测state页
所以 native align为2 + 3 = 5页/层/请求；RecoverSSM为2 + 0 = 2页；none模式不留对齐checkpoint，为4→1页
correction = 69个KDA层 × ${p.maxSeqs} 请求槽(max_num_seqs) × 96头 × 4验证token × 128个value维度 × 4 B/FP32
= ${B(r.rmem.correction)} B；4验证token = 1个当前token + 3个推测token
key_gate = 69层 × ${p.maxSeqs} 请求槽 × 96头 × 4验证token × 256个key/gate元素 × 2 B/BF16
= ${B(r.rmem.keyGate)} B；256 = 128个key维度 + 128个gate维度
commit metadata分别为：
10个地址/stride数组 × 69层 × 8 B/INT64 = ${B(10*69*8)} B
A_log堆叠：69层 × 96头 × 4 B/FP32 = ${B(69*96*4)} B
dt_bias堆叠：69层 × 96头 × 128维 × 4 B/FP32 = ${B(69*96*128*4)} B
6个请求数组 × ${p.maxSeqs} 请求槽 × 4 B/INT32 = ${B(6*p.maxSeqs*4)} B
metadata合计 = ${B(10*69*8)} + ${B(69*96*4)} + ${B(69*96*128*4)} + ${B(6*p.maxSeqs*4)} = ${B(r.rmem.metadata)} B
新增总量 = correction ${B(r.rmem.correction)} + key_gate ${B(r.rmem.keyGate)} + metadata ${B(r.rmem.metadata)}
= ${B(r.rmem.total)} B ÷ 1024³ B/GiB = ${G(r.rmem.total)} GiB</div>
      ${trace('10个INT64数组 = conv的base/block/dim/token四种地址或步长 + state的base/block两种 + correction的base/block两种 + key_gate的base/block两种。6个INT32请求数组是commit_lens、state_commit_lens、boundary_commit_lens、final_state_indices、boundary_state_indices、boundary_recovery_lens。上述记录按max_num_seqs请求槽预分配，不按池页数，也不只按当前活跃请求数。')}
      <p>69层FP32 state的3个额外副本被去掉，不是将 DSA 的1M KV变小。当前保留其他已测 runtime 残差，只扣明确可定位的state/池变化；没有虚构 state-copy workspace 的额外收益。<b>固定cache预算时总HBM可能不降，收益变为更多可用block</b>。</p>
      <div class="mem-note">PR21549 @ ${E.recoverHead.slice(0,10)} 当前拒绝 KV transfer。本 A3 实测基线带 P/D connector，故这是容量反事实，不能在该部署上直接打开；需要先有兼容实现或改为受支持的共置配置。</div>
      <h3>Shared expert：只改92层 shared，不动 routed、latent 或首层 dense</h3>
      <div class="mem-code">每层 gate/up [6144,7168]，down [7168,6144]：7168是模型hidden维度，6144是shared expert intermediate维度
92个shared层 = 主模型93层 − 第0层dense MLP；每个shared层有gate、up、down共3个矩阵
每卡权重元素 = 92层 × 3矩阵/层 × 7168 × 6144 ÷ ${p.localTP} 个localTP分片 = ${B(sharedElements)} 元素
旧payload = ${B(sharedElements)} 元素 × 1 B/W8 = ${B(r.shared.oldPayload)} B = ${G(r.shared.oldPayload)} GiB
新payload = ${B(sharedElements)} 元素 × 0.5 B/W4（2个4bit权重/byte） = ${B(r.shared.newPayload)} B = ${G(r.shared.newPayload)} GiB
${p.platform==='a3'?`metadata通道数 = 92层 × (2个列投影×6144/${p.localTP} 个本地输出 + 7168个down输出) = ${B(sharedOutputs)} 通道
旧metadata = ${B(sharedOutputs)} 通道 × 8 B/通道 = ${B(r.shared.oldMetadata)} B
新metadata = ${B(sharedOutputs)} 通道 × 20 B/通道 = ${B(r.shared.newMetadata)} B
8 B = FP32 scale 4 B + FP32 offset 4 B
20 B = INT64 scale 8 B + 保留FP32 scale 4 B + FP32 offset 4 B + folded scale_bias 4 B`:`group32表示每32个权重共用1个U8 scale，不是32 B/scale
旧/新metadata均为 ${B(sharedElements)} 权重元素 ÷ 32 元素/组 × 1 B/U8 scale = ${B(r.shared.oldMetadata)} B`}
净减少 = (旧payload ${B(r.shared.oldPayload)} + 旧metadata ${B(r.shared.oldMetadata)}) − (新payload ${B(r.shared.newPayload)} + 新metadata ${B(r.shared.newMetadata)})
= ${B(r.shared.saved)} B ÷ 1024³ B/GiB = ${G(r.shared.saved)} GiB</div>
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
      ${trace(`主账里的默认运行时残差来源：A3物理HBM采样上限 ${E.physicalSnapshot.maxGiB.toFixed(6)} GiB − 加载权重 ${m.weights_main_plus_draft_gib_measured.toFixed(6)} GiB − 实际缓存池 ${m.kv_pool_physical_gib.toFixed(6)} GiB = ${(E.physicalSnapshot.maxGiB-m.weights_main_plus_draft_gib_measured-m.kv_pool_physical_gib).toFixed(6)} GiB。当前输入使用 ${p.runtimeGiB.toFixed(6)} GiB；A5沿用这项假设或由用户覆盖。以下诊断表的加载、allocated、reserved和设备HBM是仪表读数，不是shape公式计算值，保留其测量来源。`)}
      <div class="mem-note">框架可见 ${r.frameworkCapacityGiB.toFixed(6)} GiB × utilization ${p.utilization} = <b>${G(r.requestedBudgetBytes)} GiB</b> 的请求预算。${r.knownBytes>r.requestedBudgetBytes?`仅已核算权重＋缓存＋新增记录就超预算 ${G(r.knownBytes-r.requestedBudgetBytes)} GiB，<b>不能满足该 utilization</b>。`:'已核算部分在请求预算内，仍需为non-Torch、激活、graph和分配器开销留足空间。'}physical容量有余量不等于框架预算通过；物理残差也不能未经转换再从框架预算重复扣一次。</div>
      <div class="mem-note">本页算的是1M已驻留后的decode。实测跳过MM profiling、使用模拟KV；没有实际1M prefill峰值、精度或prefix命中率验证。全模型权重已含ViT/MM参数；未测的图像激活、1M prefill工作区、不同并发/graph容量变化不能宣称为0，需用上面的额外峰值输入预留并实测。</div>
      <h3>A3 实测校准：分阶段读数，以下不再加到主账</h3>
      ${table(['实测指标','GiB','关系 / 时点'],[
        ['加载权重',m.weights_main_plus_draft_gib_measured,'已在 Torch allocated 内'],
        ['物理 KV backing',rawPool/M.GiB,'62个实测pool ID × 302,579,712 B/ID；后者 = 24层×12288 token×656 B + 24层×12288 token×130 B + 5层×12288 token×1152 B；同一arena只计一次'],
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
