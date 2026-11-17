/* Incremental A5 DP4/TP8/DCP8 paired-cache capacity model. Bytes per NPU. */
(function(root){
  'use strict';
  const GiB=2**30, TP=8, DP=4, KDA=69, DSA=24, DRAFT=5, HEADS=96/TP, DIM=128;
  const evidence=root.KIMI_DCP_EVIDENCE||(typeof module!=='undefined'?require('../data/memory-dcp-evidence.json'):null);
  function defaults(){return {context:131072,output:1024,blockSize:1536,spec:3,maxSeqs:64,
    requests:52,shared:true,dropTail:true,indexerSharded:true,recover:true,c8:false,
    kvGiB:evidence.postGraphBudgetBytes/GiB,budgetIncludesRecords:true,weightsGiB:evidence.loadWithoutRecordsGiB,runtimeGiB:null,budgetMode:'postGraph'};}
  function records(p){
    if(!p.recover)return {correction:0,keyGate:0,metadata:0,total:0};
    const correction=KDA*p.maxSeqs*HEADS*(p.spec+1)*DIM*4;
    const keyGate=KDA*p.maxSeqs*HEADS*(p.spec+1)*DIM*2*2;
    const metadata=KDA*10*8+KDA*HEADS*4+KDA*HEADS*DIM*4+6*p.maxSeqs*4;
    return {correction,keyGate,metadata,total:correction+keyGate+metadata};
  }
  function calculate(p){
    for(const k of ['context','blockSize','maxSeqs','requests'])if(!Number.isSafeInteger(p[k])||p[k]<1)throw Error(k+' 必须是正整数');
    for(const k of ['output','spec'])if(!Number.isSafeInteger(p[k])||p[k]<0)throw Error(k+' 必须是非负整数');
    if(p.blockSize%128)throw Error('物理 block_size 必须为128的整数倍');
    for(const k of ['kvGiB','weightsGiB'])if(!Number.isFinite(p[k])||p[k]<0)throw Error(k+' 必须非负');
    if(p.runtimeGiB!==null&&(!Number.isFinite(p.runtimeGiB)||p.runtimeGiB<0))throw Error('其他运行时必须非负，未知请留空');
    const coverage=p.blockSize*TP,mainPage=p.blockSize*656;
    const conv=(3+p.spec)*3*HEADS*DIM*2,recurrent=HEADS*DIM*DIM*4,state=conv+recurrent;
    if(state>mainPage)throw Error('当前 block_size 的main物理页装不下一份TP8 KDA state');
    const indexPage=p.blockSize*132*(p.indexerSharded?1:TP),draftPage=p.blockSize*(p.c8?576:1152);
    const mainRegion=DSA*mainPage,indexRegion=DSA*indexPage,draftRegion=DRAFT*draftPage;
    const bytesPerId=mainRegion+indexRegion+draftRegion;
    const totalHistory=Math.ceil((p.context+p.output+p.spec)/coverage);
    // Conservative whole-block sharing scenario, not a runtime APC hit predictor.
    // Fine-grained APC can replay one hash unit without losing a whole physical ID.
    const sharedIds=p.shared?Math.max(0,Math.floor(p.context/coverage)-(p.dropTail?1:0)):0;
    const privateHistory=totalHistory-sharedIds;
    const copies=2+(p.recover?0:p.spec),groups=3,stateIdsPerRequest=groups*copies;
    const idsPerRequest=privateHistory+stateIdsPerRequest,ids=sharedIds+p.requests*idsPerRequest+1;
    const rec=records(p),scaleBytes=p.c8?DRAFT*2352:0;
    const available=Math.max(0,Math.floor(p.kvGiB*GiB)-(p.budgetIncludesRecords?0:rec.total+scaleBytes));
    const poolIds=Math.floor(available/bytesPerId);
    const capacity=Math.max(0,Math.floor((poolIds-sharedIds-1)/idsPerRequest));
    const admitted=Math.min(capacity,p.maxSeqs);
    const sharedBytes=sharedIds*bytesPerId,privateBytes=p.requests*privateHistory*bytesPerId;
    const stateContent=p.requests*KDA*copies*state,statePadding=p.requests*KDA*copies*(mainPage-state);
    const holes=p.requests*stateIdsPerRequest*mainPage;
    const sidecars=p.requests*stateIdsPerRequest*(indexRegion+draftRegion);
    const poolBytes=ids*bytesPerId;
    const sum=sharedBytes+privateBytes+stateContent+statePadding+holes+sidecars+bytesPerId;
    if(sum!==poolBytes)throw Error('DCP物理池分项未闭合');
    const knownBytes=p.weightsGiB*GiB+poolBytes+rec.total+scaleBytes;
    return {p,coverage,mainPage,indexPage,draftPage,mainRegion,indexRegion,draftRegion,bytesPerId,
      conv,recurrent,state,totalHistory,sharedIds,sharedTokens:sharedIds*coverage,
      privateHistory,copies,stateIdsPerRequest,idsPerRequest,ids,poolIds,capacity,admitted,
      globalCapacity:capacity*DP,globalAdmitted:admitted*DP,rec,available,sharedBytes,privateBytes,
      stateContent,statePadding,holes,sidecars,poolBytes,scaleBytes,knownBytes,
      totalBytes:p.runtimeGiB===null?null:knownBytes+p.runtimeGiB*GiB};
  }
  const api={defaults,records,calculate,GiB,DP,TP};
  root.KimiDcpMemory=api;if(typeof module!=='undefined')module.exports=api;
  if(typeof document==='undefined')return;
  const host=document.getElementById('mem-dcp');if(!host)return;
  const E=evidence;let p=defaults();
  const g=x=>(x/GiB).toFixed(6),b=x=>Math.round(x).toLocaleString('en-US');
  const tr=(label,bytes,explain)=>`<tr><td>${label}</td><td class="num">${g(bytes)}</td><td>${explain}</td></tr>`;
  const table=(headers,rows)=>`<div class="mem-table-wrap"><table class="mem-table"><thead><tr>${headers.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  function draw(){
    const r=calculate(p),single=calculate({...p,context:1048576,output:0,requests:1,shared:false});
    const off=calculate({...p,recover:false}),on=calculate({...p,recover:true}),recordPlan=on.rec;
    host.innerHTML=`<h2>A5 · DP4 / TP8 / DCP8 · 共享前缀并发计算器</h2>
    <p>新增于2026-11-17。32张D卡组成4个DP engine，每个engine内8张TP卡同时承担DCP8；DCP不再额外乘8。以下按最繁忙engine核算，所有显存均为每卡。沿用paired布局，主模型93层（69 KDA、24 DSA），另有5层DSpark。</p>
    <div class="mem-note"><b>默认采用同拓扑2026-11-14启动与warmup记录校准。</b>profile时KV约17.57–17.58 GiB，但发生在2.10–2.22 GiB图捕获之前。32个worker中，守住原utilization的最小warmup建议为 ${b(E.postGraphBudgetBytes)} B = ${g(E.postGraphBudgetBytes)} GiB，作为默认净KV预算；不再重复扣已经纳入建议值的runtime。当前模式：<b>${p.budgetMode==='postGraph'?'按图捕获后、符合utilization的预算':p.budgetMode==='allocated'?'按历史已分配池（未回扣graph）':'用户自填预算'}</b>。历史已分配池与修改预算后重启的规划容量分别展示，不能混用。更换max_num_seqs、图档位或软件版本须重新profile。</div>
    <div class="dcp-inputs">
    ${[['context','共享输入 tokens'],['output','每请求输出 tokens'],['blockSize','每卡物理 block_size'],['spec','DSpark推测 tokens'],['requests','展示的请求数 / engine'],['maxSeqs','max_num_seqs / engine'],['kvGiB','图捕获后净KV预算 GiB / 卡'],['weightsGiB','加载驻留扣已含records GiB / 卡']].map(([k,label])=>`<label>${label}<input data-dcp="${k}" type="number" min="0" step="${k.endsWith('GiB')?'any':'1'}" value="${p[k]}"></label>`).join('')}
    <label>其他运行时 GiB / 卡（未知留空）<input data-dcp="runtimeGiB" type="number" min="0" step="0.01" value="${p.runtimeGiB??''}"></label>
    ${[['shared','所有请求共享同一份前缀'],['dropTail','额外私有1个整逻辑块（保守重放假设）'],['indexerSharded','Kimi Indexer按DCP8切分'],['recover','开启 RecoverSSM'],['c8','DSpark C8（576 B/token容量方案）'],['budgetIncludesRecords','输入KV预算已扣records及量化参数']].map(([k,label])=>`<label class="mem-check"><input data-dcp="${k}" type="checkbox" ${p[k]?'checked':''}>${label}</label>`).join('')}
    </div><div class="mem-toolbar"><button id="dcp-preset">按图捕获后预算（默认）</button><button id="dcp-allocated">按历史已分配497块</button><button id="dcp-json">导出DCP计算JSON</button><a href="#mem-controls">原DP64/DP32 localTP计算器</a></div>
    <div class="mem-kpis"><div class="mem-kpi"><span>净KV预算下的容量规划（整块共享假设）</span><strong>${r.capacity} <small>/ engine</small></strong><small>4个DP合计 ${r.globalCapacity}；按align 2页上限预留，非已验证最大并发</small></div>
    <div class="mem-kpi"><span>再受max_num_seqs=${p.maxSeqs}限制</span><strong>${r.globalAdmitted} <small>全局并发</small></strong><small>${r.admitted}/engine；不代表吞吐或SLO保证</small></div>
    <div class="mem-kpi"><span>当前${p.requests}/engine的完整物理池</span><strong>${g(r.poolBytes)} <small>GiB/卡</small></strong><small>${r.ids} IDs；${r.ids>r.poolIds?'超过当前KV预算':'在当前KV预算内'}${p.requests>p.maxSeqs?'；超过max_num_seqs':''}</small></div></div>
    <h3>四个DP engine的实测预算基线</h3>${table(['engine','profile KV GiB/卡','历史池IDs（推导）','图后守util建议 GiB/卡','新池IDs','共享前缀规划并发'],E.profiles.map(x=>{
      const s=calculate({...p,kvGiB:x.fitRequestedBytes/GiB,budgetIncludesRecords:true});
      return `<tr><td>DP${x.dp}</td><td class="num">${x.profileKvGiB.toFixed(2)}</td><td class="num">${x.allocatedPoolIdsInferred}</td><td class="num">${g(x.fitRequestedBytes)}</td><td class="num">${s.poolIds}</td><td class="num">${Math.min(s.capacity,p.maxSeqs)}</td></tr>`;
    }))}<p>上表均按当前页面的cache几何、共享及RecoverSSM设置重新计算；历史预算仅适用于原spec3/max_num_seqs64的图档位。默认采用最紧张worker预算统一规划；按各DP独立预算并允许不等量分流时，合计为 ${E.profiles.reduce((n,x)=>n+Math.min(calculate({...p,kvGiB:x.fitRequestedBytes/GiB,budgetIncludesRecords:true}).capacity,p.maxSeqs),0)} 条。所有值是容量规划，不是并发完成实测。历史池ID由启动token容量反推并与paired成本复核；不是直接读取的cache-spec dump。</p>
    <h3>整块物理共享账本，不等于APC token命中率</h3><p>物理页 ${b(p.blockSize)} token/卡 × DCP8 = <b>${b(r.coverage)} token/完整逻辑块</b>。本页按整块假设计 ${r.sharedIds} 个共用F组ID，完整块覆盖 <b>${b(r.sharedTokens)} token（输入的${(100*r.sharedTokens/p.context).toFixed(3)}%）</b>；这个比例不是D侧实测APC命中率。${p.shared?'共用前缀每个DP engine分别保存一份，4个engine之间不共享显存。':'本次关闭共享，历史按每请求独立计入。'}</p>
    <div class="mem-note"><b>额外减1个整块是本页的保守物理排布假设，尚未由该D服务的有效匹配粒度与命中block IDs验证。</b>已读取的实验容器源码支持细粒度APC/CoW：FullAttentionManager裁尾为 <code>hit_length -= min(alignment_tokens, block_size)</code>。细粒度模式可能只退一个hash单位，并不丢掉一整份${b(r.coverage)}-token物理块；关闭上述整块选项也不等于关闭DSpark重放。此前208/210等数值均须连同本页整块策略理解，不能当作已验证的真实最大并发。</div>
    <details><summary>向下取整、DSpark裁尾和P/D拉取分别做什么</summary><p><b>取整：</b>131072 = 10×12288 + 8192。10个完整逻辑块与8192-token部分块分开核算，取整本身没有再丢一个完整块。<b>裁尾：</b>本地前缀查找在EAGLE/DSpark路径下缩短复用边界，以取得draft需要的hidden states并保护相关验证窗口；这是命中列表的裁剪，不是把旧KV从显存删除。只有整块匹配/重放策略才可写成10−1=9；细粒度模式不能直接套这个式子。</p><p><b>拉取：</b>Mooncake根据外部可提供token数减去本地已命中token数，向该请求的新分配物理块补齐KV。以本页整块假设为例，本地保留110592 token；若P提供131071 token，需拉20479 token，再由D重算最后1个prompt token。这条N−1规则是1个token，与额外裁1个block不同。</p><p><b>共享：</b>异步拉取成功后，Scheduler调用cache_blocks登记有效前缀，后续请求可以复用。被当前请求裁掉的块不会被自动附回它的block table；外部缺口经get_new_blocks分配后，即使内容相同，也不是自动与旧块共享同一物理ID。只有请求引用相同block ID，才可在并发显存账本里只计一份。需要先完成seed预热；多个冷请求同时拉相同prompt，也不保证物理去重。</p><p><a href="https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/v1/core/single_type_kv_cache_manager.py">前缀裁尾与外部分配源码</a> · <a href="https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/v1/core/sched/scheduler.py">拉取完成后cache_blocks</a> · <a href="https://github.com/Dawn952/vllm-ascend/blob/3634a7eae94897d1a7b14d00807323e5af6b4a04/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py">Mooncake外部缺口与N−1规则</a></p></details>
    <div class="mem-code">每ID成本 = 24层 × (${b(p.blockSize)} token × 656 B主MLA + ${b(p.blockSize)} token × 132 B索引 × ${p.indexerSharded?1:8})
             + 5层 × ${b(p.blockSize)} token × ${p.c8?576:1152} B普通MLA
           = ${b(r.bytesPerId)} B = ${g(r.bytesPerId)} GiB
完整历史ID/请求 = ceil((${b(p.context)} 输入 + ${b(p.output)} 输出 + ${p.spec} lookahead) / ${b(r.coverage)}) = ${r.totalHistory}
共享ID（整块假设） = ${p.shared?`max(0, floor(${b(p.context)} / ${b(r.coverage)}) − ${p.dropTail?1:0} 整块重放预留)`:'0（关闭共享）'} = ${r.sharedIds}
私有历史ID/请求 = ${r.totalHistory} − ${r.sharedIds} = ${r.privateHistory}
KDA ID/请求 = 3组 × (2个align基础页 + ${p.recover?0:p.spec}个投机state页) = ${r.stateIdsPerRequest}
总池ID = ${r.sharedIds} 共享 + ${p.requests} 请求 × (${r.privateHistory} 私有历史 + ${r.stateIdsPerRequest} KDA) + 1 null = ${r.ids}
净KV池ID = floor(${b(r.available)} B可用 / ${b(r.bytesPerId)} B每ID) = ${r.poolIds}
并发容量/engine = floor((${r.poolIds} − ${r.sharedIds} 共享 − 1 null) / ${r.idsPerRequest} ID每请求) = ${r.capacity}</div>
    <h3>完整物理池逐项对账</h3>${table(['分项','GiB/卡','计算式与含义'],[
      tr('共享历史ID全部区域',r.sharedBytes,`${r.sharedIds} ID × ${b(r.bytesPerId)} B/ID；每engine只计一次`),
      tr('私有历史ID全部区域',r.privateBytes,`${p.requests} 请求 × ${r.privateHistory} ID/请求 × ${b(r.bytesPerId)} B/ID；含尾块空位`),
      tr('KDA state有效内容',r.stateContent,`69层 × ${p.requests} 请求 × ${r.copies} 副本 × ${b(r.state)} B；每份recurrent ${b(r.recurrent)} B + conv ${b(r.conv)} B`),
      tr('KDA页padding',r.statePadding,`69层 × ${p.requests} 请求 × ${r.copies} 副本 × (${b(r.mainPage)} − ${b(r.state)}) B`),
      tr('KDA组内空main slot',r.holes,`${p.requests} 请求 × ${r.stateIdsPerRequest} KDA IDs × 1空slot × ${b(r.mainPage)} B`),
      tr('KDA ID空Indexer与draft区',r.sidecars,`${p.requests*r.stateIdsPerRequest} KDA IDs × (${b(r.indexRegion)} 索引 + ${b(r.draftRegion)} draft) B`),
      tr('全池1个null ID',r.bytesPerId,`1 × ${b(r.bytesPerId)} B；每卡各自保留，已含所有层区域`),
      tr('物理池合计',r.poolBytes,`${r.ids} IDs × ${b(r.bytesPerId)} B；分项严格相等`)
    ])}
    <h3>dtype、切分及RecoverSSM临时记录</h3><p>主MLA：512个C8 latent + 64个BF16 RoPE + 4个FP32 scale = 656 B/token/层。Indexer：128个FP8 key + 1个FP32 scale = 132 B。draft普通MLA：576个BF16元素 = 1152 B。三者按DCP8分token；旧replicated Indexer模式会额外×8索引区。<b>KDA不按DCP切分，而按TP8：96/8=12 heads。</b></p>
    <div class="mem-code">每层recurrent = 12 heads × 128 V维 × 128 K维 × 4 B/FP32 = ${b(r.recurrent)} B
每层conv = (4−1+${p.spec})历史位置 × 3路QKV × 12 heads × 128维 × 2 B/BF16 = ${b(r.conv)} B
若开启RecoverSSM，correction = 69层 × ${p.maxSeqs} 请求槽 × 12 heads × (${p.spec}+1)位置 × 128维 × 4 B = ${b(recordPlan.correction)} B
key/gate = 69 × ${p.maxSeqs} × 12 × (${p.spec}+1) × (2×128) × 2 B = ${b(recordPlan.keyGate)} B
metadata = 69×10×8 B指针/stride + 69×12×4 B A_log + 69×12×128×4 B dt_bias + 6×${p.maxSeqs}×4 B = ${b(recordPlan.metadata)} B
开启时records总计 = ${b(recordPlan.total)} B = ${g(recordPlan.total)} GiB；按max_num_seqs分配，不按活跃数或pool IDs分配。
当前开关${p.recover?'开':'关'}，实际计入records = ${b(r.rec.total)} B。</div>
    <p>${p.budgetIncludesRecords?'本次输入是已扣records/量化参数的KV预算，因此不重复扣。':'本次预算尚未扣records/量化参数，已先扣再向下取整分配KV池。'} RecoverSSM只移除${p.spec}份投机完整state，保留align基础容量；conv的推测历史不删除。开关前后同一请求数的物理池：关闭 ${g(off.poolBytes)} GiB，开启 ${g(on.poolBytes)} GiB（当前开关${p.recover?'开':'关'}）；临时记录须单独计入。</p>
    <h3>1M单请求对照与完整HBM口径</h3>${table(['项目','GiB/卡','口径'],[
      tr('同拓扑1M输入、无输出、单请求KV池',single.poolBytes,`${single.totalHistory} 历史 IDs + 3×${single.copies} KDA IDs + 1 null；共享关闭。仅为KV容量外推；日志max_model_len=132096，该服务配置不能直接接1M，须调长并重新profile`),
      tr('加载驻留（扣除已含records payload）',p.weightsGiB*GiB,`同拓扑加载增量61.08 GiB − 加载时已分配correction/key_gate ${g(E.recordsInMeasuredLoadBytes)} GiB；保留其余loader驻留，不冒充纯checkpoint权重`),
      tr('RecoverSSM记录及metadata',r.rec.total,'KV池之外的请求槽记录；计算完整HBM时只计一次'),
      tr('C8独立量化参数',r.scaleBytes,'启用时按5层×2352 B容量估计，未证明draft C8算子已支持'),
      `<tr><td>通信、图、激活、allocator及其他运行时合计</td><td class="num">${p.runtimeGiB===null?'n.a.':p.runtimeGiB.toFixed(6)}</td><td>目标拓扑未测；未知保持n.a.，不当0，不迁移A3残差</td></tr>`,
      `<tr><td>当前请求数完整HBM参考总量</td><td class="num">${r.totalBytes===null?'n.a.':g(r.totalBytes)}</td><td>扣已含records后的加载驻留 + 所需物理池 + records + C8参数 + 其他运行时；61.08本身含records，不可原样再叠加；不重复加Torch reserved</td></tr>`
    ])}
    <p>容量按align最多2页/层/请求保守规划；稳态可能仅1页。保留额外旧前缀、多份seed、外部KV临时页或不均衡DP负载会继续占池。输出1024只用来约束尾部容量；没有吞吐/SLO或max_num_batched_tokens保证。共享页可被前缀缓存淘汰，反复命中还需运行证据。</p>
    <details><summary>来源、版本和逐层排布</summary><p>paired几何、DCP切分与Kimi分片Indexer依据源码 ${E.layoutHead}，RecoverSSM公式对照已合入PR21549，head ${E.recoverHead}。预算来自11月14日review_s1实验镜像日志，非合入head原样部署；启动CLI请求768，KDA页对齐后的1536几何与日志token容量相符。旧localTP计算器仍保留11月12日基线，不能把它的TP1记录大小直接用于TP8。<a href="${E.prUrl}">PR21549及A5实验条件</a>；<a href="${E.layoutUrl}">固定版paired布局</a>。此新增项是历史测量校准的容量分析，非本场景并发压测。</p><div class="mem-code">KDA L0/1/2、4/5/6…88/89/90：每层[12,128,128] FP32 recurrent及[${3+p.spec},4608] BF16 conv。
M1层0/4/…/88、M2层1/5/…/89、M3层2/6/…/90：各23层，overlay 24 main slots中的23个；三组使用不同ID。
DSA L3/7/…/91及L92：各自独立[block,${p.blockSize},1,656] packed主缓存及[block,${p.blockSize},1,128] FP8索引 + FP32 scale（旧replicated索引另乘DCP8）。
DSpark L93–97：5个独立普通MLA缓存，每元素${p.c8?'C8':'BF16'}；共享F组ID，不共享各层内容。
同一ID在每层tensor中对应不同区域；每ID字节成本是24份main +24份indexer +5份draft的和。</div></details>`;
    host.querySelectorAll('[data-dcp]').forEach(el=>el.addEventListener('change',()=>{
      const k=el.dataset.dcp,v=el.type==='checkbox'?el.checked:(k==='runtimeGiB'&&el.value===''?null:Number(el.value));
      const prev=p;p={...p,[k]:v,budgetMode:k==='kvGiB'?'custom':p.budgetMode};try{calculate(p);draw();}catch(e){p=prev;el.setCustomValidity(e.message);el.reportValidity();}
    }));
    document.getElementById('dcp-preset').onclick=()=>{p=defaults();draw();};
    document.getElementById('dcp-allocated').onclick=()=>{const q=defaults(),x=calculate(q);p={...q,kvGiB:E.allocatedPoolIds*x.bytesPerId/GiB,requests:60,budgetMode:'allocated'};draw();};
    document.getElementById('dcp-json').onclick=()=>{
      const data=JSON.stringify({evidence:E,result:r,oneMillion:single},null,2),url=URL.createObjectURL(new Blob([data],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download='kimi-a5-dp4-tp8-dcp8-memory.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
    };
  }
  draw();
})(typeof window!=='undefined'?window:globalThis);
