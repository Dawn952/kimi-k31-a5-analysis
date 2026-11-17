/* Incremental A5 DP4/TP8/DCP8 paired-cache capacity model. Bytes per NPU. */
(function(root){
  'use strict';
  const GiB=2**30, TP=8, DP=4, KDA=69, DSA=24, DRAFT=5, HEADS=96/TP, DIM=128;
  const evidence=root.KIMI_DCP_EVIDENCE||(typeof module!=='undefined'?require('../data/memory-dcp-evidence.json'):null);
  function defaults(){return {context:131072,output:1024,blockSize:1536,spec:3,maxSeqs:64,
    requests:60,shared:true,dropTail:false,indexerSharded:true,recover:true,c8:false,
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
    // DSpark needs no EAGLE trailing-block drop. Partial hits still need CoW.
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
  function benchmarkView(){
    const B=root.KIMI_CACHE_BENCHMARK;if(!B)return '';
    const ms=x=>x.toFixed(3),names={KDA_contiguous:'KDA连续缓冲对照',KDA_69_layer_segments:'69层KDA state',attention_one_physical_ID:'完整F物理页对照',KDA_and_one_attention_ID:'KDA + 完整F页对照'};
    return `<h3 id="mem-cache-timing">CoW与P→D实测：操作耗时不等于TTFT增量</h3>
    <div class="mem-note"><b>2026-11-17 · Ascend950DT · 实测微基准。</b>沿用实验环境torch 2.10.0/torch_npu 2.10.0.post7。CoW直接执行归档函数并构造paired张量；65个ID，占用约2.294 GiB，每次尾页复制${b(B.payloadPerCowIdBytes)} B/卡。两测点、4档各100次。原服务仍驻留，测试前观测NPU利用率0%，未独占整机；数据逐区域验证。此处没有测32卡服务TTFT、TPOT或有效未掩盖等待。</div>
    ${table(['测点','同批新请求','CPU提交p50 ms','NPU事件区间p50 ms','完成wall p50 ms','完成wall p95 ms'],B.cow.map(x=>`<tr><td>${x.node}</td><td>${x.batch}</td><td>${ms(x.cpuSubmitMs.p50)}</td><td>${ms(x.deviceEventSpanMs.p50)}</td><td>${ms(x.wallMs.p50)}</td><td>${ms(x.wallMs.p95)}</td></tr>`))}
    <p>CPU提交和NPU执行相互重叠，<b>不能把CPU列与NPU列相加</b>。NPU事件区间包含主机提交间隙，不代表设备kernel一直忙。wall覆盖本次整批复制及完成同步；64请求一批的wall不能除以64作为每条请求等待时间。77个独立缓存平面逐一stack、再逐目的页copy_，所以0.152 ms的纯HBM带宽估算不能代表当前实现实测延迟。</p>
    <h4>“整批64请求”到底怎么执行</h4>
    <p>一次函数接收64对src/dst物理ID，本次都是同一源页→64个不同目的页。独立缓存平面共77个（24 main +24 index key +24 index scale +5 draft），函数逐平面处理：先stack出64份源快照，再对64个目的页逐一copy_。这是<b>一次Python函数调用内的77次stack +4928次copy_</b>，不是一个融合的大CoW指令。这些是两类张量API调用次数，未统计低层kernel/指令数。</p>
    ${table(['测点','64请求提交方式','函数调用数','stack次数','copy_次数','wall p50 ms','wall p95 ms'],B.cowComparison.map(x=>`<tr><td>${x.node}</td><td>${({bulk:'整批调用，末尾等完成',separate_final_wait:'逐请求调用，仅末尾等完成',separate_per_request_wait:'逐请求调用，每次等完成'})[x.mode]}</td><td>${x.helperCalls}</td><td>${x.stackCalls}</td><td>${x.copyCalls}</td><td>${ms(x.wallMs.p50)}</td><td>${ms(x.wallMs.p95)}</td></tr>`))}
    <p>上表是后续新一轮对照，每测点在同一进程、相同65-ID张量上，三种方式依次各预热3次、测30次；环境仍有原服务驻留，不能与上方先前100次测量混成同一次结果。逐请求但不逐次同步也更慢，故收益不能全部归因于减少同步。整批共用一次缓存视图整理，并把4928次小stack变为77次大stack；目的页仍要写64份。64×${b(B.payloadPerCowIdBytes)} = ${b(64*B.payloadPerCowIdBytes)} B有效副本（约2.259 GiB），且同一源在stack里仍被重复装入64份，因此耗时仍远大于单请求。没有逐项profile，不能把总差额精确分摊给视图整理、提交、分配和同步。</p>
    ${table(['方向','内容','payload MiB','段数','warm p50 ms','warm p95 ms'],B.transfer.map(x=>`<tr><td>${x.direction}</td><td>${names[x.name]}</td><td>${(x.bytes/2**20).toFixed(3)}</td><td>${x.segments}</td><td>${ms(x.warmMs.p50)}</td><td>${ms(x.warmMs.p95)}</td></tr>`))}
    <p>使用合成KV缓冲调用真实Mooncake <code>batch_transfer_sync_read</code>，AscendDirectTransport/ADXL，QoS=1；两个方向、4布局各预热10次再测100次。138段KDA含69层conv及recurrent，合计${b(B.kdaStateBytes)} B/卡；分段之间留64 KiB间隔，避免相邻片段直接合并。每个布局首尾校验内容。首笔连接调用分别为 ${B.transfer.filter(x=>x.name==='KDA_contiguous').map(x=>x.direction+' '+ms(x.firstCallMs)+' ms').join('；')}，<b>不能与已建链warm样本混用</b>。单P源会话的微基准未复现P TP8/PP4的多源汇聚、接收队列和布局整理。“完整F页”是整页传输对照，<b>不是声称511-token逻辑缺口实际传了整页</b>；本页尚未采集该真实请求的length_list。</p>
    <h4>哪些耗时能被掩盖</h4>
    ${table(['关系','当前证据与TTFT归因'],[
      '<tr><td>CoW的CPU提交 / NPU复制</td><td>提交与执行交错，wall已包含二者，不能重复计时。</td></tr>',
      '<tr><td>P→D接收 / 其他请求decode</td><td>接收在后台线程异步执行，可以重叠；但仅观察到其他请求在算，不能证明本请求没有增加TTFT。</td></tr>',
      '<tr><td>P→D接收 / 本请求首次forward</td><td>必须等KV就绪；未就绪的尾部等待和错过的调度轮次可能延迟首token。传输ms不能直接等同TTFT增量。</td></tr>',
      '<tr><td>CoW / P→D接收</td><td>worker先提交CoW，随后启动加载；是否存在设备层重叠尚无对应服务trace，不能默认完全串行或完全掩盖。</td></tr>',
      '<tr><td>P prefill / 同请求D接收</td><td>当前bulk路径的代理等待P响应再派发D；不能套用layerwise传输与同请求prefill重叠的假设。</td></tr>'
    ])}
    <p><b>本页不输出“CoW + 传输 = TTFT增量”。</b>要算未掩盖部分，需关联同一请求的D接纳、CoW完成、远端KV完成、调度可进入批次、首次forward及客户端首token时间；同时记录其他批次。只有完整时序或受控对照才能区分资源等待、依赖等待和调度量化，现有微基准不能完成这个归因。</p>`;
  }
  function draw(){
    const r=calculate(p),single=calculate({...p,context:1048576,output:0,requests:1,shared:false});
    const off=calculate({...p,recover:false}),on=calculate({...p,recover:true}),recordPlan=on.rec;
    host.innerHTML=`<h2>A5 · DP4 / TP8 / DCP8 · 共享前缀并发计算器</h2>
    <p>新增于2026-11-17。32张D卡组成4个DP engine，每个engine内8张TP卡同时承担DCP8；DCP不再额外乘8。以下按最繁忙engine核算，所有显存均为每卡。沿用paired布局，主模型93层（69 KDA、24 DSA），另有5层DSpark。</p>
    <div class="mem-note"><b>默认采用同拓扑2026-11-14启动与warmup记录校准。</b>profile时KV约17.57–17.58 GiB，但发生在2.10–2.22 GiB图捕获之前。32个worker中，守住原utilization的最小warmup建议为 ${b(E.postGraphBudgetBytes)} B = ${g(E.postGraphBudgetBytes)} GiB，作为默认净KV预算；不再重复扣已经纳入建议值的runtime。当前模式：<b>${p.budgetMode==='postGraph'?'按图捕获后、符合utilization的预算':p.budgetMode==='allocated'?'按历史已分配池（未回扣graph）':'用户自填预算'}</b>。历史已分配池与修改预算后重启的规划容量分别展示，不能混用。更换max_num_seqs、图档位或软件版本须重新profile。</div>
    <div class="dcp-inputs">
    ${[['context','共享输入 tokens'],['output','每请求输出 tokens'],['blockSize','每卡物理 block_size'],['spec','DSpark推测 tokens'],['requests','展示的请求数 / engine'],['maxSeqs','max_num_seqs / engine'],['kvGiB','图捕获后净KV预算 GiB / 卡'],['weightsGiB','加载驻留扣已含records GiB / 卡']].map(([k,label])=>`<label>${label}<input data-dcp="${k}" type="number" min="0" step="${k.endsWith('GiB')?'any':'1'}" value="${p[k]}"></label>`).join('')}
    <label>其他运行时 GiB / 卡（未知留空）<input data-dcp="runtimeGiB" type="number" min="0" step="0.01" value="${p.runtimeGiB??''}"></label>
    ${[['shared','所有请求共享同一份前缀'],['dropTail','额外私有1个整逻辑块（压力对照，非DSpark实际裁尾）'],['indexerSharded','Kimi Indexer按DCP8切分'],['recover','开启 RecoverSSM'],['c8','DSpark C8（576 B/token容量方案）'],['budgetIncludesRecords','输入KV预算已扣records及量化参数']].map(([k,label])=>`<label class="mem-check"><input data-dcp="${k}" type="checkbox" ${p[k]?'checked':''}>${label}</label>`).join('')}
    </div><div class="mem-toolbar"><button id="dcp-preset">按图捕获后预算（默认）</button><button id="dcp-allocated">按历史已分配497块</button><button id="dcp-json">导出DCP计算JSON</button><a href="#mem-controls">原DP64/DP32 localTP计算器</a></div>
    <div class="mem-kpis"><div class="mem-kpi"><span>净KV预算下的容量规划（整块共享假设）</span><strong>${r.capacity} <small>/ engine</small></strong><small>4个DP合计 ${r.globalCapacity}；按align 2页上限预留，非已验证最大并发</small></div>
    <div class="mem-kpi"><span>再受max_num_seqs=${p.maxSeqs}限制</span><strong>${r.globalAdmitted} <small>全局并发</small></strong><small>${r.admitted}/engine；不代表吞吐或SLO保证</small></div>
    <div class="mem-kpi"><span>当前${p.requests}/engine的完整物理池</span><strong>${g(r.poolBytes)} <small>GiB/卡</small></strong><small>${r.ids} IDs；${r.ids>r.poolIds?'超过当前KV预算':'在当前KV预算内'}${p.requests>p.maxSeqs?'；超过max_num_seqs':''}</small></div></div>
    <h3>四个DP engine的实测预算基线</h3>${table(['engine','profile KV GiB/卡','历史池IDs（推导）','图后守util建议 GiB/卡','新池IDs','共享前缀规划并发'],E.profiles.map(x=>{
      const s=calculate({...p,kvGiB:x.fitRequestedBytes/GiB,budgetIncludesRecords:true});
      return `<tr><td>DP${x.dp}</td><td class="num">${x.profileKvGiB.toFixed(2)}</td><td class="num">${x.allocatedPoolIdsInferred}</td><td class="num">${g(x.fitRequestedBytes)}</td><td class="num">${s.poolIds}</td><td class="num">${Math.min(s.capacity,p.maxSeqs)}</td></tr>`;
    }))}<p>上表均按当前页面的cache几何、共享及RecoverSSM设置重新计算；历史预算仅适用于原spec3/max_num_seqs64的图档位。默认采用最紧张worker预算统一规划；按各DP独立预算并允许不等量分流时，合计为 ${E.profiles.reduce((n,x)=>n+Math.min(calculate({...p,kvGiB:x.fitRequestedBytes/GiB,budgetIncludesRecords:true}).capacity,p.maxSeqs),0)} 条。所有值是容量规划，不是并发完成实测。历史池ID由启动token容量反推并与paired成本复核；不是直接读取的cache-spec dump。</p>
    <h3>整块物理共享账本，不等于APC token命中率</h3><p>物理页 ${b(p.blockSize)} token/卡 × DCP8 = <b>${b(r.coverage)} token/完整逻辑块</b>。本页按整块假设计 ${r.sharedIds} 个共用F组ID，完整块覆盖 <b>${b(r.sharedTokens)} token（输入的${(100*r.sharedTokens/p.context).toFixed(3)}%）</b>；这个比例不是D侧实测APC命中率。${p.shared?'共用前缀每个DP engine分别保存一份，4个engine之间不共享显存。':'本次关闭共享，历史按每请求独立计入。'}</p>
    <div class="mem-note"><b>已纠正：当前DSpark不执行EAGLE额外裁尾，默认不再减1个整块。</b>实验容器的 <code>needs_eagle_block_drop()</code> 明确排除DFlash/DSpark：它们的draft KV仅依赖target hidden state，不依赖下一个token。Scheduler将 <code>use_eagle_block_drop()</code> 传给cache manager，当前为false。此前208/210来自错误套用的整块预留；修正后默认容量规划为240全局并发（60/engine），仍非并发压测结果。</div>
    <details open><summary>当前128K共享前缀场景：D实际复用、CoW和P/D拉取</summary><p><b>有效粒度：</b>同拓扑启动日志的group sizes为(12288,1536,1536,1536)，未指定prefix_match_unit。resolver取GCD，故hash粒度1536，调度对齐LCM为12288；Mamba align与细粒度FullAttention查找使部分物理块也能命中。DSpark不额外裁尾。</p><p><b>预热后：</b>假设本DP内seed的有效前缀仍在cache，D的本地命中为floor((131072−1)/1536)×1536 = <b>130560 token</b> = 10×12288 + 7680。前10个完整F组块直接引用相同物理ID；第11块命中其中7680 token，但继续写入前需CoW（复制到本请求私有页），不能让并发请求共写这一页。131072−130560 = 512 token是剩余prompt，不是再丢一个12288-token整块。</p><p><b>从P补齐：</b>KDA使Mooncake的need_truncate为true。P提供到N−1=131071；D上报本地130560后，external匹配数为<b>511 token</b>，最后1个prompt token由D重算。511是逻辑缺口，不是网络精确传输量；底层按kernel/block范围传输，同时恢复KDA state。请求的1024输出及3个lookahead后，总长132099，仍放在第11个逻辑块内（上限135168），因此历史布局为10个共用完整块+每请求1个私有尾块。</p><p><b>拉取后能否共享：</b>Scheduler收到成功通知会调用cache_blocks发布有效前缀，后来的请求可命中这些hash；但未满物理块的尾页在下次继续写入时仍需CoW。接收路径不会把已经分配给两个请求的不同ID自动合并。冷启动时首条请求本地命中0，拉131071 token；只有接收发布后到来的请求才走上述warm hit。4个DP各自预热和保留一份前缀。RecoverSSM减少投机state副本，不改变这条共享/拉取规则。</p><p>以上是读取实验容器源码与历史配置得到的调用链推导；未新增NPU压测，未采集实际请求block-ID trace。池容量仍按KDA align两页上限规划；独立seed尾页、CoW复制短时引用与checkpoint保留可能另占池，不能把240称为已验证最大并发。</p><p><a href="https://github.com/Dawn952/vllm-ascend/blob/3634a7eae94897d1a7b14d00807323e5af6b4a04/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py">固定版Mooncake缺口与N−1规则</a>；DSpark裁尾开关及细粒度CoW依据已归档实验镜像源码，不能用通用EAGLE行为代替。</p></details>
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
    ${benchmarkView()}
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
