/* Actual localTP capture, independent of the old what-if weight calculator. */
(function(root){
  function create(data){
    const names={kda:'KDA',dsa:'DSA',shared:'Shared expert',dense:'Dense MLP',head:'LM head',embedding:'Embedding'};
    const pairNames={
      'ag:f':'input AG ↔ F_a/F_b','quant:beta':'MX quant ↔ beta BF16',
      'gr:quant':'G A2A ↔ MX quant','ret:zero':'QKV/beta A2A ↔ ZerosLike',
      'gr:core':'G A2A ↔ RecurrentKDA','ret:g':'QKV/beta A2A ↔ G GEMM',
      'ag:prolog':'G input AG ↔ MLA prolog','gr:idx':'G A2A ↔ Indexer',
      'gr:keyproj':'G A2A ↔ index K/weight GEMM','qprep:kprep':'index Q aux ↔ K/weights main',
      'ag:latent_down':'shared AG ↔ latent down','rs:latent_up':'shared RS ↔ latent up',
      'up:gmm1':'shared gate/up ↔ routed GMM1','down:combine':'shared down ↔ routed combine',
      'ag1:up0':'AG chunk 1 ↔ gate/up chunk 0','rs0:up1':'RS chunk 0 ↔ gate/up chunk 1',
      'ag1:mm0':'AG chunk 1 ↔ head GEMM chunk 0','a2a0:mm1':'logits A2A chunk 0 ↔ head GEMM chunk 1',
      'rs0:lookup1':'RS chunk 0 ↔ lookup/mask chunk 1'};
    const source=file=>'https://github.com/Dawn952/vllm-ascend/blob/'+data.ascend_revision+'/vllm_ascend/'+file;
    const fmt=x=>Number(x).toFixed(3);
    function graph(family){
      const domain=data.domains[family],nodes=[],edges=[],parallelGroups=[],windows=[];
      const {events,groups}=domain.first;
      function n(id,title,col,row,lines,description,kind='op',file='ops/finegrained_tp.py'){
        const records=(groups[id]||[]).map(i=>events[i]);
        const start=records.length?Math.min(...records.map(x=>x.start_us)):null;
        const end=records.length?Math.max(...records.map(x=>x.start_us+x.duration)):null;
        const streams=[...new Set(records.map(x=>x.stream))].join('/');
        const cores=[...new Set(records.map(x=>x.core))].join(' / ');
        nodes.push({id,title,col,row,kind,lines,description,source:file,records,
          status:records.length?`${fmt(start)}–${fmt(end)} μs · ${streams==='N/A'?'comm':('s'+streams)}`:'源码依赖 / 数据归属',
          engine:cores,sourceRevision:data.ascend_revision,evidenceKind:records.length?'device':'source'});
      }
      function e(src,dst,label='数据就绪',shape='',order=false){edges.push({id:'p'+edges.length,src,dst,label,shape,order,
        description:order?'同流/通信发起顺序或源码显式等待，不等同于张量数学依赖。':'消费者必须等待生产者完成；首例时序来自同一层/调用，非跨样本拼接。'});}
      function seq(ids){for(let i=1;i<ids.length;i++)e(ids[i-1],ids[i]);}
      function frame(left,right,color){
        const key=left+':'+right,stat=domain.overlap_stats[key];
        if(stat.median_us>0)parallelGroups.push({id:key,color,title:`交集中位 ${stat.median_us} μs`,parts:[[left],[right]]});
      }
      function output(from,row){n('out','返回 owner / 下一模块',0,row,['[16,7168] · 保留 token 顺序','完成 join 后才可消费'],
        '这是返回结果的依赖边界，不额外假造一个 device kernel。');e(from,'out');}
      if(family==='kda'){
        const file='ops/kimi_kda.py';
        n('in','AttnRes / owner 输入',0,0,['[16,7168] · DP owner','每卡请求不同'],'本例每 owner capacity=16；不能将全局 C128 并发误认为每卡 token 行数。','op',file);
        n('f','F_a → F_b（本地）',1,1,['16×7168→128→12288','辅助流 422 · 两次 GEMM'],'本地低秩 F；连续输出 reshape 不是独立计算。','op',file);
        n('ag','Input AG',3,1,['[16,H] → [128,H]','localTP8 · 不收集 KV 状态'],'与 F 两次 GEMM 有实际时间交集。','comm');
        n('g','G BF16 GEMM',1,2,['[128,H] → [128,1536]','主流 425 · MIX_AIC'],'本部署在 AG/F 完成后先算 G，再执行 QKV 输入量化与投影。','op',file);
        n('quant','QKV input MX quant',2,3,['[128,7168] → FP8 + scale','主流 425 · MIX_AIV'],'与辅助流 beta BF16、提前返回的 G 通信有交集。','op',file);
        n('beta','beta BF16 GEMM',1,3,['[128,H] → [128,12]','辅助流 422 · MIX_AIC'],'beta 原始 BF16 与 QKV 合并通信；接收后才 float/sigmoid。','op',file);
        n('gr','G A2A（提前返回）',3,3,['[128,1536] → owner G','此时尚未进入 Recurrent'],'本 capture 的 690 层样本中，它与 Recurrent 的交集均为 0；不能再计算一次隐藏收益。','comm');
        n('q','QKV MXFP GEMM',1,4,['[128,H] → [128,4608]','主流 425 · AI_CORE'],'等 MX quant 与 beta Cube 完成。没有 beta/G 后处理与此 GEMM 的实际重叠。','op',file);
        n('cat','Concat QKV + raw beta',2,5,['4608 + 12 → 4620','实际有发送端 copy'],'输出 [128,4620]；两者不是独立的 beta/QKV 消息。');
        n('zero','ZerosLike',2,6,['初始化 owner core 输出','与返回通信局部相交'],'独立初始化，不读取 QKV 接收结果。','op',file);
        n('ret','QKV + beta 合并 A2A',3,6,['[128,4620] BF16','一笔 all_to_all_single'],'一笔消息同时携带 QKV 与 raw beta；G GEMM 此时已完成。','comm');
        n('restore','QKV 重排 / beta 后处理',2,7,['AsStrided → Slice → Transpose','→ Cast FP32 → Sigmoid'],'接收后这些 device kernel 全在主流 425 串行执行，位于 Conv 之前。','op',file);
        n('conv','CausalConv1d',2,8,['owner Q/K/V [16,36864]','主流 425 · MIX_AIV'],'读取并更新 owner conv state。','op',file);
        n('state','owner recurrent / conv state',0,9,['状态不按 localTP 再切','缓存数据，不是计算引擎'],'Recurrent 内部读写状态；不将缓冲区画成可重叠算子。','cache',file);
        n('core','RecurrentKDA',2,9,['等 conv QKV + F + sigmoid beta','主流 425 · AI_VECTOR_CORE'],'完整 owner heads 参与状态递推，G 返回此时早已完成。','op',file);
        n('grestore','G owner Transpose',2,10,['[8,16,12,128]','→ [16,8,12,128]'],'实际到 Recurrent 后才在主流物化；未提前到 QKV Cube 窗口。');
        n('norm','Gated RMSNorm',2,11,['core × owner G','逐 head_dim=128 归一化'],'等待 core 与 G owner 重排。','op',file);
        n('olayout','O 发送布局 Transpose',2,12,['[16,8,1536] → [8,16,1536]','实际 device copy'],'位于 gated norm 后，随后才能发 output heads A2A。');
        n('oa','O heads A2A',3,13,['owner heads → [128,1536]','完整 head 保持成组'],'给按输入 head 维切分的 O 权重。','comm');
        n('o','O BF16 GEMM',1,14,['[128,1536] × W_O shard','partial [128,7168]'],'这是各 head shard 的 hidden 部分和。','op','ops/linear_op.py');
        n('rs','O output RS',3,15,['sum → owner [16,7168]','同 chunk 必须等待 O GEMM'],'O A2A→GEMM→RS 在实测里仍为串行尾部。','comm');output('rs',16);
        e('in','f');e('in','ag');e('ag','g');e('f','g','源码等待 F_b / Cube 次序','',true);
        e('g','gr');e('g','quant','主流次序','',true);e('ag','beta');e('g','beta','本部署 Cube 次序','',true);e('quant','q');e('beta','q','显式 wait beta','',true);
        e('q','cat');e('beta','cat');e('cat','zero','主流次序','',true);e('cat','ret');e('ret','restore');
        e('zero','restore','主流次序','',true);seq(['restore','conv','core','grestore','norm','olayout','oa','o','rs']);
        e('f','core','owner raw F');e('state','core','读写状态');e('zero','core','输出 buffer ready');e('gr','grestore','已返回的 G');e('core','norm');
        frame('ag','f','#71dac8');frame('quant','beta','#efc978');frame('ret','zero','#a996ee');
        windows.push(['G 返回与 Recurrent 的实际位置','690 个 KDA 样本的交集 min/median/max 均为 0；G A2A 实际位于 MX quant / beta BF16 附近。'],
          ['接收后的主流处理','QKV/beta 合并返回 → QKV owner 重排 → beta Slice/Transpose/Cast/Sigmoid → Conv → Recurrent。beta 后处理位于 Conv 之前。']);
      }else if(family==='dsa'){
        const file='attention/sfa_v1.py';
        n('in','DSA AttnRes 输入',0,0,['owner [16,7168]','TP1 / DCP1'],'实际 pure DP localTP；没有 DCP query/candidate/output 归并。','op',file);
        n('ag','G input AG',3,1,['[16,H] → [128,H]','与实际 MLA prolog 相交'],'这是 localTP G 输入，不是 DCP query heads AG。','comm');
        n('quant','MLA input MX quant',2,1,['owner [16,7168]','主流 425'],'先量化后进入 fused MLA prolog。','op',file);
        n('prolog','MlaPrologV3',1,2,['owner Q / KV / UK 准备','主流 425 · MIX_AIC'],'本节点同时使用 Cube/Vector；不将整段标成纯 Vector。','op',file);
        n('g','G BF16 GEMM',1,3,['[128,H] → [128,1536]','主流 425'],'实际在 prolog 后、index K/weights 投影前执行。','op',file);
        n('keyproj','Indexer K / weights 投影',1,4,['[16,H] → [16,160]','128 key + 32 weights'],'与 G 返回有真实部分交集。','op',file);
        n('gr','G A2A（提前返回）',3,4,['返回 owner [16,12288]','不等 Indexer/SFA 完成'],'240 个 DSA 样本中，与 Indexer 的交集均为 0。','comm');
        n('qprep','Indexer Q 准备',1,5,['QMM → MatMul → MatMul','辅助流 417 · AI_CORE'],'实际辅助流 Q 投影，与主流 key norm/quant 有交集。','op',file);
        n('kprep','Indexer K / weights 准备',2,5,['Slice / norm / cast / quant','主流 425 · Vector/MIX_AIV'],'含 cache write；不是整条 DSA 都可与 G Cube 并行。','op',file);
        n('idx','QuantLightningIndexer',1,6,['owner Q/K/weights → top-k','MIX_AIC · 非纯 Vector'],'等 Q aux 与 K/weights main 两条准备分支。','op',file);
        n('cache','owner compressed KV / index cache',0,6,['localTP 不切历史状态','由 prolog / K prep 写入'],'数据缓冲，不占一个独立“计算窗口”。','cache',file);
        n('concat','Q nope / rope Concat',2,7,['[16,96,512] + [16,96,64]','→ [16,96,576]'],'实测有设备执行，不按 view 忽略。','op',file);
        n('sfa','Quant Sparse Attention',1,8,['top-k + owner KV','MIX_AIC'],'不是 DCP attention，也没有其 output A2A/LSE combine。','op',file);
        n('uv','UV up projection',1,9,['[16,96,512] → [16,96,128]','TransposeBatchMatMul · AI_CORE'],'UV 与 G 都占 Cube，实测按队列执行。','op',file);
        n('grestore','G owner Transpose',2,10,['[8,16,12,128] → owner heads','主流 425'],'实际位于 UV 后，G 接收已提前完成。','op',file);
        n('gate','sigmoid(G) × UV output',2,11,['owner [16,12288]','Vector gate kernel'],'仍在 owner gate 后才分发 O heads；不是 G/O co-location 候选。','op',file);
        n('olayout','O 发送布局 Transpose',2,12,['[16,8,1536] → peer-major','主流 425'],'随后发 O heads A2A。');
        n('oa','O heads A2A',3,13,['→ [128,1536]','localTP8'],'这是 O 输入交换，不是旧 DCP output/LSE 归并。','comm');
        n('o','O BF16 GEMM',1,14,['→ partial [128,7168]','AI_CORE'],'按输入 head shard 计算部分和。','op','ops/linear_op.py');
        n('rs','O output RS',3,15,['→ owner [16,7168]','sum + owner rows'],'等待本 chunk 的 O GEMM。','comm');output('rs',16);
        e('in','ag');seq(['in','quant','prolog','g','keyproj']);e('ag','g');e('g','gr');e('prolog','qprep');e('keyproj','kprep');
        e('qprep','idx');e('kprep','idx');e('prolog','cache');e('kprep','cache');e('prolog','concat');e('idx','concat','主流次序','',true);
        seq(['concat','sfa','uv','grestore','gate','olayout','oa','o','rs']);e('idx','sfa','top-k');e('cache','sfa','KV read');e('gr','grestore');e('uv','gate');
        frame('ag','prolog','#71dac8');frame('qprep','kprep','#efc978');
        windows.push(['G 投影与返回位置','G input AG 与 prolog 有交集；G return 与 Indexer/SFA 不构成实测覆盖窗口。本图采用 localTP 的设备样本。']);
      }else if(family==='shared'){
        const file='ops/fused_moe/shared_experts.py';
        n('in','MoE AttnRes 输入',0,0,['owner [16,7168]','shared 与 routed 起点'],'本图记录 3634a7e 部署的设备行为。','op',file);
        n('router','Router GEMM',1,1,['[16,7168] → [16,896]','主流 425 · MIX_AIC'],'实际先产出 router logits。','op',file);
        n('ag','Shared input AG',3,2,['[16,7168] → [128,7168]','localTP8'],'本 capture 先完成 AG，再运行 latent down；交集由设备时间区间确认。','comm');
        n('down_quant','Latent down input MX quant',2,3,['owner [16,7168]','主流 425'],'属于 routed 输入变换，不是 shared input quant。','op',file);
        n('latent_down','Routed latent down GEMM',1,4,['[16,7168] → [16,3584]','主流 425 · AI_CORE'],'本例未与 shared AG 重叠。','op','ops/fused_moe/fused_moe.py');
        n('quant','Shared input MX quant',2,5,['[128,7168] → FP8 + scale','主流 425'],'两支的 token/hidden 维度不同，不能混为同一量化。','op',file);
        n('up','Shared gate/up GEMM',1,6,['[128,7168] → [128,1536]','主流 425 · MIX_AIC'],'按 intermediate 6144/8=768；gate/up 合并 1536 列。','op',file);
        n('topk','Routed TopK',2,7,['[16,896] → [16,16]','主流 425'],'设备上排在 shared gate/up 之后。','op',file);
        n('dispatch','Routed dispatch',2,8,['EP32 · owner hidden=3584','MoeDistributeDispatchV3'],'这是 routed EP 通信/计算融合，不计为 shared AG。','op',file);
        n('gmm1','Routed GMM1 + SiTU quant',1,9,['动态 routed token buffer','MIX_AIC'],'shared 激活在这份 capture 中位于它之后。','op',file);
        n('gmm2','Routed GMM2',1,10,['→ expert partial hidden=3584','MIX_AIC'],'保持真实 routed 路径。','op',file);
        n('combine','Routed combine',2,11,['EP32 → owner [16,3584]','MoeDistributeCombineV3'],'设备完成后才执行 shared 激活。','op',file);
        n('act','Shared SiTU + MX quant',2,12,['[128,1536] → [128,768]','SituMxQuant · 主流 425'],'实测位于 combine 后、shared down GEMM 前。','op',file);
        n('down','Shared down GEMM',1,13,['[128,768] → [128,7168]','主流 425 · AI_CORE'],'shared weight shard 对所有 owner 行产生部分和。','op',file);
        n('rs','Shared output RS',3,14,['sum → owner [16,7168]','localTP8'],'本 capture 中完成后才执行 latent up。','comm');
        n('up_quant','Latent up norm + MX quant',2,15,['owner [16,3584]','RmsNormDynamicMxQuant'],'属于 routed 输出变换。','op','ops/fused_moe/fused_moe.py');
        n('latent_up','Routed latent up GEMM',1,16,['[16,3584] → [16,7168]','主流 425'],'与 shared RS 的实测交集为 0。','op','ops/fused_moe/fused_moe.py');
        n('add','Shared + routed',2,17,['owner [16,7168]','ForeachAddList'],'二者都就绪后相加。','op','ops/fused_moe/fused_moe.py');output('add',18);
        seq(['in','router','ag','down_quant','latent_down','quant','up','topk','dispatch','gmm1','gmm2','combine','act','down','rs','up_quant','latent_up','add']);
        e('in','ag','shared input');e('in','down_quant','owner hidden');e('ag','quant','gathered shared hidden');
        e('router','topk','router logits');e('latent_down','dispatch','routed latent hidden');
        e('up','act','gate/up 结果');e('combine','up_quant','routed result');e('rs','add','shared result');
        for(const edge of edges)if([['router','ag'],['ag','down_quant'],['latent_down','quant'],['up','topk'],['combine','act'],['rs','up_quant']].some(([a,b])=>edge.src===a&&edge.dst===b)){edge.order=true;edge.label='本部署实际执行顺序';}
        windows.push(['Shared 的实际设备交集','920 个 MoE 样本里，所测 shared AG/latent-down、RS/latent-up、gate-up/GMM1、down/combine 的交集 min/median/max 全为 0。'],
          ['配置与设备事实分开','multistream_overlap_shared_expert=true；这些计算 kernel 实际都在 stream 425。这里只记录该部署行为，尚未定位跨流未体现的原因。']);
      }else if(family==='dense'){
        n('in','第 0 层 Dense FFN 输入',0,0,['owner [16,7168]','实际双 chunk：8 + 8 行'],'本部署采用 ChunkedMLPTP。');
        for(let i=0;i<2;i++){
          const r=i===0?2:6;
          n('ag'+i,`Input AG · chunk ${i}`,3,i+1,['owner [8,H] → [64,H]','先提交 AG0、AG1'],'全 rank 通信序列相同。','comm');
          n('quant'+i,`MX quant · chunk ${i}`,2,r,['[64,7168] → FP8 + scale','主流 425'],'只等本 chunk 输入。');
          n('up'+i,`Gate/up · chunk ${i}`,1,r+1,['[64,7168] → [64,8448]','I/P=4224 · AI_CORE'],'本 chunk 的两支中间通道配对。');
          n('act'+i,`SiTU / MX quant · chunk ${i}`,2,r+2,['真实为多笔 Slice/Cast/Tanh/Mul','再 DynamicMxQuant → [64,4224]'],'这是多个实际 kernel 的顺序组，不是单个 SituMxQuant 融合 kernel。');
          n('down'+i,`Down GEMM · chunk ${i}`,1,r+3,['[64,4224] → [64,7168]','AI_CORE'],'计算各 owner 的 hidden 部分和。');
          n('rs'+i,`Output RS · chunk ${i}`,3,r+4,['[64,7168] → [8,7168]','sum + owner rows'],'RS0 与 chunk1 gate/up 有实际交集。','comm');
          seq(['ag'+i,'quant'+i,'up'+i,'act'+i,'down'+i,'rs'+i]);
          e('in','ag'+i);
        }
        e('ag0','ag1','通信发起次序','',true);e('down0','quant1','主流先完成 chunk0 计算','',true);e('rs0','rs1','通信次序','',true);
        n('join','还原 owner chunk 顺序',0,12,['[8,H] + [8,H] → [16,H]','等 RS0 和 RS1'],'源码写回预分配输出；图不额外假造 Concat kernel。');e('rs0','join');e('rs1','join');
        frame('ag1','up0','#71dac8');frame('rs0','up1','#efc978');
        windows.push(['Dense 双 chunk 执行顺序','10 次 Dense 调用都执行 AG0→AG1→RS0→RS1；保留两个 chunk 的输入/输出依赖。'],
          ['实际交集有限','AG1/gate-up0 与 RS0/gate-up1 有正交集；首尾通信、SiTU Vector 和同 chunk 的 GEMM 依赖仍然暴露。']);
      }else if(family==='head'){
        n('in','Final RMSNorm / logits rows',0,0,['owner [16,7168]','实际分成 8 + 8 行'],'每 chunk 是本次 logits 行，最终仍等待完整请求/verify 结果。','op','ops/vocab_parallel_embedding.py');
        for(let i=0;i<2;i++){
          n('ag'+i,`Hidden AG · chunk ${i}`,3,i+1,['[8,H] → [64,H]','先提交 AG0、AG1'],'不是旧 DCP 的 vocab logits AG。','comm');
          n('mm'+i,`Head GEMM · chunk ${i}`,1,i+2,['[64,H] × [20480,H] BF16','→ [64,20480] logits'],'两次 GEMM 在主流 59 顺序执行。');
          n('a2a'+i,`Logits A2A · chunk ${i}`,3,i+3,['返回 owner [8,163840]','每个 owner 完整词表'],'A2A0 与 GEMM1 有实测交集。','comm');
          n('restore'+i,`Owner transpose · chunk ${i}`,2,i+5,['[8,8,20480] → owner vocab','实际 Transpose kernel'],'最后仍须恢复 owner 行和词表顺序。');
          seq(['ag'+i,'mm'+i,'a2a'+i,'restore'+i]);e('in','ag'+i);
        }
        n('join','Concat logits → sampler',2,7,['[8,V] + [8,V] → [16,163840]','之后才 rejection / resample'],'完整 owner logits 经实际 ConcatD 拼齐后进入 sampler。');
        e('restore0','join');e('restore1','join');e('ag0','ag1','通信次序','',true);e('mm0','mm1','同流 Cube 次序','',true);
        e('a2a0','a2a1','通信次序','',true);e('mm1','restore0','源码先提交两块 GEMM/交换','',true);
        frame('ag1','mm0','#71dac8');frame('a2a0','mm1','#efc978');
        windows.push(['Head 双 chunk 执行顺序','10 个匹配调用都有两次 hidden AG、两次 GEMM、两次 logits A2A；不是旧 DCP logits all-gather。']);
      }else{
        n('in','Pad IDs / 双 chunk 入口',0,0,['owner pad 128 IDs → 64 + 64','实际有效行随后裁回'],'这是入口 eager 区间，不将 pad capacity 与每卡有效 token 数混同。','op','ops/vocab_parallel_embedding.py');
        for(let i=0;i<2;i++){
          n('ag'+i,`IDs AG · chunk ${i}`,3,i+1,['[64] → [512] IDs','INT64 token IDs'],'先提交两块 IDs AG。','comm');
          n('lookup'+i,`Lookup / mask · chunk ${i}`,2,i*3+3,['weight [20480,7168]','→ partial [512,7168]'],'GatherV2 / MaskedFill 及 mask 计算；没有 Cube GEMM。AI_CPU launcher 不计为普通计算交集。','op','ops/vocab_parallel_embedding.py');
          n('rs'+i,`Vectors RS · chunk ${i}`,3,i*3+4,['[512,7168] → [64,7168]','AicpuKernel 路径'],'本 capture 标记 hcom_reduceScatter_AicpuKernel，不能把所有 collective 都标成 CCU kernel。','comm');
          seq(['ag'+i,'lookup'+i,'rs'+i]);e('in','ag'+i);
        }
        n('join','还原 owner rows / 裁有效行',0,9,['[64,H] + [64,H] → pad [128,H]','后续裁回有效 token rows'],'设备上有 TensorMove；CSV 不能独自区分这些入口的 target/draft 调用者。','op','ops/vocab_parallel_embedding.py');
        e('rs0','join');e('rs1','join');e('ag0','ag1','通信次序','',true);e('lookup0','lookup1','主流查表次序','',true);e('rs0','rs1','通信次序','',true);
        frame('rs0','lookup1','#71dac8');
        windows.push(['Embedding 双 chunk 执行顺序','20 个匹配的 512-row lookup 双 chunk 调用。这里按形状匹配，不冒充已由 CSV 分离 target/draft caller。'],
          ['入口 envelope 不能当带宽','首例 eager / AICPU RS 有较大提交与等待区间；只记录实际交集，不拿该通信 envelope 拟合链路带宽。']);
      }
      const stages=['输入 / 缓存（非引擎）','Cube / MIX_AIC','Vector / MIX_AIV','Communication'];
      return {nodes,edges,parallelGroups,windows,stages,resourceLanes:true,observed:true,family};
    }
    function proof(family){
      const d=data.domains[family];
      return `<div class="lp-profile-proof"><b>A5 localTP 实测 · 2026-11-13 · ${names[family]} · ${d.sample_count} 个匹配样本</b><p>部署 ${data.ascend_revision.slice(0,12)} / vLLM ${data.vllm_revision.slice(0,12)}；DP32 / TP1 / localTP8，C128，128k→1k，DSpark3 / 固定2.7。单 rank、warmup capture 的10次worker step。图的形状固定为该 capture，不随上方历史估算参数改变；依赖布局不按时间比例。</p><div class="table-scroll"><table class="data-table"><thead><tr><th>同一样本的设备窗口</th><th>交集中位 μs</th><th>min / max μs</th></tr></thead><tbody>${Object.entries(d.overlap_stats).map(([key,s])=>`<tr><td>${pairNames[key]}</td><td>${fmt(s.median_us)}</td><td>${fmt(s.min_us)} / ${fmt(s.max_us)}</td></tr>`).join('')}</tbody></table></div><p>彩框只标已观察到的正交集；实线为数据依赖，虚线箭头为同流/发起顺序。点击节点可核对首例 CSV 行号、实际 kernel 和 stream。交集不是端到端收益；与上方最新 PR 源码学习基线分别标注。</p></div>`;
    }
    return {data,names,source,graph,proof};
  }
  if(typeof module==='object'&&module.exports)module.exports={create};
  else root.KimiA5LocalTPProfile=create(root.KimiA5LocalTPProfileData);
})(typeof globalThis!=='undefined'?globalThis:this);
