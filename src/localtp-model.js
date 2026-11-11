/* A5 source-derived planning model. Payloads are per-card SEND; equal receive in a balanced group.
   Logical collective payload is not a measured wire/protocol counter. */
(function(root){
  'use strict';
  const MiB=2**20,GiB=2**30,H=7168,P=12288,V=163840,NK=69,ND=24,NS=92;
  const revision='30572def8290cf08b300a2326b09f9c4abe8f30f';
  const upstream='24420b7a341af37a9bc3186e6fc6062512e0aa58';
  const defaults={s:8,t:384,c:384,emb:8192,l:384,b:2,bw:200,hbm:1000,launch:10,level:4,family:'kda',colocal:false,schedule:'late',pipeline:false};
  const names=['取消 full-F（前置）','QKV / beta','+ KDA / DSA O','+ vocab / dense MLP','+ full-rank G'];
  const weightBase=115.5823050737381;
  const full={q:17.5111083984375+.08843994140625,o:15.2578125,vocab:4.375,dense:.697906494140625,g:15.2578125};
  function calculate(p){
    const {s,t,c,emb,l,b,bw,hbm,launch}=p;
    if(![1,2,4,8].includes(s)||![t,c,emb,l].every(x=>Number.isInteger(x)&&x>0)||c<t||emb<t||![2,4].includes(b)||bw<=0||hbm<=0||launch<0)throw Error('需要 T>0、C≥T、embedding capacity≥T；容量为整数，带宽>0。');
    const g=(s-1)/s,ag=(s-1)*c*H*2,head=g*c*P*2,rows=[];
    const add=(id,label,level,layers,send,calls,weight,read,formula,note)=>rows.push({id,label,level,layers,send,calls:s===1?0:calls,weight,read,formula,note,
      total:send*layers,commMs:send/(bw*1e9)*1000+(s===1?0:calls)*launch/1000,readMs:read*GiB/(hbm*1e9)*1000});
    add('q','KDA QKV / beta',1,NK,ag+g*c*(3*P+96)*2,2,full.q*g,full.q*g/NK,'(S−1)·C·H·2 + g·C·(3P+96)·2','BF16 输入 AG；MXFP8 QKV + BF16 beta；按完整头 A2A 还 owner。');
    add('o','KDA / DSA O',2,NK+ND,head+ag,2,full.o*g,full.o*g/(NK+ND),'g·C·P·2 + (S−1)·C·H·2','完整头输出 A2A → 每卡局部 O → partial hidden RS。');
    add('embedding','Embedding',3,1,(s-1)*emb*8+(s-1)*emb*H*2,2,2.1875*g,0,'(S−1)·E·8 + (S−1)·E·H·2','IDs 按 int64 计；固定 E 行 AG → masked lookup → BF16 RS。查表不扫描整个词表，权重常驻下降不等于同量 HBM 读下降。');
    add('head','LM head',3,1,(s-1)*l*H*2+g*l*V*b,2,2.1875*g,2.1875*g,'(S−1)·L·H·2 + g·L·V·b','采样行 hidden AG → 分词表 GEMM → logits A2A。默认 BF16 投影返回后再做采样转换；FP32 head 可切换 b=4。');
    add('dense','Dense MLP',3,1,2*ag,2,full.dense*g,full.dense*g,'2·(S−1)·C·H·2','首层；I=33792；AG → local gate_up / SiTU / down → RS。');
    add('kg','KDA G',4,NK,p.colocal?0:head,p.colocal?0:1,full.g*NK/(NK+ND)*g,full.g*g/(NK+ND),p.colocal?'0（复用 QKV AG；G 留在 O shard）':'g·C·P·2',p.colocal?'候选 B：复用 QKV input AG；G 不还 owner，改在 O 的 head shard 上做逐头 norm/gate。':'复用同一输入、同一组的 QKV AG；G 输出按头 A2A 还 owner。可与 QKV return 打包但须等两者完成。');
    add('dg','DSA G',4,ND,ag+(p.colocal?0:head),p.colocal?1:2,full.g*ND/(NK+ND)*g,full.g*g/(NK+ND),p.colocal?'(S−1)·C·H·2':'(S−1)·C·H·2 + g·C·P·2',p.colocal?'候选 B：保留独立 input AG；G 留在 O 的 head shard，在 UV 输出 A2A 后逐元素 gate。':'DSA 没有上述 QKV AG，必须新增 input AG；O 输入是另一张量，不能复用。');
    const plans=names.map((name,level)=>{const rr=rows.filter(r=>r.level<=level);const saved=rr.reduce((a,r)=>a+r.weight,0),read=rr.reduce((a,r)=>a+r.read*r.layers,0),send=rr.reduce((a,r)=>a+r.total,0),calls=rr.reduce((a,r)=>a+r.calls*r.layers,0);return {name,level,weight:weightBase-saved,saved,read,send,calls,commMs:send/(bw*1e9)*1000+calls*launch/1000,readMs:read*GiB/(hbm*1e9)*1000};});
    return {rows,plans,ag,head,g,padding:c/t,fullF:2*t*H*P,lowF:2*t*128*(H+P),embeddingBuffer:s*emb*H*2,embeddingOut:emb*H*2};
  }
  function graph(p){
    if(['embedding','head','dense'].includes(p.family))return auxiliaryGraph(p);
    const {s,c,level,family}=p,sc=s*c,ps=P/s,hs=96/s;
    const amounts=calculate(p);const mi=x=>(x/MiB).toFixed(2)+' MiB / 方向';const nodes=[],edges=[];
    const n=(id,title,col,row,lines,description,kind='op',status='已有结构',source='vllm_ascend/ops/kimi_kda.py')=>nodes.push({id,title,col,row,lines,description,kind,status,source});
    const e=(src,dst,label,shape,description='数据就绪后才能消费')=>edges.push({id:src+'-'+dst,src,dst,label,shape,description});
    let stages,windows;
    if(family==='kda'){
      stages=['Owner 输入','输入交换 / 本地 F','投影分支','Owner 预处理（独立分支）','汇合：循环计算 / 状态缓存','O 切分与输出归并'];
      n('in','AttnRes / norm 后输入',0,1,[`owner [C=${c}, H=7168]`,'DP32 · TP1 · EP32'],'当前层 owner 的 query token；图中的 C 是所有 rank 统一的 padding 行数。');
      n('ag','AG · hidden',1,0,[`[C,H] → [${sc},H]`,'BF16 · '+mi(amounts.ag)], '组内所有 owner hidden 聚合；收齐后才能做 local QKV/G。','comm','拟新增');
      n('f','F_a → F_b',1,3,[`[C,H] → [C,128] → [C,P]`,'本地输入即可算 · 可与 AG 重叠'],'撤销 full-F=F_b@F_a 预合成；69 层少驻留 11.320 GiB。本分支只处理 owner 的 token，不依赖 input AG；两次 GEMM 舍入次序需数值验证。','op','拟恢复');
      n('quant','QKV input · MX quant',2,0,[`[${sc},7168] BF16 → MXFP8+scale`,'Vector · 动态量化'],'先等 input AG 完成，再动态 MX 量化输入。它是 Vector 计算，不能与 QKV Cube GEMM混为同一个可并行节点。当前 A5 代码已把 MX quant 放到 BFG Cube GEMM 的并行支路；本图拟沿用该硬件互补机制。','op','Vector · 拟接线');
      n('bp','beta BF16 投影',2,1,[`[${sc},H] → [${sc},${hs}]`,'Cube · 与 QKV/G GEMM 排队'],'beta 的 BF16 权重与 MXFP8 QKV 分开投影；它与 QKV、G 共同使用 Cube 队列，本图不把多个 Cube GEMM 框成并行。','op','Cube · 拟适配');
      n('q','QKV · MXFP8 GEMM',2,2,[`[${sc},H] → QKV: 3×${ps}`,'Cube · 等 quant 与 Cube 队列'],'MXFP8 QKV GEMM 使用量化后的 input；其前置动态量化另列为 Vector 节点。beta 单独 BF16 Cube 投影，不与 QKV/G 同时占用 Cube 作为掩盖窗口。','op','Cube · 拟适配');
      n('return','A2A · QKV / beta',3,0,[`[${sc},heads/S] → [C,96 heads]`,mi(amounts.g*c*(3*P+96)*2)],'conv/state 继续归 token owner 管理。beta 是每头一个标量；不是对 head_dim 再切分。','comm','拟新增');
      n('conv','① causal conv1d · Q/K/V',3,1,['只等 QKV return','Vector · 读写 conv_state'],'QKV A2A 完成后启动 conv，含 SiLU 与短窗口缓存更新。与 beta sigmoid 没有数据依赖，但两者都占 Vector；拟放同一 Vector 队列依次执行，不以 Vec/Vec 多流作为掩盖窗口。A5 decode 的 Q/K L2 normalize 在 recurrent kernel 内，不在此单独执行。','op','Vector · 不框 Vec/Vec 并行');
      n('beta','② beta.float → sigmoid',3,2,['只等 beta return','FP32 [1,C,96] · 向量计算'],'beta 与 QKV 在同一次 A2A 返回后分支；转换到 FP32 并 sigmoid。不需要等待 conv 或 F 的数据，但它与 conv 都占 Vector，资源调度采用串行；不框成两种引擎互补。当前混合 BFG 路径已有该数学计算，localTP 中需要改为 owner 返回后调度。','op','Vector · owner 端串行处理');
      n('fp','F reshape · 通常仅 view',3,3,['连续 F_b 输出 → [1,C,96,128]','没有独立 kernel 时不计掩盖量'],'恢复低秩 F 后，F_b 的连续输出重排通常只需 view。原 BFG 切片的 contiguous 不能直接照搬为这里必有的 Vector copy。只有实际产生 copy kernel 时才计入搬运，不能用零工作 reshape 掩盖通信。F decay 在 recurrent kernel 内。','op','View · 不框作计算窗口');
      n('state','历史缓存 · recurrent_state',4,2,['每 slot [96,V128,K128] FP32','6 MiB / 层 / slot · 不是算子'],'保存本层本请求已有 token 的循环记忆矩阵，不是一个独立计算阶段，也不是逐 token KV 列表。recurrent core 读取旧状态，结合当前 Q/K/V、F、beta 计算输出并原地写回新状态；下一步继续使用。state_v_first=True，inplace_final_state=True。spec verify 的 state slot 按实际 metadata/接受关系索引；localTP 不缩小该缓存。','cache','缓存：读旧值 / 写新值','vllm_ascend/ops/kda.py');
      n('core','fused recurrent KDA',4,1,['等 conv + sigmoid β + raw F + state','融合 QK L2 / decay / 状态更新'],'decode/verify 核心循环算子：必须等 conv 后的 Q/K/V、预处理 beta、本地 raw_gate F 和已有状态就绪。内部融合 Q/K L2 normalize、F decay gate、循环状态更新及输出计算；不是与状态更新平行的另一阶段。A5 已安装 fla_npu 的 recurrent_kda.cpp 明确声明 KERNEL_TYPE_AIV_ONLY；本实现不占 Cube 做矩阵乘。读取并原地更新 recurrent_state。G 不是此算子的输入，不用等待 G。','op','Vector · AIV_ONLY 已核对','vllm_ascend/ops/kda.py');
      n('gate','RMSNorm + G gate',4,4,['等 recurrent 输出 + G','[C,96,128] → [C,P]'],'core 与 G return 两者都完成后才能输出；G 分支和 core 无相互数据依赖，可以重叠。');
      if(level>=4){n('g','local full-rank G',2,4,[`[${sc},H] → [${sc},${ps}]`,'复用 QKV input AG'],'G 按输出 head 列切分权重 [P/S,H]；需等 input AG，但不用等 conv、beta、F 或 recurrent core。QKV 与 G GEMM 都占 Cube，不能只凭两条分支就断言两者并发会更快。','op','Cube · 拟新增');n('gr','A2A · G',3,4,[`[${sc},P/S] → [C,P]`,mi(amounts.head)+' · 独立返回'],'可与 owner conv/beta/core 计算重叠。与 QKV A2A 虽无数据依赖，但同一通信组/链路可能排队，不能认定两次 collective 能并发；跨 rank 必须保持一致调用顺序。当前通信表按独立返回计。','comm','拟新增');e('ag','g','复用 AG',`[${sc},H]`);e('g','gr','局部 G',`[${sc},${ps}]`);e('gr','gate','owner G','[C,P]');}
      else{n('g','owner G',2,4,['BF16 [P,H] · 不切','[C,H] → [C,P]'],'本档 G 保持 owner 本地；输入就绪即可启动，与 AG、conv、beta/core 无数据依赖，但与其他 GEMM 争用 Cube。');e('in','g','hidden','[C,H]');e('g','gate','owner G','[C,P]');}
      e('in','ag','hidden','[C,H]');e('in','f','hidden','[C,H]');e('ag','quant','gathered hidden',`[${sc},H]`);e('quant','q','量化 input','MXFP8 + scale');e('ag','bp','gathered hidden',`[${sc},H]`);e('q','return','局部 QKV 完整头',`[${sc},${hs},128] ×3`);e('bp','return','局部 beta',`[${sc},${hs}] BF16`);e('return','conv','分支①：QKV owner','[C,96,128] ×3');e('return','beta','分支②：beta owner','[C,96] BF16');e('f','fp','分支③：local F','[C,12288]');e('fp','core','raw F gate','[1,C,96,128]');e('beta','core','sigmoid beta','[1,C,96] FP32');e('conv','core','conv 后 QKV','[1,C,96,128] ×3');e('state','core','读旧状态 / 原地写回新状态','[slots,96,128,128] FP32','缓存读写关联，不是额外算子。core 内部读取旧值并写新值；下一 decode 步读取更新后的状态。');edges.at(-1).cache=true;e('core','gate','attention out','[C,96,128]');
      output('gate');
      windows=[['框 P1：Communication AG ↔ Cube F','输入只读且两条支路不互相等待；需要异步通信与独立计算流。属于 localTP 拟新增接线，未给出实测隐藏比例。'],[p.schedule==='early'?'框 P2：Vector MX quant ↔ Cube G':'框 P2：Communication QKV return ↔ Cube G',p.schedule==='early'?'AG 就绪后，先发 G Cube 与 QKV input quant Vector；G/β/QKV 仍在同一 Cube 队列依次执行。这个 G 不能再拿去重复抵消后续 A2A。':'先完成 beta/QKV Cube 投影；发异步 QKV A2A 时才启动 G Cube。G 与 QKV GEMM 不并行。'],['不框：conv / beta sigmoid / recurrent','A5 conv 只启动 Vector，recurrent 声明 AIV_ONLY；这些 Vector 工作按队列执行。两个 Vector 流不等于 Cube/Vector 互补。'],['不框：reshape / 状态缓存','连续 F_b 输出的 reshape 通常为 view；recurrent_state 是数据缓冲，读写在 recurrent kernel 内，不是独立计算。'],['条件窗口：G tail / G return ↔ Vector','G 若在 QKV A2A 结束后仍有 Cube 尾部，可与 owner Vector 重叠，但窗口必须看时长。G A2A 与 Vector 是否互补还取决于 HCCL 是否占 AIV；当前新拓扑算法未测，不框成已确定收益。'],['必须等待：输入 join / 输出 join','core 等 conv、beta、F 和旧 state；gate 等 core 与 G。O A2A→O GEMM→RS 是同 chunk 串行依赖，不能框成多流掩盖。']];
    }else if(family==='dsa'){
      const src='vllm_ascend/attention/sfa_v1.py';
      stages=['Owner 输入','MLA prolog / G 输入交换','Indexer 准备与选择','Sparse attention / G 返回','UV 与 gate','O 切分与输出归并'];
      n('in','AttnRes / norm 后输入',0,1,['owner [C,H]','TP1 · DCP1 · PCP1'],'DSA query/cache 保持 owner；新方案是 pure DP，不把 DCP query gather / TopK merge / output combine 搬过来。','op','输入',src);
      n('prolog','MLA prolog（含 UK Q 变换）',1,0,['Q_a / KV_a / norm / Q_b / UK','MIX_AIC · 同时用 Cube / Vector'],'本地 DCP trace 实际走 MlaPrologV3，而非图上多段独立 GEMM。当前源码 fused/native 分支需按新 TP1 验证。prolog 生成 compressed query、q latent、写 main KV；UK/UV 替代已 dispose 的 KV_b，不重复计权重。整段 prolog 不标成纯 Vector，不能与 G Cube 整段视作互补。','op','MIX_AIC · TP1 适配待验证',src);
      n('prep','Indexer Q / K / weights 准备',2,0,['Cube 投影 + Vector key norm / quant','按资源排队；不框 G Cube 并行'],'DCP 实测：indexer query 投影在辅助流，key norm / quant / cache write 在主流，部分 Cube/Vector 交叠；但整条准备管线同时有 Cube 与 Vector。G GEMM 必须与其 Cube 投影排队，不能把这整段任意当纯 Vector 窗口。','op','Cube + Vector 混合管线',src);
      n('idx','Quant Lightning Indexer',2,1,['Q/K/weights + index cache → top-k','MIX_AIC · 不是纯 Vector TopK'],'DCP 样本 VllmAscendQuantLightningIndexerV2 的 Accelerator Core 为 MIX_AIC。pure DP 不需要 DCP 候选 all-gather 与 merge；其历史空隙不能当新 G 的计算窗口。','op','MIX_AIC',src);
      n('cache','owner compressed KV cache',2,2,['latent 512 + rope / quant scale','缓存数据 · 不除 localTP'],'prolog 写入当前 token，稀疏 attention 读取历史与新 KV；本框是数据，不是计算。','cache','缓存',src);
      n('attn','Quant Sparse Attention',3,0,['UK Q + owner KV + top-k','MIX_AIC · 输出 [C,96,512]'],'DCP trace KvQuantSparseFlashAttentionVllm 为 MIX_AIC。pure DP 取消 DCP output/LSE all-to-all 与 combine；此处不是一个纯 Vector 引擎窗口。','op','MIX_AIC',src);
      n('uv','UV up projection',4,0,['[C,96,512] → [C,96,128]','Cube · 不与 G Cube 叠算'],'UV 是 TransposeBatchMatMul / Cube 路径；G GEMM 与 UV 共享 Cube 队列。','op','Cube',src);
      n('gate','sigmoid(G) × attention output',4,2,['等 UV 输出 + G','Vector · owner [C,P]'],'原方案 A 的 join：UV 与返回的完整 G 都完成后才能 gate。','op','Vector',src);
      if(level>=4){
        n('ag','AG · G input',1,3,[`[C,H] → [${sc},H]`,mi(amounts.ag)+' · Communication'],'拟在层输入 ready 后异步发 G input AG，与 owner MLA prolog 的 MIX_AIC 工作交叠。DCP trace 已显示 CCU/UBDMA 通信可和 Indexer 混合计算交叠，但新 DP group、消息尺寸和带宽仍需验证。','comm','拟：Communication',src);
        n('g','local full-rank G',2,3,[`[${sc},H] → [${sc},${ps}]`,'Cube · 与 owner Cube 工作排队'],'G 不自动与 prolog / indexer / sparse attention 互补，因为这些算子也用 Cube。A 中需排在相关 owner Cube 工作之间，再发 G return；G 计算自身可能增加关键路径，不把这部分说成全隐藏。B 可把 G 推迟到 owner UV ready 后，与 O heads A2A 配对。','op','拟：Cube',src);
        n('gr','A2A · G',3,3,[`[${sc},P/S] → [C,P]`,mi(amounts.head)+' · Communication'],'拟在 G ready 后发异步 return，让独立 Indexer/SFA 混合计算继续。Comm+计算窗口有 DCP 设备重叠证据支持，但不把 G GEMM 本身算进去；新拓扑协议仍待测。','comm','拟：Communication',src);
        e('in','ag','owner hidden','[C,H]');e('ag','g','gathered G input',`[${sc},H]`);e('g','gr','G shard',`[${sc},${ps}]`);e('gr','gate','owner G','[C,P]');
      }else{n('g','owner G',2,3,['[C,H] → [C,P]','Cube · 不与 MIX_AIC 直接叠算'],'当前源码 G 在 attention / UV 后计算。提前属于调度改造，需与其他 Cube 工作排队，不框成已实现的收益。','op','Cube · 拟提前',src);e('in','g','hidden','[C,H]');e('g','gate','G','[C,P]');}
      e('in','prolog','layer input','[C,H]');e('in','prep','raw hidden 给 index key / weights','[C,H]');e('prolog','prep','index query latent','[C,1536]');e('prep','idx','index Q/K/weights','量化 cache + query');e('prolog','cache','写 compressed KV','[C,512+64]');e('prolog','attn','absorbed query','[C,96,512+64]');e('cache','attn','KV read','owner KV');e('idx','attn','top-k indices','[C,1,top-k]');e('attn','uv','latent attention output','[C,96,512]');e('uv','gate','value heads','[C,96,128]');
      output('gate');
      windows=[['DCP 实测：query AG ↔ Indexer 准备','504 个 steady DSA 样本中，query AG 中位58.56 μs，设备计算区间交集中位57.96 μs。包含辅助流 Cube 与主流 Vector / MIX 工作，不等于省掉同量墙钟时间。'],['拟 D1：G input AG ↔ owner prolog','Communication 与 MIX_AIC 的设备交叠在旧 DCP 样本已有证据；新 AG group / payload / async event 接线需要实现验证。'],['G Cube 不与整段 DSA 核心互框','prolog、量化 Indexer、Sparse Attention 均有 Cube；UV 也是 Cube。G 必须排队，或寻找确实存在的 Vector 尾窗。'],['方案 A：只隐藏 G return 通信','可把已 ready 的 G return 与随后 Indexer/SFA 计算配对；前置 G GEMM 仍可能增加关键路径，不能将两者合并称作全隐藏。'],['方案 B：O heads A2A ↔ Cube G','G 留在 O shard 后，将 G GEMM 推迟到 UV ready / heads A2A 发起时，再在 local gate join。需保留 gathered input buffer 至此。'],['DCP 空档不可搬到 pure DP','旧样本 TopK 候选 AG 约69.07 μs、output A2A 约179.45 μs，交集均0；这两类通信在 pure DP 消失，不能拿来抵新 G 的成本。']];
    }
    function output(src){
      const col=5;
      if(level>=2){n('oa','A2A · output heads',col,0,[`[C,P] → [${sc},${ps}]`,mi(amounts.head)],'输入已经完成 KDA gated norm 或 DSA UV+G；随后才分发给 O 权重 shard。','comm','拟新增');n('o','local O GEMM',col,1,[`[${sc},P/S] × [P/S,H]` ,`partial hidden [${sc},H]`],'O 按输入 head 维度切分；计算各 owner 对应 hidden 的 partial sum。','op','拟新增');n('rs','RS · hidden sum',col,2,[`[${sc},H] → [C,H]`,mi(amounts.ag)+' · 尾部等待'],'不同 head partial hidden 求和并返回 owner；同 chunk 无法绕过该依赖。','comm','拟新增');e(src,'oa','完整 owner heads','[C,P]');e('oa','o','local head rows',`[${sc},${ps}]`);e('o','rs','partial hidden',`[${sc},H]`);}
      else{n('o','owner O GEMM',col,1,['[C,P] × [P,H]','owner [C,H]'],'未启用 O 切分时，不增加 O collective。');e(src,'o','full heads','[C,P]');}
      n('out','AttnRes → FFN',col,4,['owner [C,H]','同 chunk 等 O 尾部完成'],'FFN 接收 attention 残差混合后的 hidden，不是直接复用 attention 输入。');e(level>=2?'rs':'o','out','hidden ready','[C,H]');
    }
    if(p.colocal&&level>=4){
      // Norm/gate operates within a complete head. G and O must share the exact group,
      // rank/head ordering, owner-row padding and activation dtype. This is a proposal.
      nodes.splice(nodes.findIndex(n=>n.id==='gr'),1);
      const producer=family==='kda'?'core':'uv';
      for(let i=edges.length-1;i>=0;i--){const x=edges[i];if(x.src==='gr'||x.dst==='gr'||(x.src===producer&&x.dst==='gate')||(x.src==='gate'&&x.dst==='oa')||(x.src==='oa'&&x.dst==='o'))edges.splice(i,1);}
      const gate=nodes.find(n=>n.id==='gate');gate.col=5;gate.row=1;gate.title=family==='kda'?'local RMSNorm + G gate':'local sigmoid(G) × UV output';gate.lines=[`wait heads A2A + local G`,`[${sc},${hs},128] → [${sc},P/S]`];gate.kind='op';gate.status='候选 B · 拟移到 O shard';gate.description='KDA norm 只归一化 head_dim=128，DSA gate 逐元素；可移到相同完整头 shard。需要拆开当前 KDA core/norm 接口并保持 graph padding、dtype、行次序与原始舍入契约。';
      nodes.find(n=>n.id==='o').row=2;nodes.find(n=>n.id==='rs').row=3;
      nodes.find(n=>n.id==='oa').description='候选 B 在 owner core（KDA）或 UV（DSA）后立刻 A2A 完整头，随后在 O shard 上与 local G 汇合；发送字节不变。';
      if(family==='dsa')stages[4]='UV 输出（G 留在 shard）';
      e(producer,'oa','未 gate 的完整头','[C,P]');e('oa','gate','sharded attention heads',`[${sc},${ps}]`);e('g','gate','local G（无需返回）',`[${sc},${ps}]`);e('gate','o','gated local heads',`[${sc},${ps}]`);
      windows=windows.map(([a,b])=>[a.replaceAll('G return','local G').replaceAll('G 分支','local G 分支'),b.replaceAll('G return','local G')]);
      windows.splice(1,0,['候选 B：G 留在 O shard','省去每层一次 G return；attention heads A2A 与 local G 并行，最终在 O shard 的 gate join。不是额外算量削减。']);
    }
    if(family==='kda'){
      // Resource lanes replace the previous numbered logical-stage columns.
      for(let i=nodes.length-1;i>=0;i--)if(['beta','fp'].includes(nodes[i].id))nodes.splice(i,1);
      for(let i=edges.length-1;i>=0;i--)if(['beta','fp'].includes(edges[i].src)||['beta','fp'].includes(edges[i].dst))edges.splice(i,1);
      e('f','core','本地 raw F（连续输出仅 view）','[1,C,96,128]');
      const prep=nodes.find(n=>n.id==='conv');prep.title='conv → beta.float / sigmoid';prep.lines=['Vector 同队列依次执行','独立算子串行，不是融合 kernel'];prep.status='Vector · 串行预处理';prep.description='QKV/beta A2A ready 后，conv 与 beta 后处理在 Vector 队列顺序执行，不把两个 Vector 支路当互补。conv 读写 conv_state，beta float/sigmoid 输出 FP32；两者都完成后才发 recurrent。';
      const ret=edges.find(e=>e.src==='return'&&e.dst==='conv');ret.label='owner QKV + raw beta';ret.shape='Q/K/V + [C,96] beta';
      const coreIn=edges.find(e=>e.src==='conv'&&e.dst==='core');coreIn.label='conv QKV + sigmoid beta';coreIn.shape='[1,C,96,128] ×3 + [1,C,96]';
      nodes.find(n=>n.id==='core').lines=['等 conv / sigmoid β / F / 旧 state','Vector · 融合 L2 / decay / state'];
      nodes.find(n=>n.id==='bp').description+=' 拟把这段 beta BF16 Cube 主计算和 QKV input MX quant Vector 放在同一发起窗口；beta 比当前 BFG 小，不能沿用 BFG 的时间宣称量化被完全覆盖。';
      if(level>=4&&!p.colocal)nodes.find(n=>n.id==='gr').description='G GEMM ready 且通信队列允许后尽早发异步 A2A；与只依赖 owner QKV/F/beta 的 RecurrentKDA 在数据上独立，最终 gate 等两者。11/5 DCP A2A 的相同 connection_id 对应 CCU_LAUNCH，支持 Comm/Vector 资源互补判断；新 G shape/group 需验证 CCU 路径和真实时间交集。若 G 提前返回，不能重复算作 recurrent 的掩盖收益。';
      const positions={in:[0,0],f:[1,1],ag:[3,1],bp:[1,2],quant:[2,2],q:[1,3],g:[1,4],return:[3,4],conv:[2,5],core:[2,6],state:[0,6],gr:[3,6],gate:[2,7],oa:[3,8],o:[1,9],rs:[3,10],out:[0,11]};
      if(p.colocal&&level>=4){positions.gate=[2,9];positions.o=[1,10];positions.rs=[3,11];positions.out=[0,12];}
      for(const node of nodes){const at=positions[node.id];node.col=at[0];node.row=at[1];}
      stages=['输入 / 缓存（非引擎）','Cube','Vector','Communication'];
      windows=[['框 C1：Comm input AG ↔ Cube F','两者只读 owner hidden，F 不切分。异步发 AG 的同时算本地 F_a/F_b；最终 QKV 投影需要 AG，recurrent 需要 F。'],['框 C2：Vector MX quant ↔ Cube beta BF16','按用户补充固定为这组 CV 窗口。AG ready 后并发发起，QKV GEMM 等量化与自己的 Cube 队列。beta 很小，实际可覆盖量不能套用原 BFG 时长。'],['框 C3：Comm QKV/beta A2A ↔ Cube G','QKV 投影完成后，同一组框内并发启动 A2A 返回和独立 G GEMM。两个 Cube GEMM仍串行；不是 G 与 QKV GEMM 叠算。'],['框 C4：Comm G A2A ↔ Vector recurrent','两者在最终 norm/gate join；G return ready 后可尽早发。若它已在 conv 阶段结束，就不能再声称 recurrent 隐藏了同一次通信；图不是按比例的实测时间线。A5 DCP A2A 有 CCU_LAUNCH 证据，新消息需验证同样的推进路径。'],['不框：Vector 内部与状态缓存','conv→beta 后处理→recurrent 按 Vector 队列处理。连续 F reshape 是 view；recurrent_state 是历史数据，读写发生在 recurrent 内，不是额外计算。'],['O 尾部保持依赖','同 chunk 的 O heads A2A→O GEMM→RS 串行，最终输出必须等 RS。候选 B 省掉 G return，对应 C4 框自然消失。']];
    }
    const parallelGroups=family==='kda'?[
      {id:'C1',title:'C1 · Cube F ∥ Comm input AG（拟）',color:'#71dac8',parts:[['f','ag']]},
      {id:'C2',title:'C2 · Cube beta BF16 ∥ Vector MX quant（拟）',color:'#efc978',parts:[['bp','quant']]},
      {id:'C3',title:'C3 · Cube G ∥ Comm QKV/beta A2A（拟）',color:'#9fc5fb',parts:[['g','return']]},
      ...(level>=4&&!p.colocal?[{id:'C4',title:'C4 · Vector recurrent ∥ Comm G A2A（拟）',color:'#caaff9',parts:[['core','gr']]}]:[])
    ]:[];
    if(family==='dsa'){
      const engines={in:'Input',prolog:'Mixed',prep:'Mixed',idx:'Mixed',cache:'Cache',attn:'Mixed',uv:'Cube',ag:'Communication',g:'Cube',gr:'Communication',gate:'Vector',oa:'Communication',o:'Cube',rs:'Communication'};
      for(const node of nodes)node.engine=engines[node.id]||'Pipeline';
      if(level>=4){
        parallelGroups.push({id:'D1',title:'D1 · Comm G input（拟）',color:'#71dac8',parts:[['ag']]},{id:'D1',title:'D1 · owner MIX prolog（拟）',color:'#71dac8',parts:[['prolog']]});
        if(p.colocal)parallelGroups.push({id:'D2',title:'D2 · Comm O heads（拟）',color:'#efc978',parts:[['oa']]},{id:'D2',title:'D2 · Cube G（同色配对）',color:'#efc978',parts:[['g']]});
        else parallelGroups.push({id:'D2',title:'D2 · Comm G return（拟）',color:'#efc978',parts:[['gr']]},{id:'D2',title:'D2 · owner MIX Indexer（拟）',color:'#efc978',parts:[['idx']]});
      }
    }
    if(family==='kda'){
      const engines={in:'Input',ag:'Communication',f:'Cube',quant:'Vector',bp:'Cube',q:'Cube',return:'Communication',conv:'Vector',beta:'Vector',fp:'View',state:'Cache',core:'Vector',gate:'Vector',g:'Cube',gr:'Communication',oa:'Communication',o:'Cube',rs:'Communication'};
      for(const node of nodes)node.engine=engines[node.id]||'Pipeline';
    }
    return {nodes,edges,stages,windows,parallelGroups,resourceLanes:family==='kda'};
  }
  function auxiliaryGraph(p){
    const {s,c,emb,l,family}=p,sc=s*c,a=calculate(p),nodes=[],edges=[],parallelGroups=[];
    const src=family==='dense'?'vllm_ascend/models/kimi_k3/kimi_k3.py':'vllm_ascend/ops/vocab_parallel_embedding.py';
    const n=(id,title,col,row,lines,engine,description)=>nodes.push({id,title,col,row,lines,engine,description,kind:engine==='Communication'?'comm':'op',status:engine+' · 拟 localTP',source:src});
    const e=(src,dst,label,shape)=>edges.push({id:src+'-'+dst,src,dst,label,shape,description:'同一 chunk 的消费者必须等输入完成；跨 chunk 重叠另用配对框标注。'});
    let stages,windows;
    if(family==='embedding'){
      stages=['模型入口','IDs 交换','分词表查表与 mask','返回 owner','进入第 0 层'];
      n('in','本次 input IDs',0,1,[`owner 有效 T=${p.t} 行`,`固定 exchange capacity E=${emb}`],'Input','位于主模型入口，尚未得到 embedding，不能启动依赖它的第一层计算。');
      n('ids','AG · padded token IDs',1,1,[`[E] → [${s*emb}] int64`,'Communication · IDs 很小'],'Communication','先收齐所有 owner 的 IDs 才能对这批完整 rows 查表。当前 fine-grained 路径 E=max(potential_max_tokens,max_num_batched_tokens)。');
      n('lookup','masked vocabulary lookup',2,1,[`local weight [${V/s},7168]`,`output [${s*emb},7168] BF16`],'Vector','查表与越界 mask 属 Vector/搬运；没有 Cube GEMM。词表权重省内存不等于少读一整张词表。');
      n('rs','RS · embedding vectors',3,1,[`[${s*emb},H] → [E,H]`,`发送 ${((s-1)*emb*H*2/MiB).toFixed(2)} MiB / 方向`],'Communication','收齐全词表 shard 的和后才得到 owner 的 embedding。当前整块调用没有本批独立 Cube 分支。');
      n('out','裁 pad → 首层 AttnRes',4,1,['[E,H] → [T,H]','等待 RS 后才能继续'],'Vector','这是主干的模型入口串行依赖。跨 batch/草稿模型的独立计算需要额外调度，不能默认存在。');
      e('in','ids','pad IDs','[E]');e('ids','lookup','gathered IDs',`[${s*emb}]`);e('lookup','rs','masked vectors',`[${s*emb},7168]`);e('rs','out','owner embedding','[E,H]');
      windows=[['DCP 实测：主模型查表 → all-reduce','本地 11/5 TP8/DCP8 的主模型查表权重 [20480,7168]，随后 all-reduce envelope 与设备计算交集为0；不是已经有隐藏窗口。'],['不框 Cube/Comm：这里没有 Cube 算量','lookup / mask 是 Vector 工作。想用 RS 与下一块 lookup 交叠，需分块、双 buffer，并确认具体 HCCL RS 不争用 AIV；当前没有此证据，不框收益。'],['先解决容量浪费，再谈掩盖','E8192 与 E384 的向量 payload 相差21.33倍；缩小 E 必须覆盖所有允许输入/graph pad，不能截断 prefill。入口等待无法靠给后续层多开流绕过。']];
    }else if(family==='head'){
      stages=['模型末端','hidden 行交换','分词表投影','logits 返回 owner','采样 / verify'];
      const chunks=p.pipeline?['a','b']:['a'];
      for(const [i,id] of chunks.entries()){
        const row=i*2,label=p.pipeline?'chunk '+id.toUpperCase():'整批 L='+l;
        n('in'+id,label+' · final hidden',0,row,[p.pipeline?'按完整 request / verify 行组分块':'last norm 后 [L,H]','前层完成后就绪'],'Input','LM head 在模型尾部；不能与仍在产生同一行 hidden 的最后一层随意重叠。');
        n('ag'+id,'AG · hidden '+(p.pipeline?id.toUpperCase():''),1,row,['[chunk,H] → [S·chunk,H]','Communication'],'Communication','fine-grained head 的 owner hidden 聚合；同一 chunk 的 GEMM 必须等自己的 AG。');
        n('mm'+id,'LM head GEMM '+(p.pipeline?id.toUpperCase():''),2,row,[`weight [${V/s},7168] BF16`,'Cube · 输出分词表 logits'],'Cube','每个 owner 只需要自己的最终完整 vocab logits；不同 row chunk 在投影时无依赖。');
        n('a2a'+id,'A2A · logits '+(p.pipeline?id.toUpperCase():''),3,row,['[S·chunk,V/S] → [chunk,V]','Communication'],'Communication','同一 chunk 的全 vocab 拼齐后才能做完整采样/verify。新路径是 hidden AG + logits A2A，旧 DCP 路径是 logits vocab AG，不能混算。');
        n('sample'+id,'sampling / rejection '+(p.pipeline?id.toUpperCase():''),4,row,['完整 vocab / verify 行组','Vector · 最终仍须汇合'],'Vector','MTP acceptance 需要完整的请求行组与相应 metadata；不能把单个 verify token 的中间结果当作整请求已完成。');
        e('in'+id,'ag'+id,'hidden ready','[chunk,H]');e('ag'+id,'mm'+id,'gathered rows','[S·chunk,H]');e('mm'+id,'a2a'+id,'local logits','[S·chunk,V/S]');e('a2a'+id,'sample'+id,'owner full logits','[chunk,V]');
      }
      if(p.pipeline){
        parallelGroups.push({id:'H1',title:'H1 · Cube GEMM A（拟）',color:'#71dac8',parts:[['mma']]},{id:'H1',title:'H1 · Comm AG B（拟）',color:'#71dac8',parts:[['agb']]},{id:'H2',title:'H2 · Comm logits A（拟）',color:'#efc978',parts:[['a2aa']]},{id:'H2',title:'H2 · Cube GEMM B（拟）',color:'#efc978',parts:[['mmb']]});
        n('join','采样 / 接受 metadata 汇合',4,1,['等 chunk A + B 的完整请求结果','再进入下一次模型 / draft 调度'],'Sync','这是跨 chunk 的最终同步点。需要改造 sampler 以按完整请求组处理，并按 owner 行顺序还原结果；不声称现有整批 sampler 已支持该调度。');e('samplea','join','A 结果','request groups');e('sampleb','join','B 结果','request groups');
      }
      windows=[['模型出口：整批实现没有独立分支','旧 DCP 实测 GEMM 中位186.45 μs → logits AG中位152.42 μs → transpose → rejection；AG期间计算交集0。新 hidden AG→GEMM→A2A 同 chunk 也是串行。'],['分块候选：A 投影 ↔ B hidden AG','开启分块图才框。每个 chunk 自己的 AG→GEMM→A2A 依赖保留，利用不同行块的 Cube/Communication 互补；是待实现方案，不是当前代码已有结果。'],['分块候选：A logits A2A ↔ B 投影','需双 buffer、异步 collective、全 rank 一致的调用顺序。AG B 与 A2A A 共用通信资源，不能再画成两次通信同时执行。首尾仍有 fill/drain，不能完全隐藏所有通信。']];
    }else{
      stages=['第0层 FFN 输入','输入量化','gate_up 投影','激活与 down','返回 owner','下一残差入口'];
      const chunks=p.pipeline?['a','b']:['a'];
      for(const [i,id] of chunks.entries()){
        const row=i*3;
        n('ag'+id,'AG · dense hidden '+(p.pipeline?id.toUpperCase():''),0,row,['owner [chunk,H] → [S·chunk,H]','Communication · 等 attention/AttnRes'],'Communication','首层 dense 没有 routed experts 独立支路；不能套用 shared MLP 与 routed MoE 的掩盖。');
        n('quant'+id,'Dynamic MX quant',1,row,['BF16 input → MXFP8 + scale','Vector'],'Vector','必须等该 chunk hidden 就绪；不把它与同一输入 AG 当作可直接并行。');
        n('up'+id,'gate_up GEMM '+(p.pipeline?id.toUpperCase():''),2,row,[`I/S=${33792/s}, 2I/S=${67584/s}`,'Cube · MXFP8'],'Cube','gate 和 up 在同一个 intermediate shard；只有不同行块可以独立投影。');
        n('silu'+id,'SiTU + MX quant',3,row,['gate/up → intermediate','Vector'],'Vector','同 chunk 的 SiTU 依赖 gate_up 完成。');
        n('down'+id,'down GEMM '+(p.pipeline?id.toUpperCase():''),3,row+1,['[S·chunk,I/S] → [S·chunk,H]','Cube · partial sum'],'Cube','必须等待同 chunk SiTU；不能把 gate_up/down 两个 Cube GEMM 圈成并发。');
        n('rs'+id,'RS · dense output '+(p.pipeline?id.toUpperCase():''),4,row+1,['[S·chunk,H] → [chunk,H]','Communication'],'Communication','RS 必须等自己的 down output；下一残差依赖这个归约结果。');
        n('out'+id,'下一 AttnRes '+(p.pipeline?id.toUpperCase():''),5,row+1,['owner [chunk,H]','Vector · 不省略 RS barrier'],'Vector','跨 chunk 图的两个输出最终仍须按 owner token 次序合并，后续 KDA state/metadata 必须正确。');
        e('ag'+id,'quant'+id,'hidden ready','[S·chunk,H]');e('quant'+id,'up'+id,'quantized input','MXFP8 + scale');e('up'+id,'silu'+id,'gate / up','[S·chunk,2I/S]');e('silu'+id,'down'+id,'intermediate','[S·chunk,I/S]');e('down'+id,'rs'+id,'partial output','[S·chunk,H]');e('rs'+id,'out'+id,'owner result','[chunk,H]');
      }
      if(p.pipeline){
        parallelGroups.push({id:'F1',title:'F1 · Cube up A（拟）',color:'#71dac8',parts:[['upa']]},{id:'F1',title:'F1 · Comm AG B（拟）',color:'#71dac8',parts:[['agb']]},{id:'F2',title:'F2 · Comm RS A（拟）',color:'#efc978',parts:[['rsa']]},{id:'F2',title:'F2 · Cube up B（拟）',color:'#efc978',parts:[['upb']]});
        for(const id of ['a','b'])nodes.find(n=>n.id==='out'+id).title='owner chunk '+id.toUpperCase()+' 结果';
        n('join','输出汇合 → 下一 AttnRes',5,2,['等 A / B，恢复 owner token 次序','沿用完整 recurrent metadata'],'Sync','各 chunk 可写预分配结果的不同切片，最后 event join，不一定新增 cat copy。随后再以完整 batch metadata 进入下一层，不把下层 KDA 状态依赖省掉。');e('outa','join','A ready','owner rows A');e('outb','join','B ready','owner rows B');
      }
      windows=[['首层 FFN：当前 DCP 是串行链','样本中 input AG中位15.45 μs，gate_up42.85 μs，down20.54 μs，output RS29.63 μs；两端通信与计算交集均0。'],['分块候选：Cube A ↔ Communication B','只有不同 chunk 的输入 AG、已完成 chunk 的输出 RS，可以与另一个 chunk 的 Cube GEMM 配对。整批路径不框。分块会增加小 M GEMM、collective launch 和 buffer 成本，必须实测净收益。'],['资源与依赖都要保留','量化 / SiTU 都占 Vector，up/down 都占 Cube；不能互相套框。通信序列例如 AG A→AG B→RS A→RS B，所有 rank 必须一致；只框 Cube/Comm 的交叠，不框两次 collective 并发。']];
    }
    return {nodes,edges,stages,windows,parallelGroups};
  }
  const api={MiB,GiB,H,P,V,NK,ND,NS,revision,upstream,defaults,names,calculate,graph};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;root.KimiLocalTP=api;
})(typeof window==='undefined'?globalThis:window);
