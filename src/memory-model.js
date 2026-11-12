/* K3.1 mixed-cache physical accounting. Integer bytes; all quantities per rank. */
(function(root){
  'use strict';
  const GiB=2**30, MiB=2**20;
  const NK=69, ND=24, DRAFT=5, H=7168, I=6144, HEADS=96, DIM=128, SPEC=3;
  const BLOCK=12288, MAIN=656, CONV=(4-1+SPEC)*3*HEADS*DIM*2;
  const STATE=HEADS*DIM*DIM*4+CONV, PAGE=BLOCK*MAIN;
  function defaults(platform='a3', evidence){
    const dp=platform==='a3'?64:32;
    const e=evidence, residual=e?e.physicalSnapshot.maxGiB-e.measured.weights_main_plus_draft_gib_measured-e.measured.kv_pool_physical_gib:0;
    return {platform,context:1048576,output:0,globalConcurrency:dp,maxSeqs:1,localTP:platform==='a3'?16:8,utilization:platform==='a3'?0.93:0.92,
      align:true,c8:false,recover:false,shared4:false,extraFullBlocks:0,extraStateBlocks:0,
      runtimeGiB:residual,extraGiB:0,scaleBytesPerLayer:platform==='a3'?2350:2352};
  }
  function plannedWeights(platform, s, e){
    const dp=platform==='a3'?64:32;
    const split=new Set(['Embedding','LM head','Dense MLP','KDA QKV','KDA beta','KDA G','KDA O','DSA G','DSA O','Shared experts']);
    const rows=Object.entries(e.weights[platform]).map(([name,x])=>{
      let bytes=x.weight+x.meta+x.runtime_delta;
      if(name==='KDA composed F')bytes=0;
      else if(name==='Routed experts')bytes=bytes/896*Math.ceil(896/dp);
      else if(split.has(name))bytes/=s;
      // W8A8 row-parallel output scales/offsets are replicated, not split along K.
      if(platform==='a3'&&['Dense MLP','Shared experts'].includes(name)){
        const layers=name==='Shared experts'?92:1;
        bytes+=layers*H*8*(1-1/s);
      }
      return {name,bytes};
    });
    const draft=e.draftWeightsBytes;
    const main=rows.reduce((a,x)=>a+x.bytes,0);
    // Anchor A3 to actual loading, preserving packing/loader deltas, even when the localTP input changes.
    const baseMain=platform==='a3'&&s!==16?plannedWeights('a3',16,e).main:main;
    const adjustment=platform==='a3'?e.measured.weights_main_plus_draft_gib_measured*GiB-baseMain-draft:0;
    return {main,draft,adjustment,total:main+draft+adjustment,rows};
  }
  function sharedWeights(platform,s){
    const values=92*3*H*I, outputs=92*(2*I/s+H);
    if(platform==='a3'){
      // Current W8A8: FP32 scale + offset. W4A8: packed INT4, INT64 encoded scale,
      // retained FP32 scale, FP32 offset and FP32 folded scale_bias.
      const oldPayload=values/s, oldMetadata=outputs*8;
      const newPayload=values/(2*s), newMetadata=outputs*20;
      return {oldPayload,oldMetadata,newPayload,newMetadata,saved:oldPayload+oldMetadata-newPayload-newMetadata};
    }
    const oldPayload=values/s, oldMetadata=values/(32*s);
    const newPayload=values/(2*s),newMetadata=oldMetadata;
    return {oldPayload,oldMetadata,newPayload,newMetadata,saved:oldPayload-newPayload};
  }
  function recoverMemory(maxSeqs){
    const correction=NK*maxSeqs*HEADS*(SPEC+1)*DIM*4;
    const keyGate=NK*maxSeqs*HEADS*(SPEC+1)*(2*DIM)*2;
    // PR21549 CommitContext: ten INT64 layer-address/stride vectors; A_log/dt_bias
    // FP32 stacks; six INT32 arrays indexed by max_num_reqs. No page-indexed records.
    const metadata=NK*10*8+NK*HEADS*4+NK*HEADS*DIM*4+6*maxSeqs*4;
    return {correction,keyGate,metadata,total:correction+keyGate+metadata};
  }
  function calculate(p,e){
    if(!['a3','a5'].includes(p.platform))throw Error('平台必须是 A3 或 A5');
    for(const k of ['context','globalConcurrency','maxSeqs','localTP'])if(!Number.isSafeInteger(p[k])||p[k]<1)throw Error(k+' 必须是正整数');
    for(const k of ['output','extraFullBlocks','extraStateBlocks','scaleBytesPerLayer'])if(!Number.isSafeInteger(p[k])||p[k]<0)throw Error(k+' 必须是非负整数');
    for(const k of ['runtimeGiB','extraGiB'])if(!Number.isFinite(p[k])||p[k]<0)throw Error(k+' 必须非负');
    if(!Number.isFinite(p.utilization)||p.utilization<=0||p.utilization>1)throw Error('memory utilization 必须在 (0,1]');
    const dp=p.platform==='a3'?64:32;
    if(dp%p.localTP||HEADS%p.localTP||I%p.localTP)throw Error('localTP 必须同时整除 DP、96 heads 与 6144 intermediate');
    const requests=Math.ceil(p.globalConcurrency/dp);
    if(p.maxSeqs<requests)throw Error('max_num_seqs 小于最繁忙 rank 的请求数');
    const tokens=p.context+p.output, coverage=tokens+SPEC;
    const fullPerRequest=Math.ceil(coverage/BLOCK), fullBlocks=requests*fullPerRequest+p.extraFullBlocks;
    const stateCopies=(p.align?2:1)+(p.recover?0:SPEC);
    const statePerGroup=requests*stateCopies+p.extraStateBlocks;
    const stateBlocks=3*statePerGroup;
    const nullBlocks=1, blocks=fullBlocks+stateBlocks+nullBlocks;
    const indexBytes=p.platform==='a3'?130:132, draftBytes=p.c8?576:1152;
    const mainRegion=ND*PAGE,indexRegion=ND*BLOCK*indexBytes,draftRegion=DRAFT*BLOCK*draftBytes;
    const poolBytesPerID=mainRegion+indexRegion+draftRegion,poolBytes=blocks*poolBytesPerID;
    const effectiveTokens=requests*tokens;
    const contentMain=effectiveTokens*ND*MAIN,contentIndexer=effectiveTokens*ND*indexBytes;
    const contentDraft=effectiveTokens*DRAFT*draftBytes;
    const contentState=NK*statePerGroup*STATE;
    const statePadding=NK*statePerGroup*(PAGE-STATE);
    const holes=stateBlocks*(ND-23)*PAGE;
    const sidecars=stateBlocks*(indexRegion+draftRegion);
    const lookaheadBytes=requests*SPEC*(ND*(MAIN+indexBytes)+DRAFT*draftBytes);
    const fullTail=(requests*fullPerRequest*BLOCK-effectiveTokens-requests*SPEC)*(ND*(MAIN+indexBytes)+DRAFT*draftBytes);
    const retainedHistory=p.extraFullBlocks*poolBytesPerID;
    const nullBytes=poolBytesPerID;
    const waste=statePadding+holes+sidecars+fullTail;
    const content=contentMain+contentIndexer+contentDraft+contentState;
    const recon=content+lookaheadBytes+waste+retainedHistory+nullBytes;
    if(recon!==poolBytes)throw Error('物理缓存账本未闭合');
    const rmem=recoverMemory(p.maxSeqs), scaleBytes=p.c8?DRAFT*p.scaleBytesPerLayer:0;
    const shared=sharedWeights(p.platform,p.localTP),weights=plannedWeights(p.platform,p.localTP,e);
    const weightBytes=weights.total-(p.shared4?shared.saved:0);
    const addedRuntime=scaleBytes+(p.recover?rmem.total:0);
    const knownBytes=weightBytes+poolBytes+addedRuntime;
    const totalBytes=knownBytes+(p.runtimeGiB+p.extraGiB)*GiB;
    const layers=[];
    const kda=e.layers[p.platform].kda, dsa=e.layers[p.platform].dsa;
    for(let layer=0;layer<93;layer++){
      const isKda=kda.includes(layer), j=isKda?kda.indexOf(layer):dsa.indexOf(layer);
      if(isKda){
        const group=j%3+1,slot=Math.floor(j/3);
        layers.push({layer,type:'KDA',group:'M'+group,slot,headShape:'[96,128,128] FP32',convShape:'[6,36864] BF16',
          tokenBytes:0,contentBytes:statePerGroup*STATE,allocatedBytes:statePerGroup*PAGE,
          paddingBytes:statePerGroup*(PAGE-STATE),lookaheadBytes:0,retainedBytes:0,offsetBytes:slot*blocks*PAGE,
          layerStride:blocks*PAGE,blockStride:PAGE,indexOffset:null,
          layout:`主 MLA 区 slot ${slot}；M1/M2/M3 的 pool ID 互斥，不能同时写同一 ID；localTP 不切 state`});
      }else{
        layers.push({layer,type:'DSA',group:'F',slot:j,headShape:'512×C8 + 64×BF16 + 4×FP32 = 656 B',
          convShape:`Indexer: 128×C8 + 1×${p.platform==='a3'?'FP16':'FP32'} = ${indexBytes} B`,
          tokenBytes:MAIN+indexBytes,contentBytes:effectiveTokens*(MAIN+indexBytes),allocatedBytes:fullBlocks*BLOCK*(MAIN+indexBytes),
          paddingBytes:(requests*fullPerRequest*BLOCK-effectiveTokens-requests*SPEC)*(MAIN+indexBytes),
          lookaheadBytes:requests*SPEC*(MAIN+indexBytes),retainedBytes:p.extraFullBlocks*BLOCK*(MAIN+indexBytes),offsetBytes:j*blocks*PAGE,
          layerStride:blocks*PAGE,blockStride:PAGE,indexOffset:blocks*mainRegion+j*blocks*BLOCK*indexBytes,
          layout:'独立 MLA / Indexer region；共用 F 组 block table，不共享各层 KV 内容'});
      }
    }
    for(let d=0;d<DRAFT;d++)layers.push({layer:93+d,type:'DSpark',group:'F',slot:d,
      headShape:p.c8?'[512+64] × C8 = 576 B；独立 static scale':'[512+64] × BF16 = 1152 B',convShape:'无 Indexer，无 KDA state',
      tokenBytes:draftBytes,contentBytes:effectiveTokens*draftBytes,allocatedBytes:fullBlocks*BLOCK*draftBytes,
      paddingBytes:(requests*fullPerRequest*BLOCK-effectiveTokens-requests*SPEC)*draftBytes,
      lookaheadBytes:requests*SPEC*draftBytes,retainedBytes:p.extraFullBlocks*BLOCK*draftBytes,
      offsetBytes:blocks*(mainRegion+indexRegion)+d*blocks*BLOCK*draftBytes,
      layerStride:blocks*BLOCK*draftBytes,blockStride:BLOCK*draftBytes,indexOffset:null,
      layout:`draft ${d}；target feature layer ${[75,79,83,87,91][d]}。共享 block ID，不与 target 共用 KV tensor`});
    return {p,dp,requests,tokens,coverage,fullPerRequest,fullBlocks,stateCopies,statePerGroup,stateBlocks,blocks,
      indexBytes,draftBytes,mainRegion,indexRegion,draftRegion,poolBytesPerID,poolBytes,contentMain,contentIndexer,contentDraft,contentState,
      statePadding,holes,sidecars,fullTail,lookaheadBytes,retainedHistory,nullBytes,waste,content,scaleBytes,rmem,shared,weights,weightBytes,addedRuntime,
      knownBytes,totalBytes,layers,
      canonicalState:requests*NK*STATE,policyStateOverhead:contentState-requests*NK*STATE,
      physicalCapacityGiB:p.platform==='a3'?64:96,
      frameworkCapacityGiB:p.platform==='a3'?e.framework.total_device_bytes/GiB:91.75,
      requestedBudgetBytes:(p.platform==='a3'?e.framework.total_device_bytes:91.75*GiB)*p.utilization,
      clusterPoolBytes:poolBytes*dp,clusterTotalBytes:totalBytes*dp};
  }
  function scenarios(p,e){
    const base=calculate({...p,c8:false,recover:false,shared4:false},e);
    const configs=[['基线',false,false,false],['仅 DSpark C8',true,false,false],['仅 RecoverSSM',false,true,false],
      ['仅 shared W4A8',false,false,true],['C8 + RecoverSSM',true,true,false],['三项全部',true,true,true]];
    return configs.map(([name,c8,recover,shared4])=>{
      const r=calculate({...p,c8,recover,shared4},e);
      return {name,c8,recover,shared4,result:r,poolSaved:base.poolBytes-r.poolBytes,
        weightSaved:base.weightBytes-r.weightBytes,extra:r.addedRuntime-base.addedRuntime,
        saved:base.knownBytes-r.knownBytes};
    });
  }
  const api={GiB,MiB,NK,ND,DRAFT,H,I,HEADS,DIM,SPEC,BLOCK,MAIN,CONV,STATE,PAGE,defaults,plannedWeights,sharedWeights,recoverMemory,calculate,scenarios};
  root.KimiMemory=api;if(typeof module!=='undefined')module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
