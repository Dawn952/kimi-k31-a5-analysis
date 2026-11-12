const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('../src/memory-model.js'),E=require('../data/memory-evidence.json');
const calc=(hw='a3',changes={})=>M.calculate({...M.defaults(hw,E),...changes},E);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);

test('measured A3 arena is counted once: exact 62-block calibration',()=>{
  const r=calc();assert.equal(r.poolBytesPerID,302579712);
  assert.equal(E.pool.physicalBytes,62*r.poolBytesPerID);
  near(E.pool.physicalBytes/M.GiB,E.measured.kv_pool_physical_gib);
  assert.equal(E.cacheSpecs.length,122);
  assert.equal(E.pool.groups.length,4);
  for(const g of E.pool.groups.slice(1))assert.equal(g.layers.length,23);
  near(r.weightBytes/M.GiB,37.70064401626587);
  const measuredEquivalent=r.weightBytes+62*r.poolBytesPerID+r.p.runtimeGiB*M.GiB;
  near(measuredEquivalent/M.GiB,E.physicalSnapshot.maxGiB);
});
test('1M A3 and A5 match independently recorded page-count assessment',()=>{
  const a3=calc(),a5=calc('a5');
  for(const r of [a3,a5]){
    assert.equal(r.fullPerRequest,86);assert.equal(r.stateCopies,5);assert.equal(r.stateBlocks,15);assert.equal(r.blocks,102);
    assert.equal(r.layers.length,98);assert.equal(r.layers.filter(x=>x.type==='KDA').length,69);
    assert.equal(r.layers.filter(x=>x.type==='DSA').length,24);assert.equal(r.layers.filter(x=>x.type==='DSpark').length,5);
  }
  near(a3.poolBytes/M.GiB,28.7435302734375);
  near(a5.poolBytes/M.GiB,28.799560546875);
  assert.equal(a5.poolBytes-a3.poolBytes,102*24*12288*2);
  assert.equal(M.STATE,6733824);assert.equal(M.PAGE,8060928);
  assert.ok(E.profileCacheShapes.find(x=>x.type==='QuantLightningIndexer').inputShapes.includes('5952,128,1,128'));
});
test('98 layer views plus overhead reconcile without repeating shared backing',()=>{
  for(const hw of ['a3','a5'])for(const c8 of [false,true])for(const recover of [false,true])for(const align of [false,true]){
    const r=calc(hw,{c8,recover,align});
    const views=r.layers.reduce((a,x)=>a+x.allocatedBytes,0);
    for(const x of r.layers)assert.equal(x.contentBytes+x.paddingBytes+x.lookaheadBytes+x.retainedBytes,x.allocatedBytes);
    assert.equal(views+r.holes+r.sidecars+r.nullBytes,r.poolBytes);
    assert.equal(r.content+r.lookaheadBytes+r.waste+r.retainedHistory+r.nullBytes,r.poolBytes);
    assert.equal(r.layers.find(x=>x.layer===92).type,'DSA');
    assert.equal(r.layers.find(x=>x.layer===0).blockStride,8060928);
    const layers=r.layers.filter(x=>x.type==='KDA');
    for(const [g,mod] of [['M1',0],['M2',1],['M3',2]]){
      const a=layers.filter(x=>x.group===g);assert.equal(a.length,23);
      assert.ok(a.every(x=>x.layer%4===mod));
    }
  }
});
test('localTP changes weights but never owner KV or recurrent state geometry',()=>{
  for(const hw of ['a3','a5']){
    const a=calc(hw,{localTP:1}),b=calc(hw,{localTP:16});
    assert.equal(a.poolBytes,b.poolBytes);assert.equal(a.contentState,b.contentState);
    assert.ok(a.weightBytes>b.weightBytes);
  }
});
test('C8 static metadata and RecoverSSM records are charged; combo not sum of isolated savings',()=>{
  for(const hw of ['a3','a5']){
    const p=M.defaults(hw,E),s=M.scenarios(p,E),base=s[0].result,c8=s[1],rec=s[2],combo=s[4];
    assert.equal(c8.result.scaleBytes,5*p.scaleBytesPerLayer);
    assert.equal(base.poolBytes-c8.result.poolBytes,102*5*12288*576);
    assert.equal(rec.result.stateBlocks,6);assert.equal(rec.result.blocks,93);
    assert.equal(rec.result.rmem.correction,13565952);assert.equal(rec.result.rmem.keyGate,13565952);
    assert.equal(rec.result.rmem.metadata,3423528);
    const interaction=(c8.saved+rec.saved)-combo.saved;
    assert.equal(interaction,9*5*12288*576);
    assert.equal(combo.result.poolBytes,(86+6+1)*(base.poolBytesPerID-5*12288*576));
  }
});
test('shared W4A8 includes row-replicated output metadata and scale copies',()=>{
  const a=M.sharedWeights('a3',16),b=M.sharedWeights('a5',8);
  assert.equal(a.oldMetadata,92*(2*384+7168)*8);
  assert.equal(a.newMetadata,92*(2*384+7168)*20);
  assert.equal(a.saved,371085312);assert.equal(b.saved,759693312);
  const r=calc('a3',{shared4:true});assert.equal(calc().weightBytes-r.weightBytes,a.saved);
});
test('block-boundary/lookahead, global concurrency and retention are explicit',()=>{
  assert.equal(calc('a3',{context:12285}).fullPerRequest,1);
  assert.equal(calc('a3',{context:12286}).fullPerRequest,2);
  const a=calc(),b=calc('a3',{globalConcurrency:128,maxSeqs:2});
  assert.equal(b.requests,2);assert.equal(b.blocks,203);
  assert.equal(b.poolBytes,2*a.poolBytes-a.nullBytes); // exactly one pool null
  const c=calc('a3',{extraFullBlocks:7,extraStateBlocks:2});
  assert.equal(c.poolBytes-a.poolBytes,13*a.poolBytesPerID);
  assert.equal(c.content+c.lookaheadBytes+c.waste+c.retainedHistory+c.nullBytes,c.poolBytes);
  assert.throws(()=>calc('a3',{globalConcurrency:65}),/max_num_seqs/);
  assert.throws(()=>calc('a3',{localTP:64}),/localTP/);
  assert.throws(()=>calc('a3',{context:1.5}),/正整数/);
  assert.throws(()=>calc('a3',{runtimeGiB:NaN}),/必须非负/);
  assert.throws(()=>calc('a3',{utilization:1.01}),/utilization/);
  assert.ok(calc('a5',{c8:true,recover:true,shared4:true}).knownBytes>calc('a5').requestedBudgetBytes);
});
test('public page self-contained and excludes local paths and host identifiers',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../docs/memory.html'),'utf8');
  assert.ok(!/<script[^>]+src=/i.test(html));
  assert.ok(!/C:\/Users|D:\/|\/mnt\/|hust17yixuan|80\.5\.17\.|141\.61\./.test(html));
  assert.ok(html.includes('1M 上下文'));assert.ok(html.includes('runtimeGiB'));
});
