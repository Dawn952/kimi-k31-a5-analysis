const test=require('node:test'),assert=require('node:assert/strict');
const M=require('../src/memory-dcp.js');
const calc=changes=>M.calculate({...M.defaults(),...changes});
test('TP8 state and DCP8 history have independent split dimensions',()=>{
  const r=calc();
  assert.equal(r.coverage,12288);assert.equal(r.mainPage,1007616);
  assert.equal(r.recurrent,12*128*128*4);assert.equal(r.conv,6*4608*2);
  assert.equal(r.state,841728);assert.equal(r.bytesPerId,37896192);
  assert.equal(r.rec.correction,69*64*12*4*128*4);
  assert.equal(r.rec.total,217489536);
  assert.equal(r.poolBytes,r.sharedBytes+r.privateBytes+r.stateContent+r.statePadding+r.holes+r.sidecars+r.bytesPerId);
});
test('explicit budget sharing, DSpark tail and null are counted once per engine',()=>{
  const r=calc({kvGiB:21,requests:64});assert.equal(r.sharedIds,9);assert.equal(r.sharedTokens,110592);
  assert.equal(r.totalHistory,11);assert.equal(r.privateHistory,2);
  assert.equal(r.ids,9+64*(2+6)+1);assert.equal(r.poolIds,595);
  assert.equal(r.capacity,73);assert.equal(r.globalCapacity,292);
  assert.equal(r.globalAdmitted,256);
  const next=calc({kvGiB:21,requests:65});assert.equal(next.poolBytes-r.poolBytes,8*r.bytesPerId);
  assert.equal(calc({kvGiB:21,requests:73}).ids,594);assert.equal(calc({kvGiB:21,requests:74}).ids,602);
});
test('sharing, replicated Indexer and unknown runtime stay explicit',()=>{
  const a=calc(),no=calc({shared:false}),rep=calc({indexerSharded:false});
  assert.equal(no.ids,52*(11+6)+1);assert.ok(no.poolBytes>a.poolBytes);
  assert.equal(rep.indexRegion,8*a.indexRegion);assert.ok(rep.capacity<a.capacity);
  assert.equal(a.totalBytes,null);assert.equal(calc({runtimeGiB:0}).totalBytes,a.knownBytes);
  assert.equal(calc({dropTail:false}).sharedTokens,122880);
  assert.equal(calc({budgetIncludesRecords:false}).available,a.available-a.rec.total);
});
test('default budget comes from matched post-graph evidence, never the unrelated 21 GiB run',()=>{
  const e=require('../data/memory-dcp-evidence.json'),r=calc();
  assert.equal(r.available,16423116451);assert.equal(r.poolIds,433);
  assert.equal(r.capacity,52);assert.equal(r.globalAdmitted,208);
  assert.equal(r.p.weightsGiB,61.08-e.recordsInMeasuredLoadBytes/M.GiB);
  assert.equal(e.postGraphBudgetBytes,Math.min(...e.profiles.map(x=>x.fitRequestedBytes)));
  const groupCaps=e.profiles.map(x=>calc({kvGiB:x.fitRequestedBytes/M.GiB}).capacity);
  assert.deepEqual(groupCaps,[53,53,52,52]);assert.equal(groupCaps.reduce((a,b)=>a+b,0),210);
  for(const x of e.profiles){
    assert.equal(Math.floor(x.allocatedPoolIdsInferred/17*x.modelLen),x.reportedCapacityTokens);
  }
  assert.equal(calc({kvGiB:497*r.bytesPerId/M.GiB}).capacity,60);
});
test('records follow max_num_seqs, capacity respects memory and scheduler limits',()=>{
  const a=calc(),b=calc({requests:1}),c=calc({maxSeqs:128});
  assert.equal(a.rec.total,b.rec.total);assert.ok(c.rec.total>a.rec.total);
  assert.equal(c.globalAdmitted,c.globalCapacity);
  assert.equal(calc({kvGiB:0}).globalAdmitted,0);
  assert.throws(()=>calc({blockSize:768}),/装不下/);
  assert.throws(()=>calc({blockSize:1500}),/128/);
});
test('1M paired pool and native speculative state difference reconcile',()=>{
  const r=calc({context:1048576,output:0,requests:1,shared:false});
  assert.equal(r.ids,86+6+1);assert.equal(r.poolBytes,93*37896192);
  const off=calc({context:1048576,output:0,requests:1,shared:false,recover:false});
  assert.equal(off.ids-r.ids,9);assert.equal(off.poolBytes-r.poolBytes,9*r.bytesPerId);
});
