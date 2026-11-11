const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const ctx={window:{}};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/dcp-overlap.js'),'utf8'),ctx);const d=ctx.window.A5_DCP_OVERLAP;
test('DCP evidence keeps sample sizes and module boundaries explicit',()=>{
  assert.equal(d.summary.dsaQueryAG.total,528);assert.equal(d.summary.dsaQueryAG.steady,504);
  for(const key of ['dense','head','embeddingMain'])assert.equal(d.summary[key].steady,21);
  assert.ok(d.source.includes('1105_pr20878'));assert.ok(d.limitations.some(x=>x.includes('envelope')));
});
test('device interval intersections stay bounded and do not count CPU launch as compute',()=>{
  for(const snippet of Object.values(d.snippets)){
    assert.ok(Number.isFinite(snippet.spanUs)&&snippet.spanUs>0);
    for(const e of snippet.events){
      assert.ok(Number.isFinite(e.start)&&Number.isFinite(e.us)&&e.us>=0);
      assert.ok(['COMMUNICATION','AI_CORE','MIX_AIC','AI_VECTOR_CORE','MIX_AIV'].includes(e.engine));
    }
    for(const comm of snippet.events.filter(e=>e.engine==='COMMUNICATION')){
      const clipped=snippet.events.filter(e=>e.engine!=='COMMUNICATION').map(e=>[Math.max(e.start,comm.start),Math.min(e.start+e.us,comm.start+comm.us)]).filter(([a,b])=>b>a).sort((a,b)=>a[0]-b[0]);
      let total=0,left=0,right=0;for(const [a,b] of clipped){if(a>right){total+=right-left;left=a;right=b;}else right=Math.max(right,b);}total+=right-left;assert.ok(total>=0&&total<=comm.us+1e-6);
    }
  }
  assert.ok(d.summary.dsaQueryAG.computeOverlap.median>50);
  for(const key of ['dsaCandidateAG','dsaOutputA2A','head','embeddingMain'])assert.equal(d.summary[key].computeOverlap.median,0);
  assert.equal(d.summary.dense.agComputeOverlap.median,0);assert.equal(d.summary.dense.rsComputeOverlap.median,0);
});
