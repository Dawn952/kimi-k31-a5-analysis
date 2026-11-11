const test=require('node:test');
const assert=require('node:assert/strict');
const M=require('../src/localtp-model.js');
const D=require('../src/diagram-layout.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
test('A5 calibrated residency stages and existing shared reference reconcile',()=>{
  const r=M.calculate(M.defaults);
  [115.5823050737381,100.18270027637482,86.83211433887482,82.39332115650177,69.04273521900177].forEach((x,i)=>near(r.plans[i].weight,x));
  near(r.plans[4].weight-10.214813232421875,58.827921986579895);
  assert.ok(!r.rows.some(x=>x.id==='shared'));
});
test('collective accounting includes DSA G AG, excludes duplicate KDA G AG',()=>{
  const r=M.calculate(M.defaults),row=id=>r.rows.find(x=>x.id===id);
  near((row('dg').send-row('kg').send)/M.MiB,36.75);
  near(row('q').send/M.MiB,60.4365234375);near(row('o').send/M.MiB,44.625);
  near(r.plans[4].send,r.rows.reduce((sum,x)=>sum+x.send*x.layers,0));
  const s1=M.calculate({...M.defaults,s:1});assert.ok(s1.rows.every(x=>x.send===0&&x.calls===0));near(s1.plans[4].weight,115.5823050737381);
});
test('capacity scales traffic and GEMM work; embedding lookup does not save full-table reads',()=>{
  const a=M.calculate({...M.defaults,emb:384}),b=M.calculate({...M.defaults,emb:8192});
  near(b.rows.find(x=>x.id==='embedding').send/a.rows.find(x=>x.id==='embedding').send,8192/384);
  near(a.rows.find(x=>x.id==='embedding').read,0);
  const padded=M.calculate({...M.defaults,c:768});near(padded.padding,2);near(padded.rows[0].send,2*a.rows[0].send);
  assert.throws(()=>M.calculate({...M.defaults,c:1}));
  const fp32=M.calculate({...M.defaults,b:4}),bf16=M.calculate(M.defaults);
  near(fp32.rows.find(x=>x.id==='head').send-bf16.rows.find(x=>x.id==='head').send,(7/8)*384*163840*2);
});
test('co-located G/O removes exactly the G returns, preserves residency and real gate dependencies',()=>{
  const a=M.calculate(M.defaults),b=M.calculate({...M.defaults,colocal:true});
  near(a.plans[4].weight,b.plans[4].weight);near((a.plans[4].send-b.plans[4].send)/M.MiB,732.375);
  assert.equal(a.plans[4].calls-b.plans[4].calls,93);
  for(const family of ['kda','dsa']){
    const g=M.graph({...M.defaults,colocal:true,family});D.customLayout(g,g.stages);
    assert.ok(!g.nodes.some(n=>n.id==='gr'));assert.ok(!g.edges.some(e=>e.dst==='gate'&&['core','uv'].includes(e.src)));
    assert.deepEqual(g.edges.filter(e=>e.dst==='gate').map(e=>e.src).sort(),['g','oa']);
    assert.ok(g.edges.some(e=>e.src==='gate'&&e.dst==='o'));assert.ok(g.edges.some(e=>e.src==='o'&&e.dst==='rs'));
  }
});
test('KDA overlap frames pair Cube with Vector or Communication, never views/cache or same-engine branches',()=>{
  for(const schedule of ['early','late']){
    const g=M.graph({...M.defaults,schedule}),byId=Object.fromEntries(g.nodes.map(n=>[n.id,n]));
    const groups=new Map();
    for(const box of g.parallelGroups){if(!groups.has(box.id))groups.set(box.id,[]);groups.get(box.id).push(...box.parts.flat());}
    for(const ids of groups.values()){
      const engines=new Set(ids.map(id=>byId[id].engine));
      assert.equal(engines.size,2);assert.ok([...engines].every(x=>['Cube','Vector','Communication'].includes(x)));
      assert.ok(!engines.has('View')&&!engines.has('Cache'));
    }
    assert.equal(byId.core.engine,'Vector');assert.equal(byId.conv.engine,'Vector');assert.ok(!byId.fp&&!byId.beta);assert.ok(g.resourceLanes);
    assert.deepEqual(g.edges.filter(e=>e.dst==='core').map(e=>e.src).sort(),['conv','f','state']);
    assert.deepEqual(g.edges.filter(e=>e.dst==='return').map(e=>e.src).sort(),['bp','q']);
  }
});
test('KDA and DSA dependency DAGs route every edge around non-endpoint nodes',()=>{
  for(const family of ['kda','dsa'])for(const level of [1,2,3,4])for(const colocal of [false,true]){
    const g=M.graph({...M.defaults,family,level,colocal});D.customLayout(g,g.stages);
    const ids=new Set(g.nodes.map(x=>x.id));assert.equal(ids.size,g.nodes.length);
    const visited=new Set(),active=new Set();
    function visit(id){assert.ok(!active.has(id),'cycle: '+id);if(visited.has(id))return;active.add(id);for(const e of g.edges.filter(e=>e.src===id))visit(e.dst);active.delete(id);visited.add(id);}
    visit('in');assert.equal(visited.size,g.nodes.length-(family==='kda'?1:0)); // state cache is an external source.
    for(const e of g.edges){assert.ok(ids.has(e.src)&&ids.has(e.dst));assert.ok(e.points.length>=2);
      for(let i=1;i<e.points.length;i++){const a=e.points[i-1],b=e.points[i];assert.ok(a.x===b.x||a.y===b.y);
        for(const n of g.nodes.filter(n=>n.id!==e.src&&n.id!==e.dst))assert.ok(!(a.x===b.x?a.x>n.x&&a.x<n.x+n.w&&Math.max(a.y,b.y)>n.y&&Math.min(a.y,b.y)<n.y+n.h:a.y>n.y&&a.y<n.y+n.h&&Math.max(a.x,b.x)>n.x&&Math.min(a.x,b.x)<n.x+n.w),'edge crosses '+n.id);
      }
    }
    if(level>=4){const ag=g.edges.filter(x=>x.dst==='g');assert.equal(ag[0].src,'ag');}
    assert.ok(g.edges.some(x=>x.src==='g'&&x.dst==='gate')||g.edges.some(x=>x.src==='gr'&&x.dst==='gate'));
    if(family==='dsa'){
      assert.ok(g.edges.some(x=>x.src==='prolog'&&x.dst==='prep'));
      assert.ok(g.edges.some(x=>x.src==='prep'&&x.dst==='idx'));
      assert.equal(g.nodes.find(n=>n.id==='prolog').engine,'Mixed');
      assert.equal(g.nodes.find(n=>n.id==='idx').engine,'Mixed');
      assert.equal(g.nodes.find(n=>n.id==='attn').engine,'Mixed');
    }
  }
});
test('auxiliary blocks have no whole-batch overlap frames; chunk candidates preserve their own barriers',()=>{
  for(const family of ['embedding','head','dense'])for(const pipeline of [false,true]){
    const g=M.graph({...M.defaults,family,pipeline});D.customLayout(g,g.stages);
    const byId=Object.fromEntries(g.nodes.map(n=>[n.id,n]));
    if(!pipeline||family==='embedding')assert.equal(g.parallelGroups.length,0);
    else{
      const groups={};for(const box of g.parallelGroups)(groups[box.id]??=[]).push(...box.parts.flat());
      for(const ids of Object.values(groups))assert.deepEqual([...new Set(ids.map(id=>byId[id].engine))].sort(),['Communication','Cube']);
      for(const id of ['a','b']){
        if(family==='head'){
          assert.ok(g.edges.some(e=>e.src==='ag'+id&&e.dst==='mm'+id));assert.ok(g.edges.some(e=>e.src==='mm'+id&&e.dst==='a2a'+id));assert.ok(g.edges.some(e=>e.src==='a2a'+id&&e.dst==='sample'+id));
        }else{
          assert.ok(g.edges.some(e=>e.src==='silu'+id&&e.dst==='down'+id));assert.ok(g.edges.some(e=>e.src==='down'+id&&e.dst==='rs'+id));
        }
      }
    }
    const active=new Set(),done=new Set();function visit(id){assert.ok(!active.has(id));if(done.has(id))return;active.add(id);for(const e of g.edges.filter(e=>e.src===id)){assert.ok(byId[e.dst]);visit(e.dst);}active.delete(id);done.add(id);}for(const n of g.nodes)visit(n.id);
    for(const box of g.parallelGroups)for(const part of box.parts)for(const id of part)assert.ok(byId[id]);
  }
});
