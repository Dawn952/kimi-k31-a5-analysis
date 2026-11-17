const test=require('node:test'),assert=require('node:assert/strict');
const data=require('../data/localtp-profile-20261113.json');
const P=require('../src/localtp-profile.js').create(data),D=require('../src/diagram-layout.js');
const exported=require('../data/localtp-profile-graphs.json');
test('profile topology, phase and module counts stay tied to the actual capture',()=>{
  assert.equal(data.ascend_revision,'3634a7eae94897d1a7b14d00807323e5af6b4a04');
  assert.deepEqual([data.configuration.dp,data.configuration.tp,data.configuration.localtp],[32,1,8]);
  assert.equal(data.phase,'warmup-c128');assert.equal(data.kernel_records,43520);
  assert.deepEqual(Object.fromEntries(Object.entries(data.domains).map(([k,v])=>[k,v.sample_count])),
    {kda:690,dsa:240,shared:920,dense:10,head:10,embedding:20});
});
test('KDA keeps actual merged return and post-receive order, without invented G overlap',()=>{
  const g=P.graph('kda'),edge=(a,b)=>g.edges.some(e=>e.src===a&&e.dst===b);
  for(const [a,b] of [['quant','q'],['beta','q'],['q','cat'],['cat','ret'],['ret','restore'],['restore','conv'],['conv','core'],['gr','grestore'],['core','norm'],['olayout','oa'],['oa','o'],['o','rs']])assert.ok(edge(a,b),a+'→'+b);
  assert.equal(data.domains.kda.overlap_stats['gr:core'].max_us,0);
  assert.equal(data.domains.kda.overlap_stats['ret:g'].max_us,0);
  assert.ok(g.nodes.find(n=>n.id==='restore').records.some(r=>r.type==='Sigmoid'));
  assert.ok(g.nodes.find(n=>n.id==='restore').records.every(r=>r.stream==='425'));
});
test('shared positive config does not manufacture device overlap or extra streams',()=>{
  assert.equal(data.configuration.multistream_overlap_shared_expert,true);
  assert.ok(Object.values(data.domains.shared.overlap_stats).every(x=>x.max_us===0));
  const g=P.graph('shared');assert.equal(g.parallelGroups.length,0);
  assert.ok(g.nodes.filter(n=>n.records.length&&n.kind!=='comm').flatMap(n=>n.records).every(x=>x.stream==='425'));
  assert.ok(g.edges.some(e=>e.src==='combine'&&e.dst==='act'&&e.order));
  assert.ok(g.edges.some(e=>e.src==='rs'&&e.dst==='up_quant'&&e.order));
});
test('actual dense/head/embedding preserve both chunks and communication ordering',()=>{
  for(const family of ['dense','head','embedding']){
    const g=P.graph(family),ids=new Set(g.nodes.map(n=>n.id));
    assert.ok(ids.has('ag0')&&ids.has('ag1')&&ids.has('join'));
    assert.ok(g.edges.some(e=>e.src==='ag0'&&e.dst==='ag1'&&e.order));
    assert.ok(g.parallelGroups.length>0);
  }
  const g=P.graph('embedding');
  assert.ok(g.nodes.filter(n=>n.id.startsWith('lookup')).flatMap(n=>n.records).every(x=>x.core!=='AI_CPU'));
  assert.equal(g.nodes.find(n=>n.id==='rs0').records[0].type,'hcom_reduceScatter_AicpuKernel');
});
test('every rendered graph is a complete acyclic DAG with unobstructed edges',()=>{
  for(const family of Object.keys(data.domains)){
    const g=P.graph(family);D.customLayout(g,g.stages);
    assert.deepEqual(exported[family],g,'static graph drift: '+family);
    const ids=new Set(g.nodes.map(n=>n.id)),active=new Set(),done=new Set();
    const visit=id=>{assert.ok(!active.has(id),family+' cycle '+id);if(done.has(id))return;active.add(id);
      for(const edge of g.edges.filter(e=>e.src===id)){assert.ok(ids.has(edge.dst));visit(edge.dst);}
      active.delete(id);done.add(id);};
    for(const node of g.nodes){visit(node.id);assert.ok(Number.isFinite(node.x+node.y));}
    for(const edge of g.edges){assert.ok(ids.has(edge.src)&&ids.has(edge.dst));assert.ok(edge.points.length>=2);
      for(let i=1;i<edge.points.length;i++){
        const a=edge.points[i-1],b=edge.points[i];assert.ok(a.x===b.x||a.y===b.y);
        for(const n of g.nodes.filter(n=>n.id!==edge.src&&n.id!==edge.dst)){
          const crosses=a.x===b.x?a.x>n.x&&a.x<n.x+n.w&&Math.max(a.y,b.y)>n.y&&Math.min(a.y,b.y)<n.y+n.h:
            a.y>n.y&&a.y<n.y+n.h&&Math.max(a.x,b.x)>n.x&&Math.min(a.x,b.x)<n.x+n.w;
          assert.ok(!crosses,family+' edge crosses '+n.id);
        }
      }
    }
    for(const group of g.parallelGroups){const stat=data.domains[family].overlap_stats[group.id];assert.ok(stat.median_us>0);
      for(const part of group.parts)for(const id of part)assert.ok(ids.has(id));}
  }
});
