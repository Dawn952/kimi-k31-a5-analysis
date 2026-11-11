const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
test('standalone entry initializes and renders every family without external scripts or private paths',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../docs/index.html'),'utf8');
  assert.ok(!/<script[^>]+src=/i.test(html));
  assert.ok(!/C:\/Users|D:\/|\/mnt\/|hust17yixuan|141\.61\.|80\.5\.17\./.test(html));
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,{innerHTML:'',value:'',classList:{add(){},remove(){},toggle(){}},addEventListener(){}});return elements.get(id);};
  const document={body:{classList:{remove(){}}},querySelector:get,querySelectorAll:()=>[]};
  const ctx={document,setTimeout};ctx.window=ctx;vm.createContext(ctx);
  vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1],ctx);
  for(const family of ['kda','dsa','embedding','head','dense'])for(const pipeline of [false,true]){
    vm.runInContext(`lp.family='${family}';lp.pipeline=${pipeline};renderLocalTP();`,ctx);
    assert.ok(get('#lp-canvas').innerHTML.includes('<svg'));
    assert.ok(!get('#lp-canvas').innerHTML.includes('NaN'));
    assert.ok(get('#lp-results').innerHTML.includes('每卡新增通信账本'));
  }
});
