const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const S=require('../src/shared-tp-current.js');
test('current shared source map covers sharding, collectives and quant contracts',()=>{
  const html=S.render();
  for(const text of ['976ea36e9','8cd3c3762','linear_op.py:457','shared_experts.py:686','fused_moe.py:367','W8A8MXFP','W4A8MXFP','W8A8_DYNAMIC'])assert.ok(html.includes(text),text);
  assert.ok(html.includes(S.source('ops/linear_op.py',457)));
  assert.ok(!/C:\/Users|D:\/|\/mnt\/|141\.61\./.test(html));
});
test('static shared SVG matches the inline diagram and keeps owner dimensions',()=>{
  assert.equal(fs.readFileSync('docs/images/shared-localtp-current-20261117.svg','utf8'),S.svg);
  for(const text of ['[C,H]','[P·C,H]','[M/P,H]','[H,M/P]','Output RS','owner p'])assert.ok(S.svg.includes(text),text);
});
