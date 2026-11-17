import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(import.meta.url);
const evidence=JSON.parse(fs.readFileSync(path.join(root,'data/localtp-profile-20261113.json'),'utf8'));
const P=require('../src/localtp-profile.js').create(evidence);
const M=require('../src/localtp-model.js'),D=require('../src/diagram-layout.js');
const elements={};
const ctx={window:{KimiLocalTP:M,KimiA5LocalTPProfile:P},KimiDiagram:D,
  fmt:(n,d=2)=>n.toFixed(d),$:(key)=>elements[key]||(elements[key]={innerHTML:''}),$$:()=>[],
  esc:x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(root,'src/localtp-view.js'),'utf8'),ctx);
const graphs={};
for(const family of Object.keys(evidence.domains)){
  vm.runInContext(`lp.family='${family}';lpZoom=1;drawLocalGraph();`,ctx);
  fs.writeFileSync(path.join(root,`docs/images/a5-${family}-localtp.svg`),elements['#lp-canvas'].innerHTML);
  const graph=P.graph(family);D.customLayout(graph,graph.stages);graphs[family]=graph;
}
fs.writeFileSync(path.join(root,'data/localtp-profile-graphs.json'),JSON.stringify(graphs,null,2)+'\n');
console.log('Exported six source-aligned localTP SVG diagrams');
