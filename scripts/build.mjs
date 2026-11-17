import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=name=>fs.readFileSync(path.join(root,'src',name),'utf8');
const css=['style.css','localtp.css'].map(read).join('\n');
const profile=JSON.parse(fs.readFileSync(path.join(root,'data/localtp-profile-20261113.json'),'utf8'));
const js=('window.KimiA5LocalTPProfileData='+JSON.stringify(profile)+';\n'+['diagram-layout.js','localtp-model.js','localtp-profile.js','shared-tp-current.js','shell.js','localtp-view.js'].map(read).join('\n;\n')).replaceAll('</script','<\\/script');
const html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Kimi K3.1 A5 纯DP/localTP：显存权重、计算通信依赖、Cube/Vector/Communication 多流与DCP profiling对照。"><title>Kimi K3.1 · A5 localTP 分析</title>
<style>${css}
.public-wrap{max-width:1900px;margin:auto;padding:22px}body[data-page=localtp] main{padding:0;max-width:none}.public-nav{display:flex;align-items:center;gap:22px;flex-wrap:wrap;max-width:1856px;margin:0 auto 22px;padding-bottom:14px;border-bottom:1px solid #304957;font-size:12px}.public-nav b{margin-right:auto;color:#bddfe3}.public-evidence{margin:24px 0}.public-evidence p{font-size:12px;line-height:1.9;color:#a9c0cd}.public-evidence code{overflow-wrap:anywhere}@media(max-width:700px){.public-wrap{padding:14px}.public-nav{gap:12px}}
</style></head><body data-page="localtp"><div class="public-wrap"><nav class="public-nav"><b>Kimi K3.1 / A5 · 2026-11-11</b><a href="memory.html">1M 显存计算器 · A3 / A5</a><a href="https://github.com/Dawn952/kimi-k31-a5-analysis/blob/main/design/README.md">软件设计 · 六类细粒度TP</a><a href="https://github.com/vllm-project/vllm-ascend/pull/21624">实现 PR21624 · Draft</a><a href="https://github.com/Dawn952/kimi-k31-a5-analysis">GitHub 图文与源码</a><a href="#source-evidence">来源 / G与O核对</a></nav><main id="main"><div id="view"></div></main>
<section id="source-evidence" class="panel public-evidence"><h2>Gating G 与 O 权重核对</h2><p>本表为 attention 输出 <code>g_proj.weight</code>。A5 实际量化 checkpoint header 中 G 与 O 都是 BF16，每层各 88,080,384 参数、176,160,768 bytes = 168 MiB。维度顺序相反，权重数值独立。93层总量各15.2578125 GiB。</p><table class="data-table"><thead><tr><th>矩阵</th><th>完整 shape [out,in]</th><th>localTP8 shape</th><th>完整 / 每卡</th></tr></thead><tbody><tr><td>G</td><td>[12288,7168]</td><td>[1536,7168]</td><td>168 / 21 MiB</td></tr><tr><td>O</td><td>[7168,12288]</td><td>[7168,1536]</td><td>168 / 21 MiB</td></tr></tbody></table>
<h2 style="margin-top:24px">来源与范围</h2><p>计算与通信依赖流采用 A5 localTP 实际 profiling 的固定部署版本3634a7eae948，节点链接指向该版本；最新 PR 源码学习版本976ea36e9单独标注。权重来自 checkpoint header / 加载后校准；payload 与带宽预算为参数估算。图中交集是单rank设备区间观察，不是端到端加速或完整硬件验收。</p></section></div><script>${js}\nrenderLocalTP();</script></body></html>`;
const outputHtml=html.replace('实现 PR21624 · Draft','实现 PR21624')
  .replace('<b>Kimi K3.1 / A5 · 2026-11-11</b>','<b>Kimi K3.1 / A5 · localTP</b>')
  .replace('软件设计 · 六类细粒度TP','细粒度TP设计')
  .replace('<a href="#source-evidence">来源 / G与O核对</a>','<a href="#lp-flow">计算与通信依赖流</a><a href="#source-evidence">G/O权重核对</a>');
fs.writeFileSync(path.join(root,'docs/index.html'),outputHtml);
fs.writeFileSync(path.join(root,'docs/images/shared-localtp-current-20261117.svg'),
  (await import('node:module')).createRequire(import.meta.url)('../src/shared-tp-current.js').svg);
console.log('Built self-contained docs/index.html: '+Buffer.byteLength(outputHtml)+' bytes');
await import('./build-memory.mjs');
await import('./export-profile-diagrams.mjs');
