import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=name=>fs.readFileSync(path.join(root,'src',name),'utf8');
const evidence=JSON.parse(fs.readFileSync(path.join(root,'data/memory-evidence.json'),'utf8'));
const css=read('style.css')+'\n'+read('memory.css');
const js=('window.KIMI_MEMORY_EVIDENCE='+JSON.stringify(evidence)+';\n'+read('memory-model.js')+'\n;\n'+read('memory-view.js')).replaceAll('</script','<\\/script');
const html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Kimi K3.1 1M缓存物理账本：A3 DP64与A5 DP32/localTP，逐层dtype、混合池浪费、DSpark C8、RecoverSSM、shared W4A8计算器。"><title>Kimi K3.1 · 1M 显存计算器</title><style>${css}</style></head>
<body data-page="memory-calculator"><div class="mem-wrap"><nav class="mem-nav"><b>Kimi K3.1 / 1M MEMORY · 2026-11-12</b><a href="index.html">A5 localTP 主站</a><a href="https://github.com/Dawn952/kimi-k31-a5-analysis">GitHub / 公式与证据</a><a href="#mem-scenarios">C8 / RecoverSSM / W4A8</a></nav>
<header class="mem-heading"><h1>1M 上下文，显存花在哪里</h1><p>以 A3 实际分配的 62 个 cache pool blocks 校准，逐字节展开主模型、DSpark、KDA state、混合页空耗与运行时残差。对照 A3 DP64/TP1/localTP16 与 A5 DP32/TP1/localTP8；所有结果显式区分实测、源码推算和运行时假设。</p></header>
<div class="mem-layout"><aside id="mem-controls" class="mem-controls"></aside><main id="mem-results" style="padding:0;max-width:none;min-width:0"></main></div><footer class="mem-footer">整数 bytes 核算；GiB = 2³⁰ bytes。离线可用，无网络依赖。容量规划不是1M端到端或精度验收。</footer></div><script>${js}</script></body></html>`;
fs.writeFileSync(path.join(root,'docs/memory.html'),html);
console.log('Built self-contained docs/memory.html: '+Buffer.byteLength(html)+' bytes');
