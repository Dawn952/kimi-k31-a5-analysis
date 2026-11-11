/* Minimal standalone host for the A5 analysis; no framework or network dependency. */
const $=(s,r=document)=>r.querySelector(s);
const $$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=2)=>Number.isFinite(v)?Number(v).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
const state={model:'k31'};
function metric(label,value,unit,note){return `<div class="metric"><div class="label">${label}</div><strong class="value">${value}<span class="unit">${unit}</span></strong><small>${note}</small></div>`;}
function heading(number,title,description){return `<h1>${esc(title)}</h1><p>${esc(description)}</p>`;}
function download(name,value,type='application/json'){
  const blob=new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
