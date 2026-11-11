/* Presentation geometry only. Tensor algebra and graph dependencies stay unchanged. */
(function(root){
  'use strict';
  const layouts={
    attention:{
      stages:['输入与序列分片','投影分支','Query 与历史','Attention','归并与输出'],
      positions:{in:[0,110],sp:[0,310],proj:[1,310],idx:[1,510],gate:[1,710],qgather:[2,110],qli:[2,510],cache:[2,710],merge:[3,510],attn:[3,110],a2a:[4,110],vup:[4,310],oproj:[4,510],out:[4,710],expand:[2,510]},
    },
    kda:{
      stages:['恢复 token 顺序','投影与门分支','局部预处理','状态计算','归并与输出'],
      positions:{in:[0,110],sp:[0,310],pcp:[0,510],qkv:[1,110],bfg:[1,510],conv:[2,110],l2:[2,310],gates:[2,510],state:[2,710],core:[3,310],gateout:[3,510],oproj:[4,310],reduce:[4,510],scatter:[4,710],out:[4,910]},
    },
    moe:{
      stages:['输入与路由','Hidden 分支','分发与专家输入','前馈计算','归并与输出'],
      positions:{in:[0,110],router:[0,310],latent:[1,110],dispatch:[2,110],gmm1:[2,310],situ:[3,310],gmm2:[3,510],combine:[4,510],norm:[4,710],sum:[4,910],sharedin:[1,710],shared:[2,710],sharedout:[3,710],dense:[1,910]},
    },
    attnres:{
      stages:['模型 / PP 输入','深度状态','候选打分','残差混合','模块与 PP 输出'],
      positions:{in:[0,310],bank:[1,310],scores:[2,110],mix:[3,310],attn:[4,110],ffn:[4,310],prefix:[4,510],out:[4,710]},
    },
    overview:{
      stages:['输入','残差入口','Attention 二选一','FFN 二选一','层间与模型输出'],
      positions:{input:[0,310],res1:[1,310],kda:[2,110],attn:[2,510],res2:[3,310],moe:[3,510],dense:[3,710],next:[4,510],output:[4,710],draft:[0,710]},
    },
  };
  function layout(graph,family){
    const spec=layouts[family]||layouts.overview;
    for(const n of graph.nodes){const at=spec.positions[n.id];if(!at)throw Error('Missing diagram position: '+n.id);n.x=36+at[0]*330;n.y=at[1];n.w=258;n.h=118;n.stage=at[0];}
    graph.width=1650;graph.height=Math.max(...graph.nodes.map(n=>n.y+n.h))+58;
    graph.stages=spec.stages.map((title,i)=>({title,index:i,x:20+i*330,y:18,w:290,h:graph.height-36}));
    routeGraph(graph);
    return graph;
  }
  function routeGraph(g){
    const byId=Object.fromEntries(g.nodes.map(n=>[n.id,n]));
    const blockers=g.nodes.map(n=>({x:n.x-10,y:n.y-10,w:n.w+20,h:n.h+20,id:n.id}));
    const ports=[];
    for(const e of g.edges){
      const a=byId[e.src],b=byId[e.dst];
      let from,to;
      if(a.stage===b.stage){const down=b.y>a.y;from={x:a.x+a.w/2,y:down?a.y+a.h:a.y,dir:down?'s':'n'};to={x:b.x+b.w/2,y:down?b.y:b.y+b.h,dir:down?'n':'s'};}
      else{const right=b.x>a.x;from={x:right?a.x+a.w:a.x,y:a.y+a.h/2,dir:right?'e':'w'};to={x:right?b.x:b.x+b.w,y:b.y+b.h/2,dir:right?'w':'e'};}
      ports.push({edge:e,from,to});
    }
    // Spread connections along the card boundary to expose forks and merges.
    const slots=new Map();
    for(const p of ports)for(const [side,id] of [['from',p.edge.src],['to',p.edge.dst]]){const k=id+':'+p[side].dir;if(!slots.has(k))slots.set(k,[]);slots.get(k).push(p[side]);}
    for(const group of slots.values())group.forEach((p,i)=>{const delta=(i-(group.length-1)/2)*14;if(p.dir==='n'||p.dir==='s')p.x+=delta;else p.y+=delta;});
    const outer=p=>({x:p.x+(p.dir==='e'?18:p.dir==='w'?-18:0),y:p.y+(p.dir==='s'?18:p.dir==='n'?-18:0)});
    const xs=new Set([8,g.width-8]),ys=new Set([86,g.height-18]);
    for(const n of g.nodes){xs.add(n.x-22);xs.add(n.x+n.w+22);xs.add(n.x+n.w+36);ys.add(n.y-26);ys.add(n.y+n.h+26);}
    for(const p of ports){p.start=outer(p.from);p.end=outer(p.to);xs.add(p.start.x);xs.add(p.end.x);ys.add(p.start.y);ys.add(p.end.y);}
    const X=[...xs].sort((a,b)=>a-b),Y=[...ys].sort((a,b)=>a-b);
    const clear=(a,b)=>!blockers.some(r=>a.x===b.x?
      a.x>r.x&&a.x<r.x+r.w&&Math.max(a.y,b.y)>r.y&&Math.min(a.y,b.y)<r.y+r.h:
      a.y>r.y&&a.y<r.y+r.h&&Math.max(a.x,b.x)>r.x&&Math.min(a.x,b.x)<r.x+r.w);
    const segmentCache=new Map();
    const clearIds=(a,b)=>{const key=a<b?a+':'+b:b+':'+a;if(!segmentCache.has(key))segmentCache.set(key,clear({x:X[a%X.length],y:Y[Math.floor(a/X.length)]},{x:X[b%X.length],y:Y[Math.floor(b/X.length)]}));return segmentCache.get(key);};
    const occupied=new Map();
    for(const p of ports){
      const start=Y.indexOf(p.start.y)*X.length+X.indexOf(p.start.x),end=Y.indexOf(p.end.y)*X.length+X.indexOf(p.end.x);
      const queue=[],best=new Map([[start,0]]),parent=new Map(),direction=new Map();
      const estimate=id=>Math.abs(X[id%X.length]-p.end.x)+Math.abs(Y[Math.floor(id/X.length)]-p.end.y);
      const push=item=>{queue.push(item);let i=queue.length-1;while(i){const j=(i-1)>>1;if(queue[j].f<=item.f)break;queue[i]=queue[j];i=j;}queue[i]=item;};
      const pop=()=>{const first=queue[0],last=queue.pop();if(queue.length){let i=0;while(i*2+1<queue.length){let j=i*2+1;if(j+1<queue.length&&queue[j+1].f<queue[j].f)j++;if(queue[j].f>=last.f)break;queue[i]=queue[j];i=j;}queue[i]=last;}return first;};
      push({id:start,cost:0,f:estimate(start)});
      while(queue.length){const cur=pop();if(cur.cost!==best.get(cur.id))continue;if(cur.id===end)break;const ix=cur.id%X.length,iy=Math.floor(cur.id/X.length);
        for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const nx=ix+dx,ny=iy+dy;if(nx<0||nx>=X.length||ny<0||ny>=Y.length)continue;const id=ny*X.length+nx;if(!clearIds(cur.id,id))continue;
          const segment=cur.id<id?cur.id+':'+id:id+':'+cur.id;
          const dir=dx?'h':'v',cost=cur.cost+(Math.abs(X[nx]-X[ix])+Math.abs(Y[ny]-Y[iy]))*(1+(occupied.get(segment)||0)*2)+(direction.has(cur.id)&&direction.get(cur.id)!==dir?22:0);
          if(cost>=(best.get(id)??Infinity))continue;best.set(id,cost);parent.set(id,cur.id);direction.set(id,dir);push({id,cost,f:cost+estimate(id)});
        }
      }
      if(!best.has(end))throw Error('Cannot route edge '+p.edge.id);
      const points=[];for(let id=end;id!==undefined;id=parent.get(id)){points.push({x:X[id%X.length],y:Y[Math.floor(id/X.length)]});if(id===start)break;const prev=parent.get(id),segment=prev<id?prev+':'+id:id+':'+prev;occupied.set(segment,(occupied.get(segment)||0)+1);}
      points.reverse();points.unshift({x:p.from.x,y:p.from.y});points.push({x:p.to.x,y:p.to.y});
      p.edge.points=points.filter((pt,i)=>i===0||i===points.length-1||!(points[i-1].x===pt.x&&pt.x===points[i+1].x||points[i-1].y===pt.y&&pt.y===points[i+1].y));
      p.edge.backedge=byId[p.edge.src].stage>byId[p.edge.dst].stage;
    }
  }
  function roundedPath(points,r=10){
    let d=`M${points[0].x} ${points[0].y}`;
    for(let i=1;i<points.length-1;i++){const a=points[i-1],b=points[i],c=points[i+1];const r1=Math.min(r,(Math.abs(b.x-a.x)+Math.abs(b.y-a.y))/2,(Math.abs(c.x-b.x)+Math.abs(c.y-b.y))/2);const pre={x:b.x-Math.sign(b.x-a.x)*r1,y:b.y-Math.sign(b.y-a.y)*r1};const post={x:b.x+Math.sign(c.x-b.x)*r1,y:b.y+Math.sign(c.y-b.y)*r1};d+=` L${pre.x} ${pre.y} Q${b.x} ${b.y} ${post.x} ${post.y}`;}
    return d+` L${points.at(-1).x} ${points.at(-1).y}`;
  }
  function customLayout(graph,titles){
    for(const n of graph.nodes){n.x=36+n.col*300;n.y=(graph.resourceLanes?90:110)+n.row*(graph.resourceLanes?150:180);n.w=242;n.h=graph.resourceLanes?106:112;n.stage=n.col;}
    graph.width=titles.length*300;graph.height=Math.max(...graph.nodes.map(n=>n.y+n.h))+58;
    graph.stages=titles.map((title,i)=>({title,index:i,x:20+i*300,y:18,w:270,h:graph.height-36}));
    routeGraph(graph);return graph;
  }
  const api={layout,roundedPath,customLayout};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.KimiDiagram=api;
})(typeof window==='undefined'?globalThis:window);
