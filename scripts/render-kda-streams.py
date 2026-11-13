"""Render measured KDA execution streams and a dependency-only candidate plan."""
import json,math,html
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data/kda-streams-20261113.json').read_text(encoding='utf-8'))
OUT=ROOT/'docs/images'
BG='#f8fafc';FG='#172b42';MUTED='#526579';GRID='#d9e2ec'
CUBE='#dbeafe';VEC='#ccfbf1';COMM='#ffedd5';PURPLE='#7756b7';RED='#b5473c'
FONT=Path('C:/Windows/Fonts/msyh.ttc')
if not FONT.exists():FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')

class Figure:
 def __init__(self,name,w,h):
  self.name=name;self.w=w;self.h=h;self.scale=2
  self.im=Image.new('RGB',(w*2,h*2),BG);self.d=ImageDraw.Draw(self.im)
  self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>']
 def text(self,x,y,text,size=18,fill=FG,center=False):
  font=ImageFont.truetype(str(FONT),size*2)
  lines=text.split('\n')
  for i,line in enumerate(lines):
   yy=y+i*(size+5);xx=x
   if center:xx-=self.d.textlength(line,font=font)/4
   self.d.text((xx*2,yy*2),line,font=font,fill=fill)
   anchor='middle' if center else 'start'
   self.svg.append(f'<text x="{x}" y="{yy+size}" text-anchor="{anchor}" font-family="Microsoft YaHei,DejaVu Sans,sans-serif" font-size="{size}" fill="{fill}">{html.escape(line)}</text>')
 def rect(self,x,y,w,h,fill=None,outline=None,r=8,dash=False):
  coords=tuple(round(v*2) for v in (x,y,x+w,y+h))
  if not dash:self.d.rounded_rectangle(coords,radius=r*2,fill=fill,outline=outline,width=2)
  else:
   for a,b in [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]:self.line([a,b],outline,dash=True)
  self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or "none"}" stroke="{outline or "none"}" stroke-width="1.5"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 def line(self,points,fill=GRID,width=1,dash=False,arrow=False):
  for a,b in zip(points,points[1:]):
   if dash:
    distance=math.dist(a,b)
    for i in range(0,math.ceil(distance),12):
     t=i/max(distance,1);u=min(i+7,distance)/max(distance,1)
     p=(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
     q=(a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u)
     self.d.line([(p[0]*2,p[1]*2),(q[0]*2,q[1]*2)],fill=fill,width=width*2)
   else:self.d.line([(a[0]*2,a[1]*2),(b[0]*2,b[1]*2)],fill=fill,width=width*2)
  self.svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{fill}" stroke-width="{width}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
  if arrow:
   a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);sz=8
   p=[b,(b[0]-sz*math.cos(angle-.5),b[1]-sz*math.sin(angle-.5)),(b[0]-sz*math.cos(angle+.5),b[1]-sz*math.sin(angle+.5))]
   self.d.polygon([(x*2,y*2) for x,y in p],fill=fill)
   self.svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in p)+f'" fill="{fill}"/>')
 def node(self,x,y,w,label,fill,h=62,size=17):
  self.rect(x,y,w,h,fill)
  lines=label.split('\n');self.text(x+w/2,y+(h-len(lines)*(size+5))/2,label,size,center=True)
 def save(self):
  self.svg.append('</svg>');OUT.mkdir(parents=True,exist_ok=True)
  (OUT/(self.name+'.svg')).write_text('\n'.join(self.svg),encoding='utf-8')
  self.im.save(OUT/(self.name+'.png'))

def measured():
 f=Figure('a5-kda-current-20261113',1300,535)
 f.text(28,17,'当前 A5：QKV / beta 合并返回，接收后的 Vec 仍在主流排队',26)
 f.text(28,57,'真实 PD · DP32/TP1/localTP8 · C128 · 128k→1k · DSpark3 / 2.7 · 首个 KDA 层实测',17,fill=MUTED)
 left=175;scale=12.6
 cube=next(r for r in DATA['window'] if r['type']=='QuantBatchMatmulV3')
 origin=cube['start_us'];rows=[r for r in DATA['window'] if r['start_us']>=origin and r['type'] not in ['RecurrentKda']]
 get=lambda op:next(r for r in rows if r['type']==op)
 cat=get('ConcatD');zero=get('ZerosLike')
 comm=next(r for r in rows if r['type']=='hcom_alltoall_' and r['start_us']<get('CausalConv1d')['start_us'])
 first=get('AsStrided');last=get('Sigmoid');conv=get('CausalConv1d')
 x=lambda t:left+(t-origin)*scale
 for tick in range(0,81,10):
  xx=left+tick*scale;f.line([(xx,116),(xx,408)],GRID);f.text(xx,91,str(tick),15,fill=MUTED,center=True)
 f.text(1210,91,'μs',15,fill=MUTED)
 for y,title,sub in [(152,'主流 425','Cube / Vec 串行'),(264,'aux 422','持久 F / beta 流'),(370,'通信引擎','CCU 设备区间')]:
  f.line([(left,y+25),(1220,y+25)],GRID)
  f.text(23,y-2,title,18);f.text(23,y+24,sub,14,fill=MUTED)
 f.node(x(cube['start_us']),145,cube['duration_us']*scale,'QKV GEMM\n23.78 μs',CUBE)
 f.node(x(cat['start_us']),145,cat['duration_us']*scale,'Cat',VEC,size=16)
 f.node(x(zero['start_us']),145,zero['duration_us']*scale,'Zero',VEC,size=14)
 f.node(x(first['start_us']),145,(last['start_us']+last['duration_us']-first['start_us'])*scale,'QKV 重排 + β后处理\n17.21 μs',VEC)
 f.node(x(conv['start_us']),145,conv['duration_us']*scale,'Conv1d\n15.13 μs',VEC)
 f.text(760,256,'beta BF16 已完成；此窗口内 aux 空闲',17,fill=MUTED,center=True)
 f.node(x(comm['start_us']),350,comm['duration_us']*scale,'QKV + beta\nA2A 10.49 μs',COMM)
 overlap0=max(comm['start_us'],zero['start_us']);overlap1=min(comm['start_us']+comm['duration_us'],zero['start_us']+zero['duration_us'])
 f.rect(x(overlap0),136,(overlap1-overlap0)*scale,283,outline=PURPLE,dash=True,r=0)
 f.text(460,427,f'只有清零与 A2A 重叠 {overlap1-overlap0:.2f} μs',17,fill=PURPLE,center=True)
 f.line([(x(comm['start_us']+comm['duration_us']),350),(x(comm['start_us']+comm['duration_us']),225),(x(first['start_us']),225),(x(first['start_us']),208)],MUTED,2,arrow=True)
 f.text(1040,239,'接收完成后才能重排 QKV',17,fill=RED,center=True)
 f.text(28,472,'690 个层样本中位数：QKV GEMM 23.25 μs  ·  A2A 10.63 μs  ·  接收后 Vec 17.45 μs',18)
 f.text(28,506,'前序 AG∥F、MX quant∥beta 与 G 返回已有并行；后续 RecurrentKDA → G 重排 → Norm → O 仍在主流顺序执行。',15,fill=MUTED)
 f.save()

def proposed():
 f=Figure('a5-kda-three-streams-plan-20261113',1300,670)
 f.text(28,17,'下一步：beta 提前返回，复用 aux 做 Vec，形成 Cube / Vector / CCU 三流',26)
 f.text(28,57,'待实现方案 · 横向距离不代表时长 · QKV 接收重排仍须等其 A2A · 不承诺全部隐藏或端到端收益',17,fill=MUTED)
 for y,title,sub in [(178,'S_main','主计算 / 消费者'),(352,'S_aux','复用持久 F / beta 流'),(510,'S_comm','有序 CCU 提交')]:
  f.line([(174,y),(1270,y)],GRID);f.text(24,y-22,title,18);f.text(24,y+7,sub,14,fill=MUTED)
 f.rect(366,126,307,445,outline=PURPLE,dash=True,r=12)
 f.text(520,94,'候选三流窗口：保留真实数据依赖',18,fill=PURPLE,center=True)
 f.node(180,147,126,'MX input\nquant',VEC)
 f.node(390,147,238,'QKV GEMM\nAI_CORE / Cube',CUBE)
 f.node(760,147,115,'QKV 重排\n双条件 join',VEC)
 f.node(900,147,100,'Conv1d',VEC)
 f.node(1030,147,110,'Recurrent\nKDA',VEC)
 f.node(1160,147,102,'Norm\nG 已就绪',VEC)
 f.node(180,321,126,'beta BF16\nGEMM',CUBE)
 f.node(390,321,118,'清零 +\nG owner 重排',VEC)
 f.node(530,321,130,'beta owner\n转置 / cast / σ',VEC)
 f.node(180,479,126,'既有 G\nA2A',COMM)
 f.node(390,479,118,'提前 beta\nA2A · BF16',COMM)
 f.node(690,479,115,'QKV\nA2A',COMM)
 for a,b in [((306,178),(390,178)),((875,178),(900,178)),((1000,178),(1030,178)),((1140,178),(1160,178)),((508,352),(530,352))]:f.line([a,b],MUTED,2,arrow=True)
 f.line([(306,352),(342,352),(342,225),(390,205)],MUTED,2,arrow=True)
 f.text(311,294,'betaRawReady',14,fill=MUTED)
 f.line([(306,352),(340,352),(340,510),(390,510)],MUTED,2,arrow=True)
 f.line([(242,211),(242,275),(395,275),(395,319)],MUTED,1,arrow=True)
 f.text(185,239,'quantDone / V 交接',14,fill=MUTED)
 f.line([(508,510),(548,510),(548,385)],PURPLE,2,arrow=True)
 f.text(550,414,'β 接收完成',14,fill=PURPLE)
 f.line([(306,510),(324,510),(324,428),(410,428),(410,385)],MUTED,2,arrow=True)
 f.text(183,410,'G 接收完成',14,fill=MUTED)
 f.line([(628,178),(708,178),(708,477)],MUTED,2,arrow=True)
 f.text(642,230,'QKV ready',14,fill=MUTED)
 f.line([(805,479),(819,479),(819,281)],RED,2,arrow=True)
 f.text(825,447,'必须等待',15,fill=RED)
 f.line([(660,352),(738,352),(738,259),(797,259)],PURPLE,2,arrow=True)
 f.rect(798,240,42,40,BG,outline=PURPLE,r=4)
 f.text(819,244,'∧',21,fill=PURPLE,center=True)
 f.line([(819,240),(819,211)],MUTED,2,arrow=True)
 f.text(860,248,'QKV_rx ∧ vecDone',15,fill=PURPLE)
 f.text(900,358,'vecDone：G / β后处理 / 清零完成',15,fill=PURPLE,center=True)
 f.text(28,593,'β A2A 在 QKV GEMM 入队前提交；G 等待 AG 与 F_b；Vector 段交接，避免 V / V 抢占。',18)
 f.text(28,628,'删除拼接 Cat 与 β拆包 Slice；多 1 次小 collective / KDA层，总 payload 不变。QKV 重排仍在关键路径。',16,fill=MUTED)
 f.save()

if __name__=='__main__':measured();proposed();print('Rendered measured and proposed SVG + PNG figures')
