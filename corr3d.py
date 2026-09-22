"""3D exposure field for a fund's holdings — a live force layout that snaps to axes.

Every holding is a dot pulled in three directions at once: toward its SECTOR
anchor, toward its REGION anchor, and toward the RETURN pole (strength set by how
tightly it tracks the rest of the book). Where a dot settles is a read on which
exposure actually dominates it. Each fund also gets a centroid dot — its weighted
centre of mass — that trails the cloud as it moves.

Toggling to SNAP morphs the same dots onto readable axes:
    X = region · Y = sector · Z = correlation to the book.

Rendered on a plain 2D canvas with hand-rolled perspective projection — no CDN,
no WebGL, nothing that can break a deploy.
"""
from __future__ import annotations

import json

SECTOR_COLORS = [
    "#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#f472b6",
    "#2dd4bf", "#fb923c", "#a3e635", "#60a5fa", "#e879f9", "#94a3b8",
]
FUND_COLORS = ["#FF8200", "#22d3ee", "#c084fc", "#facc15", "#4ade80"]

_TEMPLATE = r"""
<meta charset="utf-8">
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="hud">
    <button class="btn on" id="bMode">PULL</button>
    <button class="btn on" id="bLinks">LINKS</button>
    <button class="btn on" id="bCent">FUNDS</button>
    <button class="btn on" id="bPoles">POLES</button>
    <button class="btn on" id="bAxes">AXES</button>
    <button class="btn" id="bSpin">SPIN</button>
    <button class="btn" id="bReheat">REHEAT</button>
  </div>
  <div id="legend"></div>
  <div id="tip"></div>
  <div id="hint">drag to orbit · scroll to zoom · click a legend row to hide it ·
       double-click to isolate</div>
</div>
<style>
  html,body{margin:0;padding:0;overflow:hidden;background:__BG__;}
  #wrap{position:relative;width:100%;height:__H__px;background:__BG__;
        border:1px solid __LINE__;border-radius:10px;overflow:hidden;
        font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
  #cv{display:block;width:100%;height:100%;cursor:grab;}
  #cv.drag{cursor:grabbing;}
  #hud{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap;}
  .btn{background:__SURF__;color:__DIM__;border:1px solid __LINE__;border-radius:6px;
       padding:5px 10px;font:600 10px/1 ui-monospace,Menlo,Consolas,monospace;
       letter-spacing:.09em;cursor:pointer;transition:all .15s;}
  .btn:hover{border-color:__ACCENT__;color:__TXT__;}
  .btn.on{background:__ACCENT__;color:#0B0E17;border-color:__ACCENT__;}
  #legend{position:absolute;top:10px;right:10px;display:flex;flex-direction:column;
          gap:3px;background:__SURF__cc;border:1px solid __LINE__;border-radius:8px;
          padding:8px 10px;max-height:calc(100% - 90px);overflow:auto;}
  .lg{display:flex;align-items:center;gap:6px;font-size:10px;color:__DIM__;
      white-space:nowrap;cursor:pointer;user-select:none;padding:1px 3px;
      border-radius:4px;transition:opacity .12s,background .12s;}
  .lg:hover{background:__LINE__66;color:__TXT__;}
  .lg.off{opacity:.32;}
  .lg.off .sw{background:transparent!important;box-shadow:inset 0 0 0 1.5px __DIM__;}
  .sw{width:8px;height:8px;border-radius:2px;flex:0 0 8px;}
  .lghdr{display:flex;align-items:center;justify-content:space-between;gap:8px;
         font-size:8.5px;letter-spacing:.12em;color:__DIM__;opacity:.7;
         margin:5px 0 2px;}
  .lghdr span{cursor:pointer;text-decoration:underline;}
  .lghdr span:hover{color:__ACCENT__;}
  #tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;
       background:__SURF__f2;border:1px solid __ACCENT__;border-radius:8px;
       padding:8px 10px;font-size:11px;color:__TXT__;max-width:240px;z-index:5;
       box-shadow:0 8px 24px rgba(0,0,0,.45);}
  #tip b{color:__ACCENT__;}
  #tip .r{color:__DIM__;font-size:10px;}
  #hint{position:absolute;bottom:8px;left:12px;font-size:9.5px;color:__DIM__;
        letter-spacing:.06em;opacity:.65;}
</style>
<script>
(function(){
const D = __PAYLOAD__;
const SC = __SECTOR_COLORS__, FC = __FUND_COLORS__;
const TXT="__TXT__", DIM="__DIM__", LINE="__LINE__", ACC="__ACCENT__", BG="__BG__";
const AX_X="#38bdf8", AX_Y="#f472b6", AX_Z=ACC;   // region / sector / ρ axes
if(!D || !D.nodes || D.nodes.length < 2) return;

const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
const tip=document.getElementById("tip"), wrap=document.getElementById("wrap");
let W=0,H=0,DPR=Math.min(window.devicePixelRatio||1,2);

function resize(){
  const r=wrap.getBoundingClientRect(); W=r.width; H=r.height;
  cv.width=W*DPR; cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0);
}
new ResizeObserver(resize).observe(wrap); resize();

/* ── data ─────────────────────────────────────────────────────────────── */
const secIdx={}; D.sectors.forEach((s,i)=>secIdx[s]=i);
const N = D.nodes.map((n,i)=>{
  const a=Math.random()*6.283, b=Math.acos(2*Math.random()-1), rr=0.35+Math.random()*0.25;
  return {...n, i,
    x:rr*Math.sin(b)*Math.cos(a), y:rr*Math.sin(b)*Math.sin(a), z:rr*Math.cos(b),
    vx:0, vy:0, vz:0,
    m: 0.55 + 2.2*n.wr,                       // big positions are heavy
    col: SC[secIdx[n.sector] % SC.length],
    dx:0, dy:0, dz:0, px:0, py:0, ps:0, pz:0};
});
const L = D.links || [];
const FUNDS = Object.keys(D.funds||{}).map((f,i)=>({
  name:f, mem:D.funds[f], col:FC[i%FC.length], trail:[], x:0,y:0,z:0}));

/* ── visibility filters ───────────────────────────────────────────────────
   Hiding is PURELY visual: hidden holdings still take part in the physics and
   still count toward each fund's centre of mass, so the layout never shifts
   underneath you — you're isolating a slice of the same field, not recomputing
   a different one. */
const hidS=new Set(), hidR=new Set(), hidF=new Set();
const shown = n => !hidS.has(n.sector) && !hidR.has(n.region);

const lg=document.getElementById("legend");
const rows=[];

function section(title, items, colorOf, hideSet, round){
  const h=document.createElement("div"); h.className="lghdr";
  h.innerHTML=title+' <span data-a="all">all</span>';
  lg.appendChild(h);
  h.querySelector("[data-a=all]").onclick=e=>{
    e.stopPropagation(); hideSet.clear(); sync();
  };
  items.forEach((it,i)=>{
    const d=document.createElement("div"); d.className="lg";
    d.innerHTML='<span class="sw" style="background:'+colorOf(it,i)+
                (round?';border-radius:50%':'')+'"></span>'+it;
    d.title="click to hide · double-click to isolate";
    d.onclick=()=>{ hideSet.has(it)?hideSet.delete(it):hideSet.add(it); sync(); };
    d.ondblclick=()=>{                       // solo this one (or restore all)
      const only = hideSet.size===items.length-1 && !hideSet.has(it);
      hideSet.clear();
      if(!only) items.forEach(o=>{ if(o!==it) hideSet.add(o); });
      sync();
    };
    rows.push([d,it,hideSet]); lg.appendChild(d);
  });
}
function sync(){ rows.forEach(([d,it,set])=>d.classList.toggle("off",set.has(it))); }

section("SECTORS", D.sectors, (s,i)=>SC[i%SC.length], hidS, false);
section("REGIONS", D.regions, ()=>DIM, hidR, false);
if(FUNDS.length) section("FUNDS", FUNDS.map(f=>f.name),
                         (n,i)=>FUNDS[i].col, hidF, true);
sync();

/* ── camera ───────────────────────────────────────────────────────────── */
let theta=0.62, phi=0.30, dist=6.6, focal=0.86, spin=false;
const AS=1.60;                  // snap-cube scale, to match the pull field's size
// poles = the PULL-mode diamonds and category anchor labels
// axes  = the SNAP-mode cube edges and axis labels
// Independent, so you can strip one set without losing the other.
let mode=0, blend=0, showLinks=true, showCent=true, poles=true, axes=true, heat=1.0;

function project(x,y,z){
  const ct=Math.cos(theta), st=Math.sin(theta);
  let X = x*ct - z*st, Z = x*st + z*ct;
  const cp=Math.cos(phi), sp=Math.sin(phi);
  let Y = y*cp - Z*sp; Z = y*sp + Z*cp;
  let zc = Z + dist; if(zc<0.25) zc=0.25;
  const f = (Math.min(W,H)*focal)/zc;
  return [W/2 + X*f, H/2 - Y*f, f, zc];
}

/* ── physics: three competing pulls ───────────────────────────────────── */
const RP = D.poles.return;
function step(){
  const dt=1/60;
  for(let i=0;i<N.length;i++){
    const n=N[i];
    // One spring, toward the holding's ternary rest point: its sector anchor,
    // region anchor and the return pole mixed by how much each explains it.
    // Three separate springs would cancel and pile everything in the middle.
    let ax=3.1*(n.bx-n.x), ay=3.1*(n.by-n.y), az=3.1*(n.bz-n.z);
    n.ax=ax; n.ay=ay; n.az=az;
  }
  // peer attraction — deliberately faint. It gives correlated names a bit of
  // organic clustering, but must not overpower the ternary rest point, or a
  // dot's position stops meaning what the legend says it means.
  for(let k=0;k<L.length;k++){
    const a=N[L[k].s], b=N[L[k].t], g=0.18*(L[k].r-0.15);
    const dx=b.x-a.x, dy=b.y-a.y, dz=b.z-a.z;
    a.ax+=g*dx; a.ay+=g*dy; a.az+=g*dz;
    b.ax-=g*dx; b.ay-=g*dy; b.az-=g*dz;
  }
  // short-range repulsion so dots stay legible
  for(let i=0;i<N.length;i++){
    for(let j=i+1;j<N.length;j++){
      const a=N[i], b=N[j];
      let dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
      let d2=dx*dx+dy*dy+dz*dz; if(d2<0.0064) d2=0.0064;
      if(d2>0.62) continue;
      const d=Math.sqrt(d2), f=0.055/d2;
      a.ax+=f*dx/d; a.ay+=f*dy/d; a.az+=f*dz/d;
      b.ax-=f*dx/d; b.ay-=f*dy/d; b.az-=f*dz/d;
    }
  }
  for(let i=0;i<N.length;i++){
    const n=N[i], im=1/n.m;
    n.vx=(n.vx+n.ax*im*dt)*0.90; n.vy=(n.vy+n.ay*im*dt)*0.90; n.vz=(n.vz+n.az*im*dt)*0.90;
    const j=0.00035*heat;        // faint brownian breath so it never looks frozen
    n.x+=n.vx+(Math.random()-0.5)*j;
    n.y+=n.vy+(Math.random()-0.5)*j;
    n.z+=n.vz+(Math.random()-0.5)*j;
  }
  if(heat>1) heat*=0.985;
}

/* ── draw ─────────────────────────────────────────────────────────────── */
const ease=t=>t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;

function axisLabel(x,y,z,txt,al,size,alpha){
  const p=project(x,y,z);
  ctx.globalAlpha=alpha; ctx.fillStyle=DIM;
  ctx.font="600 "+size+"px ui-monospace,Menlo,Consolas,monospace";
  ctx.textAlign=al||"center"; ctx.textBaseline="middle";
  ctx.fillText(txt,p[0],p[1]); ctx.globalAlpha=1;
}
function line3(a,b,col,wdt,alpha){
  const p=project(a[0],a[1],a[2]), q=project(b[0],b[1],b[2]);
  ctx.globalAlpha=alpha; ctx.strokeStyle=col; ctx.lineWidth=wdt;
  ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
  ctx.globalAlpha=1;
}

/* A bold, arrow-headed, titled axis. The head and label are built in SCREEN
   space from the projected direction, so they stay correctly oriented at any
   camera angle instead of skewing with the projection. */
function arrow3(a,b,col,alpha,label){
  const p=project(a[0],a[1],a[2]), q=project(b[0],b[1],b[2]);
  const dx=q[0]-p[0], dy=q[1]-p[1], L=Math.hypot(dx,dy)||1;
  const ux=dx/L, uy=dy/L, hs=10;
  ctx.globalAlpha=alpha; ctx.strokeStyle=col; ctx.lineWidth=2.1;
  ctx.lineCap="round";
  ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0]-ux*hs,q[1]-uy*hs);
  ctx.stroke(); ctx.lineCap="butt";
  ctx.beginPath(); ctx.moveTo(q[0],q[1]);
  ctx.lineTo(q[0]-ux*hs-uy*hs*0.44, q[1]-uy*hs+ux*hs*0.44);
  ctx.lineTo(q[0]-ux*hs+uy*hs*0.44, q[1]-uy*hs-ux*hs*0.44);
  ctx.closePath(); ctx.fillStyle=col; ctx.fill();
  if(label){
    const lx=q[0]+ux*20, ly=q[1]+uy*20;
    ctx.font="700 10px ui-monospace,Menlo,Consolas,monospace";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.strokeStyle=BG; ctx.lineWidth=4; ctx.lineJoin="round";
    ctx.strokeText(label,lx,ly);
    ctx.fillStyle=col; ctx.fillText(label,lx,ly);
  }
  ctx.globalAlpha=1;
}

function draw(){
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const t=ease(blend), u=1-t;

  // positions morph between the live sim and the axis cube
  for(const n of N){
    n.dx=n.x*u+n.ax_t*t; n.dy=n.y*u+n.ay_t*t; n.dz=n.z*u+n.az_t*t;
    const p=project(n.dx,n.dy,n.dz);
    n.px=p[0]; n.py=p[1]; n.ps=p[2]; n.pz=p[3];
  }

  // ── PULL scaffolding: the three poles ──
  if(u>0.02 && poles){
    const P=D.poles;
    [["sector",P.sector],["region",P.region],["return",P.return]].forEach(([k,p])=>{
      const q=project(p[0],p[1],p[2]);
      ctx.globalAlpha=u*0.9;
      ctx.fillStyle=ACC; ctx.beginPath();
      ctx.moveTo(q[0],q[1]-7); ctx.lineTo(q[0]+7,q[1]);
      ctx.lineTo(q[0],q[1]+7); ctx.lineTo(q[0]-7,q[1]); ctx.closePath(); ctx.fill();
      ctx.fillStyle=ACC; ctx.font="700 11px ui-monospace,Menlo,Consolas,monospace";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(k.toUpperCase(), q[0], q[1]-18);
      ctx.globalAlpha=1;
    });
    for(const s in D.sectorAnchors)
      if(!hidS.has(s)) axisLabel(...D.sectorAnchors[s],s,"center",9,u*0.5);
    for(const r in D.regionAnchors)
      if(!hidR.has(r)) axisLabel(...D.regionAnchors[r],r,"center",9,u*0.5);
  }

  // ── SNAP scaffolding: the axis cube ──
  if(t>0.02 && axes){
    const c=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
             [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]].map(v=>v.map(k=>k*AS));
    const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    E.forEach(e=>line3(c[e[0]],c[e[1]],LINE,1,t*0.30));   // cage stays faint…

    const rN=D.regions.length, sN=D.sectors.length;
    const rx=i=>(-1+2*(i+0.5)/rN)*AS, sy=i=>(-1+2*(i+0.5)/sN)*AS;

    // …category gridlines carry the structure instead
    D.regions.forEach((r,i)=>{ if(hidR.has(r)) return;
      line3([rx(i),-AS,-AS],[rx(i),-AS,AS],AX_X,1,t*0.22);
      line3([rx(i),-AS,-AS],[rx(i), AS,-AS],AX_X,1,t*0.13);
      line3([rx(i),-AS,-AS],[rx(i),-1.05*AS,-AS],AX_X,1.6,t*0.75);   // tick
    });
    D.sectors.forEach((s,i)=>{ if(hidS.has(s)) return;
      line3([-AS,sy(i),-AS],[-AS,sy(i),AS],AX_Y,1,t*0.22);
      line3([-AS,sy(i),-AS],[ AS,sy(i),-AS],AX_Y,1,t*0.13);
      line3([-AS,sy(i),-AS],[-1.05*AS,sy(i),-AS],AX_Y,1.6,t*0.75);
    });

    // three bold, titled axes off the origin corner
    const O=[-AS,-AS,-AS];
    arrow3(O,[ 1.10*AS,-AS,-AS],AX_X,t*0.95,"REGION");
    arrow3(O,[-AS, 1.06*AS,-AS],AX_Y,t*0.95,"SECTOR");
    arrow3(O,[-AS,-AS, 1.10*AS],AX_Z,t*0.95,"ρ TO BOOK");

    D.regions.forEach((r,i)=>{ if(!hidR.has(r))
      axisLabel(rx(i),-1.17*AS,-1.06*AS,r,"center",10,t*0.95); });
    D.sectors.forEach((s,i)=>{ if(!hidS.has(s))
      axisLabel(-1.10*AS,sy(i),-1.06*AS,s,"right",9,t*0.85); });
    // az = -1 + 2*r01, so the LOW end of the ρ range sits at z = -1
    axisLabel(-1.14*AS,-1.14*AS,-AS,D.rhoRange[0].toFixed(2),"center",9,t*0.85);
    axisLabel(-1.14*AS,-1.14*AS, AS,D.rhoRange[1].toFixed(2),"center",9,t*0.85);
  }

  // ── correlation links ──
  if(showLinks){
    for(let k=0;k<L.length;k++){
      const a=N[L[k].s], b=N[L[k].t];
      if(!shown(a) || !shown(b)) continue;   // no links dangling off hidden dots
      const al=Math.max(0,(L[k].r-0.15))*0.5*(u*0.85+0.15);
      if(al<=0.012) continue;
      ctx.globalAlpha=al; ctx.strokeStyle=a.col; ctx.lineWidth=0.7;
      ctx.beginPath(); ctx.moveTo(a.px,a.py); ctx.lineTo(b.px,b.py); ctx.stroke();
    }
    ctx.globalAlpha=1;
  }

  // ── holdings, painter's algorithm ──
  const order=N.filter(shown).sort((a,b)=>b.pz-a.pz);
  for(const n of order){
    const r=Math.max(2.4,(0.022+0.055*Math.sqrt(n.wr))*n.ps);
    const fog=Math.max(0.30,Math.min(1,1.65-n.pz/dist));
    if(n===hover){
      ctx.globalAlpha=0.9; ctx.strokeStyle="#fff"; ctx.lineWidth=1.6;
      ctx.beginPath(); ctx.arc(n.px,n.py,r+5,0,6.283); ctx.stroke();
    }
    const g=ctx.createRadialGradient(n.px,n.py,0,n.px,n.py,r*2.6);
    g.addColorStop(0,n.col); g.addColorStop(0.42,n.col+"66");
    g.addColorStop(1,n.col+"00");
    ctx.globalAlpha=fog*0.55; ctx.fillStyle=g;
    ctx.beginPath(); ctx.arc(n.px,n.py,r*2.6,0,6.283); ctx.fill();
    ctx.globalAlpha=fog; ctx.fillStyle=n.col;
    ctx.beginPath(); ctx.arc(n.px,n.py,r,0,6.283); ctx.fill();
    ctx.globalAlpha=1;
  }

  // ── fund centroids: weighted centre of mass, with a trail ──
  if(showCent){
    FUNDS.forEach((f,fi)=>{
      if(hidF.has(f.name)){ f.trail.length=0; return; }
      let cx=0,cy=0,cz=0;
      for(const [i,w] of f.mem){ const n=N[i]; cx+=n.dx*w; cy+=n.dy*w; cz+=n.dz*w; }
      f.x=cx; f.y=cy; f.z=cz;
      f.trail.push([cx,cy,cz]); if(f.trail.length>46) f.trail.shift();
      ctx.strokeStyle=f.col; ctx.lineWidth=1.5;
      for(let k=1;k<f.trail.length;k++){
        const p=project(...f.trail[k-1]), q=project(...f.trail[k]);
        ctx.globalAlpha=(k/f.trail.length)*0.42;
        ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
      }
      ctx.globalAlpha=1;
      const p=project(cx,cy,cz), pr=9;
      const g=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],pr*3.2);
      g.addColorStop(0,f.col+"cc"); g.addColorStop(1,f.col+"00");
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(p[0],p[1],pr*3.2,0,6.283); ctx.fill();
      ctx.fillStyle=f.col; ctx.beginPath(); ctx.arc(p[0],p[1],pr*0.52,0,6.283); ctx.fill();
      ctx.strokeStyle=f.col; ctx.lineWidth=1.4; ctx.globalAlpha=0.85;
      ctx.beginPath(); ctx.arc(p[0],p[1],pr,0,6.283); ctx.stroke(); ctx.globalAlpha=1;
      ctx.font="700 10px ui-monospace,Menlo,Consolas,monospace";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      // Well-diversified sleeves land almost on top of each other, so stagger
      // the labels and halo them against the dots behind.
      const ly=p[1]-pr-9-fi*14;
      ctx.strokeStyle=BG; ctx.lineWidth=3.5; ctx.lineJoin="round";
      ctx.strokeText(f.name.toUpperCase(), p[0], ly);
      ctx.fillStyle=f.col;
      ctx.fillText(f.name.toUpperCase(), p[0], ly);
    });
  }
}

/* ── loop ─────────────────────────────────────────────────────────────── */
// step() reuses ax/ay/az for acceleration, so stash the snap targets separately
// (D.nodes is the untouched payload; N holds spread copies).
// Axis coords arrive in [-1,1]; scale them so the snap cube fills roughly the
// same volume as the pull field and the morph doesn't look like a zoom-out.
N.forEach((n,i)=>{ const s=D.nodes[i];
  n.ax_t=s.ax*AS; n.ay_t=s.ay*AS; n.az_t=s.az*AS; });

let dragging=false, lx=0, ly=0, hover=null;   // read by draw(), so declare first

function loop(){
  if(mode===0 || blend>0.001) step();
  const tgt = mode;
  blend += (tgt-blend)*0.075;
  if(Math.abs(tgt-blend)<0.002) blend=tgt;
  if(spin && !dragging) theta += 0.0022;
  draw();
  requestAnimationFrame(loop);
}
loop();

/* ── interaction ──────────────────────────────────────────────────────── */
cv.addEventListener("mousedown",e=>{dragging=true;lx=e.clientX;ly=e.clientY;
                                    cv.classList.add("drag");});
window.addEventListener("mouseup",()=>{dragging=false;cv.classList.remove("drag");});
window.addEventListener("mousemove",e=>{
  // Self-heal a stuck drag: if the button was released outside the frame the
  // mouseup never reaches us, and the view would orbit on every later move.
  if(dragging && e.buttons===0){ dragging=false; cv.classList.remove("drag"); }
  if(dragging){
    theta += (e.clientX-lx)*0.0065;
    phi   = Math.max(-1.45,Math.min(1.45, phi + (e.clientY-ly)*0.0055));
    lx=e.clientX; ly=e.clientY; return;
  }
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if(mx<0||my<0||mx>W||my>H){ hover=null; tip.style.opacity=0; return; }
  let best=null, bd=16*16;
  for(const n of N){
    if(!shown(n)) continue;
    const d=(n.px-mx)*(n.px-mx)+(n.py-my)*(n.py-my);
    if(d<bd){ bd=d; best=n; }
  }
  hover=best;
  if(best){
    const rho = best.rho===null ? "—" : best.rho.toFixed(2);
    const nm = (best.name && best.name!==best.id)
             ? '<span class="r">'+best.name+"</span><br>" : "";
    tip.innerHTML = "<b>"+best.id+"</b><br>"+nm+"<br>"+
      best.sector+" · "+best.region+"<br>"+
      '<span class="r">weight '+(best.w*100).toFixed(2)+"%  ·  ρ to book "+rho+"</span>";
    tip.style.opacity=1;
    tip.style.left=Math.min(W-250,mx+16)+"px";
    tip.style.top=Math.min(H-110,my+14)+"px";
  } else tip.style.opacity=0;
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  dist=Math.max(2.2,Math.min(12,dist*(1+Math.sign(e.deltaY)*0.09)));
},{passive:false});

const bMode=document.getElementById("bMode");
bMode.onclick=()=>{ mode=mode?0:1; bMode.textContent=mode?"SNAP":"PULL";
                    bMode.classList.toggle("on",true);
                    if(mode===0) heat=2.2; };
const bL=document.getElementById("bLinks");
bL.onclick=()=>{ showLinks=!showLinks; bL.classList.toggle("on",showLinks); };
const bC=document.getElementById("bCent");
bC.onclick=()=>{ showCent=!showCent; bC.classList.toggle("on",showCent);
                 FUNDS.forEach(f=>f.trail=[]); };
const bP=document.getElementById("bPoles");
bP.onclick=()=>{ poles=!poles; bP.classList.toggle("on",poles); };
const bA=document.getElementById("bAxes");
bA.onclick=()=>{ axes=!axes; bA.classList.toggle("on",axes); };
const bS=document.getElementById("bSpin");
bS.onclick=()=>{ spin=!spin; bS.classList.toggle("on",spin); };
document.getElementById("bReheat").onclick=()=>{
  heat=3.0;
  N.forEach(n=>{ const a=Math.random()*6.283, b=Math.acos(2*Math.random()-1),
                 rr=0.35+Math.random()*0.3;
    n.x=rr*Math.sin(b)*Math.cos(a); n.y=rr*Math.sin(b)*Math.sin(a); n.z=rr*Math.cos(b);
    n.vx=n.vy=n.vz=0; });
  FUNDS.forEach(f=>f.trail=[]);
};
})();
</script>
"""


def field_html(payload: dict, height: int = 620, dark: bool = True) -> str:
    """Render the 3D exposure field as a standalone HTML/JS block."""
    if not payload or not payload.get("nodes"):
        return "<div style='color:#94a3b8;font:12px monospace'>No 3D field to draw.</div>"
    pal = {
        "__BG__": "#0B0E17" if dark else "#0F172A",
        "__SURF__": "#141B29" if dark else "#1E293B",
        "__LINE__": "#243049" if dark else "#334155",
        "__TXT__": "#E6EAF2",
        "__DIM__": "#94A3B8",
        "__ACCENT__": "#FF8200",
    }
    html = _TEMPLATE
    html = html.replace("__PAYLOAD__", json.dumps(payload).replace("<", r"<"))
    html = html.replace("__SECTOR_COLORS__", json.dumps(SECTOR_COLORS))
    html = html.replace("__FUND_COLORS__", json.dumps(FUND_COLORS))
    html = html.replace("__H__", str(int(height)))
    for k, v in pal.items():
        html = html.replace(k, v)
    return html
