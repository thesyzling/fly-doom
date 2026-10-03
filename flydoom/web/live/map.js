"use strict";
// Anatomical fly anchors and schematic engineered nodes share one picking/edge layer.
class IntegratedMap {
  constructor(onSelect) {
    this.onSelect=onSelect; this.canvas=document.getElementById("atlas"); this.overlay=document.getElementById("mapOverlay");
    this.yaw=0; this.pitch=-.2; this.zoom=1; this.pan={x:0,y:0}; this.filter=0; this.nodes=new Map(); this.selected=null;
    this.gl=this.canvas.getContext("webgl",{alpha:false,antialias:true});
    if(!this.gl) throw new Error("WebGL is unavailable. Use Selected connections or enable browser graphics acceleration.");
    this.initGL(); this.events(); new ResizeObserver(()=>this.draw()).observe(this.overlay.parentElement);
  }
  decode(encoded, Type) { const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));return new Type(bytes.buffer); }
  async load(meta) {
    this.meta=meta;const response=await fetch("/api/map");this.data=await response.json();if(!response.ok)throw new Error(this.data.error);
    this.positions=this.decode(this.data.positions_f32,Float32Array);this.roles=this.decode(this.data.roles_u8,Uint8Array);
    this.counts=new Float32Array(this.data.ids.length);this.lookup=new Map(this.data.ids.map((id,i)=>[id,i]));
    const vertices=new Float32Array(this.data.ids.length*4);
    this.roles.forEach((role,i)=>{vertices.set(this.positions.subarray(i*3,i*3+3),i*4);vertices[i*4+3]=role});
    const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.staticBuffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.countBuffer);gl.bufferData(gl.ARRAY_BUFFER,this.counts,gl.DYNAMIC_DRAW);
    document.getElementById("mapLoading").hidden=true;document.getElementById("mapNote").textContent=`${this.data.ids.length.toLocaleString()} anatomical anchors · engineered cells use a schematic layout`;
    this.draw();
  }
  initGL() {
    const gl=this.gl;const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s};
    this.program=gl.createProgram();gl.attachShader(this.program,shader(gl.VERTEX_SHADER,`
      attribute vec3 pos; attribute float role; attribute float count;
      uniform vec2 viewport; uniform vec2 center; uniform float scale; uniform float yaw; uniform float pitch; uniform float ratio; uniform float filter;
      varying vec4 color;
      void main(){
        float x=cos(yaw)*pos.x+sin(yaw)*pos.z;
        float z=-sin(yaw)*pos.x+cos(yaw)*pos.z;
        float y=cos(pitch)*pos.y-sin(pitch)*z;
        float depth=sin(pitch)*pos.y+cos(pitch)*z;
        vec2 pixel=center+vec2(x,y)*scale*3.0/(3.0+depth);
        gl_Position=vec4(pixel.x/viewport.x*2.0-1.0,1.0-pixel.y/viewport.y*2.0,0.0,1.0);
        color=vec4(.36,.43,.49,.12);gl_PointSize=1.3*ratio;
        if(role>.5&&role<1.5)color=vec4(.15,.49,.46,.28);
        if(role>1.5)color=vec4(.76,.46,.20,.5);
        if(count>0.0){float a=min(1.0,log(1.0+count)/log(9.0));color=vec4(.81,.37,.13,.45+.5*a);gl_PointSize=(1.8+1.6*a)*ratio;}
        bool visible=filter<.5||(filter<1.5&&count>0.0)||(filter>1.5&&filter<2.5&&role>.5&&role<1.5)||(filter>2.5&&role>1.5);
        if(!visible){color.a=0.0;gl_Position=vec4(2.0,2.0,0.0,1.0);}
      }`));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec4 color;void main(){if(distance(gl_PointCoord,vec2(.5))>.5||color.a==0.0)discard;gl_FragColor=color;}`));
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);this.staticBuffer=gl.createBuffer();this.countBuffer=gl.createBuffer();
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  }
  async update(snapshot) {
    this.snapshot=snapshot;if(!this.data)return;if(snapshot.sequence===this.activitySequence){this.draw();return}
    // Never paint the preceding decision's fly activity beside a newer game frame.
    this.counts.fill(0);this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.countBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,this.counts,this.gl.DYNAMIC_DRAW);
    document.getElementById("mapNote").textContent=`Decision ${snapshot.decision} · loading matching fly activity…`;this.draw();
    const requested=snapshot.sequence;const response=await fetch(`/api/map-activity?sequence=${requested}`);const data=await response.json();
    if(this.snapshot.sequence!==requested)return;
    if(!response.ok)throw new Error(data.error || "Could not load matching neural activity");
    this.counts=Float32Array.from(this.decode(data.counts_u16,Uint16Array));this.activitySequence=data.sequence;
    document.getElementById("mapNote").textContent=`Decision ${snapshot.decision} · simulated spikes at anatomical anchors · selected links only inside the fly graph`;
    const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.countBuffer);gl.bufferData(gl.ARRAY_BUFFER,this.counts,gl.DYNAMIC_DRAW);this.draw();
  }
  inspect(result) { this.inspected=result;this.selected=result.id;this.draw(); }
  reset() {this.yaw=0;this.pitch=-.2;this.zoom=1;this.pan={x:0,y:0};this.draw();}
  geometry() {return {x:this.w*.30+this.pan.x,y:this.h*.49+this.pan.y,s:Math.min(this.w*.31,this.h*.46)*this.zoom};}
  project(index) {
    const p=this.positions,i=index*3,g=this.geometry(),x=Math.cos(this.yaw)*p[i]+Math.sin(this.yaw)*p[i+2],z=-Math.sin(this.yaw)*p[i]+Math.cos(this.yaw)*p[i+2];
    const y=Math.cos(this.pitch)*p[i+1]-Math.sin(this.pitch)*z,depth=Math.sin(this.pitch)*p[i+1]+Math.cos(this.pitch)*z,f=3/(3+depth);
    return {x:g.x+x*g.s*f,y:g.y+y*g.s*f};
  }
  point(id) {return this.lookup.has(id)?this.project(this.lookup.get(id)):this.nodes.get(id);}
  layout() {
    this.nodes.clear();const step=Math.max(10,Math.min(24,this.w*.024)),x=this.w*.70,y=this.h*.39;
    for(let i=0;i<this.meta.hidden;i++)this.nodes.set(`student:${i}`,{x:x+(i%8-3.5)*step,y:y+(Math.floor(i/8)-3.5)*step,kind:"student",index:i,r:Math.max(4,step*.28)});
    this.meta.memory_features.forEach((name,i)=>this.nodes.set(`memory:${name}`,{x:x+(i%9-4)*step,y:this.h*.68+Math.floor(i/9)*step,kind:"memory",index:i,r:4}));
    this.meta.actions.forEach((name,i)=>{this.nodes.set(`action:${name}`,{x:this.w*.94,y:y+(i-1.5)*56,kind:"action",index:i,r:9});this.nodes.set(`previous:${name}`,{x:x+(i-1.5)*step,y:this.h*.83,kind:"previous",index:i,r:4})});
  }
  line(ctx, edge, selected=false) {
    const a=this.point(typeof edge.source==="string"?edge.source:edge.source.id),b=this.point(typeof edge.target==="string"?edge.target:edge.target.id);if(!a||!b)return;
    ctx.strokeStyle=edge.weight>=0?(selected?"#387467":"#668a8a"):(selected?"#ad603a":"#ad8a70");ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=selected?1.3:.65;ctx.globalAlpha=selected?.85:.22;ctx.setLineDash(edge.weight<0?[3,3]:[]);
    const angle=Math.atan2(b.y-a.y,b.x-a.x),r=b.r||3,tx=b.x-Math.cos(angle)*r,ty=b.y-Math.sin(angle)*r;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(tx,ty);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-5*Math.cos(angle-.45),ty-5*Math.sin(angle-.45));ctx.lineTo(tx-5*Math.cos(angle+.45),ty-5*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.globalAlpha=1;
  }
  draw() {
    const rect=this.overlay.getBoundingClientRect();if(!rect.width||!rect.height||!this.data)return;
    this.w=rect.width;this.h=rect.height;const dpr=window.devicePixelRatio||1;
    for(const c of [this.canvas,this.overlay]){if(c.width!==Math.round(this.w*dpr)||c.height!==Math.round(this.h*dpr)){c.width=Math.round(this.w*dpr);c.height=Math.round(this.h*dpr)}}
    const gl=this.gl,g=this.geometry();gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(246/255,247/255,249/255,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.staticBuffer);
    for(const [name,size,offset] of [["pos",3,0],["role",1,12]]){const p=gl.getAttribLocation(this.program,name);gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,size,gl.FLOAT,false,16,offset)}
    gl.bindBuffer(gl.ARRAY_BUFFER,this.countBuffer);const p=gl.getAttribLocation(this.program,"count");gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,1,gl.FLOAT,false,4,0);
    const u=name=>gl.getUniformLocation(this.program,name);gl.uniform2f(u("viewport"),this.w,this.h);gl.uniform2f(u("center"),g.x,g.y);
    for(const [name,v] of [["scale",g.s],["yaw",this.yaw],["pitch",this.pitch],["ratio",dpr],["filter",this.filter]])gl.uniform1f(u(name),v);
    gl.drawArrays(gl.POINTS,0,this.data.ids.length);
    const ctx=this.overlay.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,this.w,this.h);this.layout();
    if(document.getElementById("showLinks").checked)for(const edge of this.data.edges)this.line(ctx,edge);
    if(document.getElementById("showSelected").checked&&this.inspected && this.snapshot && this.inspected.sequence===this.snapshot.sequence)for(const edge of this.inspected.edges)this.line(ctx,edge,true);
    this.drawLabels(ctx);
    for(const [id,node] of this.nodes){const chosen=id===this.selected;let activity=0;if(node.kind==="student")activity=(this.snapshot?.student_spikes[node.index]||0)/8;if(node.kind==="memory")activity=this.snapshot?.memory?.[node.index]||0;
      ctx.fillStyle=node.kind==="student"?`rgba(55,101,158,${.22+.78*activity})`:node.kind==="memory"?`rgba(124,132,56,${.25+.75*activity})`:"#fff";ctx.strokeStyle=chosen?"#a86024":"#91a1ae";ctx.lineWidth=chosen?2:1;
      ctx.beginPath();ctx.arc(node.x,node.y,node.r,0,Math.PI*2);ctx.fill();ctx.stroke();
      if(node.kind==="action"){ctx.font="8px Segoe UI";ctx.fillStyle="#253b4d";ctx.textAlign="right";ctx.fillText(this.meta.actions[node.index].replace("MOVE_",""),this.w-7,node.y-14);ctx.fillText(this.snapshot?.decision?`${(100*this.snapshot.probabilities[node.index]).toFixed(0)}%`:"—",this.w-7,node.y+23)}
    }
    if(this.selected&&this.lookup.has(this.selected)){const p=this.point(this.selected);ctx.strokeStyle="#a86024";ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.stroke()}
    // Show real endpoints of selected local edges, even when their layer is filtered out.
    if(this.inspected && this.snapshot && this.inspected.sequence===this.snapshot.sequence&&document.getElementById("showSelected").checked){ctx.fillStyle="#a56e40";for(const e of this.inspected.edges)for(const node of [e.source,e.target])if(this.lookup.has(node.id)){const p=this.point(node.id);ctx.beginPath();ctx.arc(p.x,p.y,2.4,0,Math.PI*2);ctx.fill()}}
  }
  drawLabels(ctx) {
    ctx.font="11px Segoe UI";ctx.fillStyle="#47596a";ctx.textAlign="center";
    ctx.fillText("Fly brain",this.w*.30,34);ctx.font="9px Segoe UI";ctx.fillText(`${this.data.ids.length.toLocaleString()} anatomical anchors`,this.w*.30,49);
    const first=this.nodes.get("student:0");ctx.font="11px Segoe UI";ctx.fillText(`${this.meta.hidden} trained cells`,this.w*.70,Math.max(34,first.y-32));ctx.font="9px Segoe UI";ctx.fillText("Schematic arrangement",this.w*.70,Math.max(48,first.y-18));
    if(this.meta.memory_features.length)ctx.fillText(`Action memory · ${this.meta.memory_features.length} inputs`,this.w*.70,this.h*.68-18);
    ctx.fillText("Previous action",this.w*.70,this.h*.83-18);
  }
  pick(x,y) {
    let best=null,distance=64;for(const [id,p] of this.nodes){const d=(p.x-x)**2+(p.y-y)**2;if(d<Math.max(64,p.r*p.r*2)&&d<distance){best=id;distance=d}}
    if(best)return best;
    for(let i=0;i<this.data.ids.length;i++){if((this.filter===1&&!this.counts[i])||(this.filter===2&&this.roles[i]!==1)||(this.filter===3&&this.roles[i]!==2))continue;const p=this.project(i),d=(p.x-x)**2+(p.y-y)**2;if(d<distance){best=this.data.ids[i];distance=d}}
    return best;
  }
  events() {
    const c=this.overlay;let drag=null;
    c.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,yaw:this.yaw,pitch:this.pitch,pan:{...this.pan},shift:e.shiftKey,moved:false};c.setPointerCapture(e.pointerId)});
    c.addEventListener("pointermove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved ||= Math.abs(dx)+Math.abs(dy)>4;if(drag.shift)this.pan={x:drag.pan.x+dx,y:drag.pan.y+dy};else{this.yaw=drag.yaw+dx*.006;this.pitch=Math.max(-1.5,Math.min(1.5,drag.pitch+dy*.006))}this.draw()});
    c.addEventListener("pointerup",e=>{if(drag&&!drag.moved&&this.data){const r=c.getBoundingClientRect(),id=this.pick(e.clientX-r.left,e.clientY-r.top);if(id)this.onSelect(id)}drag=null});c.addEventListener("pointercancel",()=>drag=null);
    c.addEventListener("wheel",e=>{e.preventDefault();this.zoom=Math.max(.4,Math.min(5,this.zoom*Math.exp(-e.deltaY*.001)));this.draw()},{passive:false});
    for(const id of ["showLinks","showSelected"])document.getElementById(id).addEventListener("change",()=>this.draw());
    document.getElementById("mapFilter").addEventListener("change",e=>{this.filter=+e.target.value;this.draw()});
  }
}
