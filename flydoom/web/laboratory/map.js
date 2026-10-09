"use strict";

// Real anatomy, spike measurements and directed graph links are separate layers.
class LaboratoryMap extends IntegratedMap {
  initGL() {
    const gl=this.gl;
    const program=(vertex,fragment)=>{
      const p=gl.createProgram();
      for(const [kind,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
        const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);
        if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));
        gl.attachShader(p,shader);gl.deleteShader(shader);
      }
      gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;
    };
    const transform=`uniform vec2 viewport;uniform vec2 center;uniform float scale;uniform float yaw;uniform float pitch;
      vec3 rotate(vec3 p){float x=cos(yaw)*p.x+sin(yaw)*p.z;float z=-sin(yaw)*p.x+cos(yaw)*p.z;return vec3(x,cos(pitch)*p.y-sin(pitch)*z,sin(pitch)*p.y+cos(pitch)*z);}
      vec4 project(vec3 p){vec3 r=rotate(p);vec2 pixel=center+r.xy*scale*3.0/(3.0+r.z);return vec4(pixel.x/viewport.x*2.0-1.0,1.0-pixel.y/viewport.y*2.0,r.z/3.0,1.0);}`;
    this.surfaceProgram=program(`attribute vec3 pos;attribute vec3 normal;${transform}uniform float lineMode;varying float light;
      void main(){gl_Position=project(pos);vec3 n=rotate(normal);light=mix(.34+.66*abs(dot(normalize(n),normalize(vec3(-.4,-.6,1.)))),1.,lineMode);}`,
      `precision mediump float;uniform vec4 tint;varying float light;void main(){gl_FragColor=vec4(tint.rgb*light,tint.a);}`);
    this.program=program(`attribute vec3 pos;attribute float role;attribute float count;${transform}uniform float ratio;uniform float anchors;uniform float activity;varying vec4 color;
      void main(){gl_Position=project(pos);gl_PointSize=1.15*ratio;color=vec4(.57,.70,.75,.15*anchors);
      if(role>1.5)color=vec4(.45,.80,.67,.35*anchors);
      if(activity>.5&&count>0.){float a=min(1.,log(1.+count)/log(9.));color=vec4(1.,.68,.25,.3+.45*a);gl_PointSize=(1.+1.4*a)*ratio;}}`,
      `precision mediump float;varying vec4 color;void main(){if(distance(gl_PointCoord,vec2(.5))>.5||color.a==0.)discard;gl_FragColor=color;}`);
    this.staticBuffer=gl.createBuffer();this.countBuffer=gl.createBuffer();this.skeletons=new Map();
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.clearDepth(1);
  }
  async load(meta){await super.load(meta);this.yaw=0;this.pitch=0;this.draw()}
  async loadSurfaces(){
    const response=await fetch('/api/anatomy'),data=await response.json();if(!response.ok)throw Error(data.error);
    const p=this.decode(data.positions_f32,Float32Array),vertices=new Float32Array(p.length*2);
    for(let i=0;i<p.length;i+=9){
      const ax=p[i+3]-p[i],ay=p[i+4]-p[i+1],az=p[i+5]-p[i+2],bx=p[i+6]-p[i],by=p[i+7]-p[i+1],bz=p[i+8]-p[i+2];
      let nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;const length=Math.hypot(nx,ny,nz)||1;nx/=length;ny/=length;nz/=length;
      for(let j=0;j<3;j++){vertices.set(p.subarray(i+j*3,i+j*3+3),(i/3+j)*6);vertices.set([nx,ny,nz],(i/3+j)*6+3)}
    }
    this.anatomyData=data;this.meshPositions=p;this.surfaceBuffer=this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.surfaceBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,vertices,this.gl.STATIC_DRAW);
    this.rebuildFit();this.draw();
  }
  rebuildFit(){
    const chunks=[this.positions,...(this.meshPositions?[this.meshPositions]:[]),...[...this.skeletons.values()].map(s=>s.positions)];
    this.fitPositions=new Float32Array(chunks.reduce((n,p)=>n+p.length,0));let offset=0;
    for(const p of chunks){this.fitPositions.set(p,offset);offset+=p.length}this.boundsKey=null;
  }
  addSkeleton(data){
    if(this.skeletons.has(data.id))return;
    if(this.skeletons.size>=16){const first=this.skeletons.keys().next().value;this.gl.deleteBuffer(this.skeletons.get(first).buffer);this.skeletons.delete(first)}
    const positions=this.decode(data.positions_f32,Float32Array),buffer=this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,buffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,positions,this.gl.STATIC_DRAW);
    this.skeletons.set(data.id,{...data,positions,buffer});this.rebuildFit();this.draw();
    document.getElementById('branchCount').textContent=`${this.skeletons.size} real cells / max 16`;
  }
  clearSkeletons(){for(const s of this.skeletons.values())this.gl.deleteBuffer(s.buffer);this.skeletons.clear();this.rebuildFit();this.draw();document.getElementById('branchCount').textContent='0 cells loaded'}
  geometry(){
    const positions=this.fitPositions||this.positions,key=`${this.yaw}:${this.pitch}`;
    if(this.boundsKey!==key){
      const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);let left=Infinity,right=-Infinity,top=Infinity,bottom=-Infinity;
      for(let i=0;i<positions.length;i+=3){const x=cy*positions[i]+sy*positions[i+2],z=-sy*positions[i]+cy*positions[i+2],y=cp*positions[i+1]-sp*z,depth=sp*positions[i+1]+cp*z,f=3/(3+depth);left=Math.min(left,x*f);right=Math.max(right,x*f);top=Math.min(top,y*f);bottom=Math.max(bottom,y*f)}
      this.bounds={left,right,top,bottom};this.boundsKey=key;
    }
    const b=this.bounds,scale=.93*Math.min((this.w-34)/(b.right-b.left),(this.h-105)/(b.bottom-b.top))*this.zoom;
    return{x:this.w/2-(b.left+b.right)*scale/2+this.pan.x,y:this.h/2-(b.top+b.bottom)*scale/2+this.pan.y,s:scale};
  }
  uniforms(program){
    const gl=this.gl,g=this.geometry();gl.useProgram(program);const u=n=>gl.getUniformLocation(program,n);
    gl.uniform2f(u('viewport'),this.w,this.h);gl.uniform2f(u('center'),g.x,g.y);
    for(const [k,v] of [['scale',g.s],['yaw',this.yaw],['pitch',this.pitch]])gl.uniform1f(u(k),v);
    return u;
  }
  attribute(program,name,size,stride,offset){const gl=this.gl,p=gl.getAttribLocation(program,name);if(p<0)return;gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,size,gl.FLOAT,false,stride,offset)}
  draw(){
    if(!this.data)return;const rect=this.overlay.getBoundingClientRect();if(!rect.width||!rect.height)return;
    this.w=rect.width;this.h=rect.height;const dpr=Math.min(window.devicePixelRatio||1,2),gl=this.gl;
    for(const c of [this.canvas,this.overlay]){const w=Math.round(this.w*dpr),h=Math.round(this.h*dpr);if(c.width!==w||c.height!==h){c.width=w;c.height=h}}
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.depthMask(true);gl.clearColor(.045,.078,.105,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const disable=()=>{for(let i=0;i<gl.getParameter(gl.MAX_VERTEX_ATTRIBS);i++)gl.disableVertexAttribArray(i)};
    if(this.anatomyData&&document.getElementById('regions').checked){
      disable();gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);
      const u=this.uniforms(this.surfaceProgram);gl.uniform1f(u('lineMode'),0);gl.bindBuffer(gl.ARRAY_BUFFER,this.surfaceBuffer);
      this.attribute(this.surfaceProgram,'pos',3,24,0);this.attribute(this.surfaceProgram,'normal',3,24,12);
      const colors=[[.48,.69,.74],[.58,.71,.73],[.40,.62,.70],[.65,.76,.68],[.50,.62,.77],[.68,.70,.60]];let start=0;
      for(const part of this.anatomyData.parts){gl.uniform4f(u('tint'),...colors[Number(part.id)%colors.length],+document.getElementById('opacity').value);gl.drawArrays(gl.TRIANGLES,start,part.count);start+=part.count}
    }
    gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.enable(gl.BLEND);
    if(document.getElementById('branches').checked&&this.skeletons.size){
      disable();const u=this.uniforms(this.surfaceProgram);gl.uniform1f(u('lineMode'),1);let i=0;
      const colors=[[.57,.83,.96],[.91,.75,.50],[.60,.91,.73],[.82,.68,.93]];
      for(const [id,s] of this.skeletons){gl.bindBuffer(gl.ARRAY_BUFFER,s.buffer);this.attribute(this.surfaceProgram,'pos',3,12,0);
        const normal=gl.getAttribLocation(this.surfaceProgram,'normal');gl.disableVertexAttribArray(normal);gl.vertexAttrib3f(normal,0,0,1);
        gl.uniform4f(u('tint'),...(id===this.selected?[1,.89,.55]:colors[i++%colors.length]),id===this.selected?1:.64);gl.drawArrays(gl.LINES,0,s.positions.length/3)}
    }
    disable();const u=this.uniforms(this.program);gl.uniform1f(u('ratio'),dpr);gl.uniform1f(u('anchors'),+document.getElementById('anchors').checked);gl.uniform1f(u('activity'),+document.getElementById('spikes').checked);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.staticBuffer);this.attribute(this.program,'pos',3,16,0);this.attribute(this.program,'role',1,16,12);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.countBuffer);this.attribute(this.program,'count',1,4,0);gl.drawArrays(gl.POINTS,0,this.data.ids.length);
    const ctx=this.overlay.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,this.w,this.h);this.nodes.clear();
    if(this.inspected&&this.snapshot&&this.inspected.sequence===this.snapshot.sequence&&document.getElementById('showSelected').checked){
      for(const edge of this.inspected.edges){if(!this.lookup.has(edge.source.id)||!this.lookup.has(edge.target.id))continue;const a=this.point(edge.source.id),b=this.point(edge.target.id),angle=Math.atan2(b.y-a.y,b.x-a.x);
        ctx.strokeStyle=edge.plastic?'#edbc69':edge.weight<0?'#d99eb0':'#7dc3ce';ctx.fillStyle=ctx.strokeStyle;ctx.globalAlpha=.65;ctx.lineWidth=edge.plastic?1.5:.8;ctx.setLineDash(edge.weight<0?[3,3]:[]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(b.x,b.y);ctx.lineTo(b.x-5*Math.cos(angle-.45),b.y-5*Math.sin(angle-.45));ctx.lineTo(b.x-5*Math.cos(angle+.45),b.y-5*Math.sin(angle+.45));ctx.fill();ctx.globalAlpha=1;
      }
    }
    if(this.selected&&this.lookup.has(this.selected)){const p=this.point(this.selected);ctx.strokeStyle='#ffdf87';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke()}
    const scale=100*1.7/this.data.extent_um*this.geometry().s;ctx.strokeStyle='#9cb2bd';ctx.fillStyle='#9cb2bd';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(20,this.h-45);ctx.lineTo(20+scale,this.h-45);ctx.stroke();ctx.font='9px Consolas';ctx.fillText('100 µm · centre plane',20,this.h-51);
    gl.depthMask(true);
  }
  reset(){this.zoom=1;this.pan={x:0,y:0};this.draw()}
  events(){super.events();for(const id of ['regions','opacity','anchors','spikes','branches'])document.getElementById(id).addEventListener('input',()=>this.draw())}
}
