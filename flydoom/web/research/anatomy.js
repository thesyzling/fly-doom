"use strict";
// Mesh coordinates and neuron anchors share the upstream FlyWire coordinate space.
IntegratedMap.prototype.loadAnatomy=async function(){
  const response=await fetch('/api/anatomy'),data=await response.json();if(!response.ok)throw Error(data.error);
  this.anatomyData=data;const p=this.decode(data.positions_f32,Float32Array),vertices=new Float32Array(p.length*2);
  for(let i=0;i<p.length;i+=9){const ax=p[i+3]-p[i],ay=p[i+4]-p[i+1],az=p[i+5]-p[i+2],bx=p[i+6]-p[i],by=p[i+7]-p[i+1],bz=p[i+8]-p[i+2];let nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;const length=Math.hypot(nx,ny,nz)||1;nx/=length;ny/=length;nz/=length;for(let j=0;j<3;j++){vertices.set(p.subarray(i+j*3,i+j*3+3),(i/3+j)*6);vertices.set([nx,ny,nz],(i/3+j)*6+3)}}
  this.fitPositions=new Float32Array(this.positions.length+p.length);this.fitPositions.set(this.positions);this.fitPositions.set(p,this.positions.length);this.baseFitPositions=this.fitPositions;this.boundsKey=null;
  const gl=this.gl;this.anatomyBuffer ||= gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.anatomyBuffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);
  const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s};
  const program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,`attribute vec3 pos;attribute vec3 normal;uniform vec2 viewport;uniform vec2 center;uniform float scale;uniform float yaw;uniform float pitch;varying float shade;void main(){float x=cos(yaw)*pos.x+sin(yaw)*pos.z;float z=-sin(yaw)*pos.x+cos(yaw)*pos.z;float y=cos(pitch)*pos.y-sin(pitch)*z;float depth=sin(pitch)*pos.y+cos(pitch)*z;vec2 pixel=center+vec2(x,y)*scale*3.0/(3.0+depth);gl_Position=vec4(pixel.x/viewport.x*2.0-1.0,1.0-pixel.y/viewport.y*2.0,0.0,1.0);shade=.55+.45*abs(dot(normal,normalize(vec3(.3,-.5,.8))));}`));
  gl.attachShader(program,compile(gl.FRAGMENT_SHADER,`precision mediump float;uniform vec4 tint;varying float shade;void main(){gl_FragColor=vec4(tint.rgb*shade,tint.a);}`));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));if(this.anatomyProgram)gl.deleteProgram(this.anatomyProgram);this.anatomyProgram=program;this.draw();
};
IntegratedMap.prototype.setSkeleton=function(data){
  const gl=this.gl,p=this.decode(data.positions_f32,Float32Array);this.skeletonData=data;this.skeletonCount=p.length/3;const base=this.baseFitPositions||this.positions;this.fitPositions=new Float32Array(base.length+p.length);this.fitPositions.set(base);this.fitPositions.set(p,base.length);this.boundsKey=null;this.zoom=1;this.pan={x:0,y:0};
  this.skeletonBuffer ||= gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.skeletonBuffer);gl.bufferData(gl.ARRAY_BUFFER,p,gl.STATIC_DRAW);this.draw();
};
IntegratedMap.prototype.drawAnatomyGL=function(){
  if(!this.anatomyProgram)return;const gl=this.gl,g=this.geometry(),program=this.anatomyProgram;gl.useProgram(program);const u=n=>gl.getUniformLocation(program,n);
  gl.uniform2f(u('viewport'),this.w,this.h);gl.uniform2f(u('center'),g.x,g.y);for(const [name,value] of [['scale',g.s],['yaw',this.yaw],['pitch',this.pitch]])gl.uniform1f(u(name),value);
  const pos=gl.getAttribLocation(program,'pos'),normal=gl.getAttribLocation(program,'normal');
  if(document.getElementById('showAnatomy')?.checked){gl.bindBuffer(gl.ARRAY_BUFFER,this.anatomyBuffer);gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(normal);gl.vertexAttribPointer(normal,3,gl.FLOAT,false,24,12);let start=0;const colors=[[.40,.65,.65],[.54,.61,.76],[.66,.65,.48],[.60,.48,.62]];for(const part of this.anatomyData.parts){gl.uniform4f(u('tint'),...colors[Number(part.id)%colors.length],.12);gl.drawArrays(gl.TRIANGLES,start,part.count);start+=part.count}}
  if(this.skeletonData?.id===this.selected){gl.bindBuffer(gl.ARRAY_BUFFER,this.skeletonBuffer);gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,12,0);gl.disableVertexAttribArray(normal);gl.vertexAttrib3f(normal,0,0,1);gl.uniform4f(u('tint'),.12,.25,.38,.95);gl.drawArrays(gl.LINES,0,this.skeletonCount)}
};
