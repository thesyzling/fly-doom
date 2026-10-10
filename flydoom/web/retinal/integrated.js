"use strict";

// Anatomical context and recorded, signed graded states share exact root IDs.
class GradedBrainMap extends LaboratoryMap {
  draw(){
    // Several matching views may finish together; render the anatomy once.
    if(this.drawRequest)return;
    this.drawRequest=requestAnimationFrame(()=>{this.drawRequest=0;super.draw()});
  }
  initGL() {
    super.initGL();
    const gl=this.gl;gl.deleteProgram(this.program);this.program=gl.createProgram();
    const vertex=`attribute vec3 pos;attribute float role;attribute float count;
      uniform vec2 viewport;uniform vec2 center;uniform float scale;uniform float yaw;uniform float pitch;
      uniform float ratio;uniform float anchors;uniform float activity;varying vec4 color;
      void main(){float x=cos(yaw)*pos.x+sin(yaw)*pos.z;float z=-sin(yaw)*pos.x+cos(yaw)*pos.z;
      float y=cos(pitch)*pos.y-sin(pitch)*z;float depth=sin(pitch)*pos.y+cos(pitch)*z;
      vec2 pixel=center+vec2(x,y)*scale*3.0/(3.0+depth);
      gl_Position=vec4(pixel.x/viewport.x*2.0-1.0,1.0-pixel.y/viewport.y*2.0,depth/3.0,1.0);
      color=vec4(.57,.70,.75,.10*anchors);gl_PointSize=1.3*ratio;
      if(role>.5)color=vec4(.45,.80,.67,.25*anchors);
      if(activity>.5&&abs(count)>.000001){float a=min(1.,log(1.+abs(count)*1000.)/log(101.));
      color=vec4(count>0.?vec3(1.,.70,.32):vec3(.36,.72,1.),.35+.5*a);gl_PointSize=(1.5+2.*a)*ratio;}}`;
    const fragment='precision mediump float;varying vec4 color;void main(){if(distance(gl_PointCoord,vec2(.5))>.5||color.a==0.)discard;gl_FragColor=color;}';
    for(const [kind,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
      const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(this.program,shader);gl.deleteShader(shader);
    }
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
  }
  async update(snapshot){
    this.snapshot=snapshot;
    // Clear old activity before fetching a different recorded decision.
    this.counts.fill(0);this.upload();this.inspected=null;this.draw();
    const data=await api('/api/map-activity?'+integrationQuery(snapshot));
    if(this.snapshot!==snapshot)return;
    const states=this.decode(data.delta_f32,Float32Array);
    this.data.visual_indices.forEach((index,i)=>{this.counts[index]=states[i]});this.upload();
    $('mapNote').textContent=`${this.data.ids.length.toLocaleString()} anatomical cells / ${states.length.toLocaleString()} simulated. Decision ${snapshot.sequence}: amber positive, blue negative; gray context is not simulated.`;
    this.draw();
  }
  upload(){this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.countBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,this.counts,this.gl.DYNAMIC_DRAW)}
}

let brain=null,selectedRoot=null,retinaIndex=null,retinaData=null,integrationKey='',selectionRequest=0,teacherRequest=false;
const integrationQuery=s=>'sequence='+s.sequence+'&run='+encodeURIComponent(s.run);
const scientific=v=>Number(v).toExponential(3);
function svgElement(parent,tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;parent.append(n);return n}

async function selectIntegratedCell(root){
  selectedRoot=root;brain.selected=root;brain.draw();
  const snapshot=shown,request=++selectionRequest;if(!snapshot)return;
  const detail=await api('/api/neuron?id='+root+'&'+integrationQuery(snapshot));
  if(shown!==snapshot||selectedRoot!==root||request!==selectionRequest)return;
  $('selectedCellType').textContent=detail.label;
  $('cellFacts').textContent=root+' / '+(detail.simulated?`state ${scientific(detail.state)} / tau ${detail.tau_ms.toFixed(3)} ms. `:'')+detail.note;
  $('biologicalEdges').replaceChildren(...detail.edges.map(edge=>{
    const row=el('tr'),cell=el('td'),button=el('button',edge.source.label);button.onclick=safe(()=>selectIntegratedCell(edge.source.id));
    cell.append(button,el('small',edge.source.id));row.append(cell,el('td',edge.contacts),el('td',scientific(edge.weight)),el('td',scientific(edge.contribution)));return row;
  }));
  $('cellReadout').textContent=detail.decoder?meta.actions.map((action,i)=>`${action}: product ${scientific(detail.decoder.contributions[i])}, weight change ${scientific(detail.decoder.weight_changes[i])}`).join(' | '):'This cell has no direct engineered motor readout. Its influence travels through biological graph edges.';
  brain.inspected=detail;brain.draw();
}

function drawRetina(snapshot,data){
  const canvas=$('retinaView'),ctx=canvas.getContext('2d'),img=new Image();
  img.onload=()=>{if(shown!==snapshot||retinaData!==data)return;ctx.drawImage(img,0,0,canvas.width,canvas.height);ctx.fillStyle='#06182055';ctx.fillRect(0,0,canvas.width,canvas.height);
    for(const dot of data.samples){ctx.beginPath();ctx.arc(dot.uv[0]*canvas.width,dot.uv[1]*canvas.height,dot.index===data.selected_index?5:2,0,Math.PI*2);ctx.fillStyle=dot.index===data.selected_index?'#fff':dot.drive>0?'#f8b15c':'#69bcea';ctx.fill()}};img.src=snapshot.frame;
  $('retinaCaption').textContent=`${data.samples.length} visible samples. Selected R1-6 ${data.selected.root_id}; raw contrast ${scientific(data.input_drive)}. Click a dot to inspect its graph path.`;
  $('pathStatus').textContent=data.reaches_decoder?'EXISTING PATH TO A DECODER CELL':'PARTIAL EXISTING PATH';
  $('pathFacts').textContent=data.edges.map(e=>`${e.source.label} -> ${e.target.label}: ${e.contacts} contacts, weight ${scientific(e.weight)}`).join(' | ');
  $('pathNodes').replaceChildren(...data.nodes.map(n=>{const b=el('button',n.label);b.append(el('small',n.id));b.onclick=safe(()=>selectIntegratedCell(n.id));return b}));
  const svg=$('retinalGraph');svg.replaceChildren();
  const nodes=[...data.nodes,...(data.reaches_decoder?[{label:snapshot.sequence?snapshot.action:'Action decoder',engineered:true}]:[])];
  nodes.forEach((node,i)=>{const x=65+i*770/Math.max(1,nodes.length-1),previous=65+(i-1)*770/Math.max(1,nodes.length-1);
    if(i)svgElement(svg,'line',{x1:previous+16,y1:90,x2:x-16,y2:90,stroke:node.engineered?'#285940':'#a36732','stroke-width':3,'stroke-dasharray':node.engineered?'5 5':'none'});
    const circle=svgElement(svg,'circle',{cx:x,cy:90,r:15,fill:node.engineered?'#285940':'#a36732',tabindex:0,role:'button','aria-label':node.label});
    if(node.id){circle.style.cursor='pointer';circle.onclick=safe(()=>selectIntegratedCell(node.id));circle.onkeydown=e=>{if(e.key==='Enter')circle.onclick()}}
    svgElement(svg,'text',{x,y:55,'text-anchor':'middle',fill:'#263c33','font-size':14},node.label.replace('MOVE_',''));
    svgElement(svg,'text',{x,y:127,'text-anchor':'middle',fill:'#637568','font-size':11},node.engineered?'engineered readout':scientific(node.state));
  });brain.path=data.path;brain.draw();
}
async function refreshRetina(snapshot){
  const data=await api('/api/retina?'+integrationQuery(snapshot)+(retinaIndex===null?'':'&index='+retinaIndex));
  if(shown!==snapshot)return;retinaData=data;drawRetina(snapshot,data);
  if(!selectedRoot)await selectIntegratedCell(data.selected.mi1_root_id);
}
function drawTeacherHistory(){
  const d=meta.distillation,svg=$('teacherChart');svg.replaceChildren();
  const rows=[{epoch:0,validation_kl:d.initial.validation.kl},...d.history,{epoch:d.selected_epoch,validation_kl:d.final.validation.kl}].sort((a,b)=>a.epoch-b.epoch);
  const max=Math.max(...rows.map(r=>r.validation_kl)),end=rows.at(-1).epoch;
  svgElement(svg,'line',{x1:45,y1:155,x2:565,y2:155,stroke:'#ccd3c7'});
  svgElement(svg,'polyline',{points:rows.map(r=>`${45+520*r.epoch/end},${155-125*r.validation_kl/max}`).join(' '),fill:'none',stroke:'#a36732','stroke-width':2});
  const x=45+520*d.selected_epoch/end;svgElement(svg,'line',{x1:x,y1:22,x2:x,y2:155,stroke:'#285940','stroke-dasharray':'4 4'});
  const selected=svgElement(svg,'circle',{cx:x,cy:155-125*d.final.validation.kl/max,r:4,fill:'#285940'});
  svgElement(selected,'title',{},`Selected epoch ${d.selected_epoch}, validation KL ${d.final.validation.kl.toFixed(6)}`);
  svgElement(svg,'text',{x:45,y:180,fill:'#637568','font-size':11},`Epoch 0 to ${end} / validation KL (lower is better)`);
  $('trainingChartTitle').textContent=d.target_source?'Aiming correction / training history':'Measured distillation history';
  $('fitEpoch').textContent='SELECTED EPOCH '+d.selected_epoch;
  $('teacherIdentity').textContent=meta.teacher.repo+' / '+meta.teacher.revision.slice(0,12)+' / '+meta.teacher.role;
  $('teacherMetrics').textContent=`${d.target_source||"Recorded Laya teacher distillation"}. Validation KL ${d.initial.validation.kl.toFixed(4)} before training; ${d.final.validation.kl.toFixed(4)} selected. Label agreement is separate from measured game success.`;
}
async function startIntegrated(){
  drawTeacherHistory();brain=new GradedBrainMap(safe(selectIntegratedCell));await brain.load(meta);await brain.loadSurfaces();
  $('fitBrain').onclick=()=>{brain.yaw=0;brain.pitch=0;brain.reset()};
  $('expandBrain').onclick=()=>{const expanded=document.querySelector('.brain-panel').classList.toggle('expanded');$('expandBrain').textContent=expanded?'Close expanded view':'Expand';brain.draw()};
  $('cellSearch').onclick=safe(async()=>{const result=await api('/api/cells?q='+encodeURIComponent($('cellQuery').value));$('cellResults').replaceChildren(...result.cells.map(c=>{const b=el('button',c.label);b.append(el('small',c.id));b.onclick=safe(()=>selectIntegratedCell(c.id));return b}))});
  $('cellQuery').onkeydown=e=>{if(e.key==='Enter')$('cellSearch').click()};
  $('loadBranches').onclick=safe(async()=>{if(!selectedRoot)throw Error('Select a neuron first');$('loadBranches').disabled=true;try{brain.addSkeleton(await api('/api/skeleton?id='+selectedRoot))}finally{$('loadBranches').disabled=false}});
  $('retinaView').onclick=safe(async e=>{if(!retinaData||!shown)return;const r=e.currentTarget.getBoundingClientRect(),u=(e.clientX-r.left)/r.width,v=(e.clientY-r.top)/r.height;
    const dot=retinaData.samples.reduce((a,b)=>(a.uv[0]-u)**2+(a.uv[1]-v)**2<(b.uv[0]-u)**2+(b.uv[1]-v)**2?a:b);retinaIndex=dot.index;await refreshRetina(shown);await selectIntegratedCell(retinaData.selected.root_id)});
  $('compareTeacher').onclick=safe(async()=>{
    if(teacherRequest||!shown)return;const snapshot=shown;teacherRequest=true;$('compareTeacher').disabled=true;$('teacherStatus').textContent='Loading retained Laya weights and comparing this exact input...';
    try{const value=await api('/api/teacher',{sequence:snapshot.sequence,run:snapshot.run});if(shown!==snapshot)return;
      $('teacherBars').replaceChildren(...meta.actions.map((name,i)=>{const row=el('div');row.append(el('b',name),el('span','Laya '+(value.teacher.probabilities[i]*100).toFixed(1)+'%'),el('span','Neural '+(value.student_probabilities[i]*100).toFixed(1)+'%'));return row}));
      $('teacherStatus').textContent='Exact-frame comparison saved. Laya did not override the neural action.';
    }finally{teacherRequest=false;$('compareTeacher').disabled=false}
  });
}
async function paintIntegrated(snapshot){
  const key=snapshot.run+':'+snapshot.sequence;if(key===integrationKey)return;integrationKey=key;
  $('teacherBars').replaceChildren();$('teacherStatus').textContent='Pause to compare Laya on decision '+snapshot.sequence;
  try{await Promise.all([brain.update(snapshot),refreshRetina(snapshot)]);if(shown===snapshot&&selectedRoot)await selectIntegratedCell(selectedRoot)}
  catch(e){if(shown===snapshot){$('error').hidden=false;$('error').textContent=e.message}}
}
