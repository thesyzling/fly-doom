"use strict";

const $ = (id) => document.getElementById(id);
const format = (n) => n.toLocaleString("en-US");
const labels = {
  no_input: ["Resting network", "No external stimulation. The model starts at rest."],
  photoreceptor_pulse: ["Photoreceptor pulse", "A brief input to R1–6 cells. Their modeled inhibitory output changes downstream voltage."],
  photoreceptor_disconnected: ["Disconnected control", "The same input, with recurrent transmission disabled. Only directly stimulated cells should respond."],
  excitatory_projection_pulse: ["Excitatory control", "Direct stimulation of positive visual projection cells. This bypasses the retina and does not demonstrate vision."]
};
let meta, vertices, gl, program, buffer, voltageBuffer, voltageDelta;
let replayData = null, frame = 0, playTimer = null, selectedNeuron = null;
let yaw = 0, pitch = -0.2, zoom = 1, condition, filter = 0, loadVersion = 0;
let dragging = null, width = 1, height = 1;
const canvas = $("brain");

async function fetchOK(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
  return response;
}

function fail(error) {
  pausePlayback();
  $("loading").hidden = false;
  $("loading").textContent = error.message;
}

function shader(type, source) {
  const object = gl.createShader(type);
  gl.shaderSource(object, source);
  gl.compileShader(object);
  if (!gl.getShaderParameter(object, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(object));
  return object;
}

function initGL() {
  gl = canvas.getContext("webgl", {alpha: false, antialias: true});
  if (!gl) throw new Error("WebGL is unavailable. Enable browser graphics acceleration and reload.");
  program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    attribute float spikes, stimulated, group, voltageDelta;
    uniform float yaw, pitch, zoom, aspect, pixelRatio, maxSpikes, filter, descendingGroup, voltageMode;
    varying vec4 color;
    void main() {
      float x = cos(yaw)*position.x + sin(yaw)*position.z;
      float z = -sin(yaw)*position.x + cos(yaw)*position.z;
      float y = cos(pitch)*position.y - sin(pitch)*z;
      float depth = sin(pitch)*position.y + cos(pitch)*z;
      float perspective = 3.0 / (3.0 + depth);
      gl_Position = vec4(x*zoom*perspective/aspect, -y*zoom*perspective, depth/4.0, 1.0);
      bool outputCell = abs(group-descendingGroup)<0.1;
      bool visible = filter<0.5 || (filter<1.5 && spikes>0.0) ||
        (filter>1.5 && filter<2.5 && stimulated>0.5) || (filter>2.5 && outputCell);
      float intensity = log(1.0+spikes)/log(1.0+max(1.0,maxSpikes));
      color = vec4(0.28,0.38,0.52,0.10);
      gl_PointSize = 1.3*pixelRatio;
      if (spikes>0.0) {
        vec3 rgb = stimulated>0.5 ? vec3(0.32,0.85,0.93) : vec3(1.0,0.71,0.36);
        if (outputCell) rgb = vec3(0.97,0.53,0.78);
        color = vec4(rgb,0.35+0.65*intensity);
        gl_PointSize = (2.0+2.5*intensity)*pixelRatio;
      }
      if (voltageMode>0.5) {
        float magnitude=min(1.0,abs(voltageDelta)/20.0);
        color=vec4(0.28,0.38,0.52,0.10);
        gl_PointSize=1.3*pixelRatio;
        if (abs(voltageDelta)>0.01) {
          vec3 rgb=voltageDelta<0.0 ? vec3(0.35,0.60,1.0) : vec3(1.0,0.71,0.36);
          color=vec4(rgb,0.25+0.75*magnitude);
          gl_PointSize=(1.8+2.5*magnitude)*pixelRatio;
        }
      }
      if (!visible) {color.a=0.0; gl_Position=vec4(2.0,2.0,2.0,1.0);}
    }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec4 color;
    void main() {
      if (distance(gl_PointCoord,vec2(0.5))>0.5 || color.a==0.0) discard;
      gl_FragColor=color;
    }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  for (const [name, size, offset] of [["position",3,0],["spikes",1,12],["stimulated",1,16],["group",1,20]]) {
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 24, offset);
  }
  voltageBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, voltageBuffer);
  const voltageLocation = gl.getAttribLocation(program, "voltageDelta");
  gl.enableVertexAttribArray(voltageLocation);
  gl.vertexAttribPointer(voltageLocation, 1, gl.FLOAT, false, 4, 0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);
}

function draw() {
  if (!gl || !vertices) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  width = canvas.clientWidth; height = canvas.clientHeight;
  if(canvas.width!==Math.round(width*ratio))canvas.width=Math.round(width*ratio);
  if(canvas.height!==Math.round(height*ratio))canvas.height=Math.round(height*ratio);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.035, 0.06, 0.10, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  const uniforms = {yaw,pitch,zoom,aspect:width/height,pixelRatio:ratio,
    maxSpikes:replayData ? meta.conditions[condition].max_bin_spikes : meta.conditions[condition].max_spikes,
    filter,descendingGroup:meta.descending_group,voltageMode:replayData && $("colorMode").value==="voltage" ? 1 : 0};
  for (const [name, value] of Object.entries(uniforms)) gl.uniform1f(gl.getUniformLocation(program,name),value);
  gl.drawArrays(gl.POINTS, 0, vertices.length / 6);
}

function svgElement(tag, attributes = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg",tag);
  for (const [k,v] of Object.entries(attributes)) node.setAttribute(k,v);
  node.textContent = text;
  return node;
}

function updateReadout(name) {
  const data = meta.conditions[name], summary = data.summary;
  const [title, description] = labels[name] || [name, "Recorded stimulation condition."];
  $("conditionTitle").textContent = title; $("description").textContent = description;
  $("spikes").textContent = format(summary.spikes);
  $("descending").textContent = format(summary.descending_spikes);
  $("stimulated").textContent = format(summary.stimulated_neurons);
  $("duration").textContent = `${summary.duration_ms} ms`;
  $("groups").replaceChildren();
  const sorted = Object.entries(data.group_totals).sort((a,b)=>b[1]-a[1]);
  const max = Math.max(1,...sorted.map(x=>x[1]));
  for (const [name, count] of sorted) {
    const row=document.createElement("div"); row.className="group";
    const label=document.createElement("div"); label.className="group-label";
    const left=document.createElement("span"),right=document.createElement("span");
    left.textContent=name.replaceAll("_"," "); right.textContent=format(count); label.append(left,right);
    const track=document.createElement("div");track.className="bar-track";
    const bar=document.createElement("div");bar.className="bar-fill";bar.style.transform=`scaleX(${count/max})`;
    track.append(bar);row.append(label,track);$("groups").append(row);
  }
  const trace=$("trace"); trace.replaceChildren();
  trace.append(svgElement("path",{d:"M32 8 V96 H330",class:"axis"}));
  const maxCount=Math.max(1,summary.spikes);
  const points=[[0,0],...summary.trace.map(t=>[t.time_ms,t.cumulative_spikes])];
  const line=points.map(([t,c],i)=>`${i ? "L":"M"}${32+298*t/summary.duration_ms},${96-80*c/maxCount}`).join(" ");
  trace.append(svgElement("path",{d:line,class:"line"}),svgElement("text",{x:32,y:114},"0 ms"),
    svgElement("text",{x:280,y:114},`${summary.duration_ms} ms`),svgElement("text",{x:32,y:10},format(summary.spikes)));
  trace.append(svgElement("line",{id:"traceCursor",x1:32,x2:32,y1:12,y2:96,class:"cursor"}));
  const interpretation = name==="photoreceptor_pulse" ?
    `${format(summary.outside_stimulus_responding_neurons)} unstimulated neurons changed voltage, but ${format(summary.descending_spikes)} descending spikes were recorded. A spike-only map does not show subthreshold inhibition.` :
    name==="excitatory_projection_pulse" ? "Activity reaches descending cells when visual projection neurons are stimulated directly. This control bypasses the retinal pathway." :
    name==="no_input" ? "The network remained at rest without external input." :
    "Recurrent transmission is disabled in this control. Direct stimulation can still make input cells spike.";
  $("interpretation").textContent=interpretation;
}

async function selectCondition(name) {
  pausePlayback();
  const version=++loadVersion;
  $("loading").hidden=false;$("loading").textContent="Loading recorded activity…";
  vertices=null;replayData=null;selectedNeuron=null;
  $("play").disabled=true;$("time").disabled=true;$("colorMode").disabled=true;
  const hasReplay=meta.conditions[name].sample_times_ms.length>0;
  const [points,recording]=await Promise.all([
    fetchOK(`/api/points/${encodeURIComponent(name)}`).then(r=>r.arrayBuffer()),
    hasReplay ? fetchOK(`/api/replay/${encodeURIComponent(name)}`).then(r=>r.arrayBuffer()) : Promise.resolve(null)
  ]);
  const next=new Float32Array(points);
  if (version!==loadVersion) return;
  condition=name;vertices=next;replayData=recording ? new Float32Array(recording) : null;
  voltageDelta=new Float32Array(meta.neurons);
  if(replayData && replayData.length!==meta.conditions[name].sample_times_ms.length*meta.neurons*2)
    throw new Error("Temporal recording length mismatch.");
  gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER,voltageBuffer);gl.bufferData(gl.ARRAY_BUFFER,voltageDelta,gl.DYNAMIC_DRAW);
  updateReadout(name);$("loading").hidden=true;
  $("inspector").textContent="Click a point to inspect this trial's neuron activity.";
  $("play").disabled=!hasReplay;$("time").disabled=!hasReplay;$("colorMode").disabled=!hasReplay;
  $("time").max=hasReplay ? meta.conditions[name].sample_times_ms.length-1 : 0;
  $("time").value=0;
  $("playbackHelp").textContent=hasReplay ?
    "Recorded playback · one saved sample every 400 ms of viewing time. No interpolated or invented activity." :
    "This older run contains only totals. Create a new recording with: python -m flydoom.brain_probe --output runs/my-replay, then open viewer with --run-dir runs/my-replay.";
  $("mapDescription").textContent=hasReplay ? "Watch the recorded response to a brief stimulus. Switch between voltage and spikes." : "Whole-trial totals only. This run has no neuron-by-neuron time recording.";
  $("mapNotice").textContent=hasReplay ?
    "Recorded replay, not live simulation. Voltage is sampled at the selected time; spikes are counted since the previous sample. The cells stay in place while their activity changes." :
    "Static map of total spikes during the whole trial. Playback requires a new recording.";
  frame=0;setFrame(0);updateColorMode();
  if(hasReplay)startPlayback();
}

function pausePlayback(){
  if(playTimer!==null)clearInterval(playTimer);
  playTimer=null;
  $("play").textContent=replayData && frame===meta.conditions[condition].sample_times_ms.length-1 ? "Replay" : "Play recording";
}

function startPlayback(){
  if(!replayData)return;
  if(frame===meta.conditions[condition].sample_times_ms.length-1)setFrame(0);
  if(playTimer!==null)clearInterval(playTimer);
  $("play").textContent="Pause";
  // One recorded 10 ms interval every 400 ms of viewing time (about 40x slower).
  playTimer=setInterval(()=>{
    setFrame(frame+1);
    if(frame===meta.conditions[condition].sample_times_ms.length-1)pausePlayback();
  },400);
}

function setFrame(index){
  if(!vertices)return;
  if(!replayData){$("timeLabel").textContent="Whole trial";$("pulseStatus").textContent="No time recording";draw();return;}
  const data=meta.conditions[condition],times=data.sample_times_ms;
  frame=Math.max(0,Math.min(times.length-1,index));
  const offset=frame*meta.neurons*2;
  for(let i=0;i<meta.neurons;i++){
    vertices[i*6+3]=replayData[offset+i*2];
    voltageDelta[i]=replayData[offset+i*2+1]-meta.rest_mv;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,vertices);
  gl.bindBuffer(gl.ARRAY_BUFFER,voltageBuffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,voltageDelta);
  $("time").value=frame;
  const start=frame ? times[frame-1] : 0,end=times[frame];
  $("timeLabel").textContent=`${end} / ${data.summary.duration_ms} ms`;
  $("pulseStatus").textContent=frame>0 && data.summary.stimulated_neurons>0 &&
    start<data.summary.pulse_end_ms && end>data.summary.pulse_start_ms ? "Stimulus ON in this interval" : "Stimulus OFF";
  const cursor=$("traceCursor");
  if(cursor){cursor.setAttribute("x1",32+298*end/data.summary.duration_ms);cursor.setAttribute("x2",32+298*end/data.summary.duration_ms);}
  if(selectedNeuron)renderNeuron();
  draw();
}

function updateColorMode(){
  const voltage=!!replayData && $("colorMode").value==="voltage";
  $("voltageLegend").hidden=!voltage;$("spikeLegend").hidden=voltage;
  draw();
}

function renderNeuron(){
  if(!selectedNeuron)return;
  const data={...selectedNeuron.data};
  data.whole_trial_spikes=data.spikes;delete data.spikes;
  if(replayData){
    const offset=(frame*meta.neurons+selectedNeuron.index)*2;
    data.selected_time_ms=meta.conditions[condition].sample_times_ms[frame];
    data.spikes_in_interval=replayData[offset];
    data.voltage_mv=Number(replayData[offset+1].toFixed(3));
    data.change_from_rest_mv=Number((replayData[offset+1]-meta.rest_mv).toFixed(3));
  }
  const list=document.createElement("dl");
  for(const [key,value] of Object.entries(data)){
    const term=document.createElement("dt"),detail=document.createElement("dd");
    term.textContent=key.replaceAll("_"," ");detail.textContent=Array.isArray(value)?value.join(", "):String(value);
    list.append(term,detail);
  }
  $("inspector").replaceChildren(list);
}

async function inspectAt(px,py) {
  if (!vertices) return;
  pausePlayback();
  let best=-1,bestDistance=64;
  for(let i=0;i<vertices.length;i+=6){
    const [x,y,z,spikes,stimulated,group]=vertices.subarray(i,i+6);
    if ((filter===1&&spikes===0)||(filter===2&&stimulated===0)||(filter===3&&group!==meta.descending_group)) continue;
    const rx=Math.cos(yaw)*x+Math.sin(yaw)*z,rz=-Math.sin(yaw)*x+Math.cos(yaw)*z;
    const ry=Math.cos(pitch)*y-Math.sin(pitch)*rz,depth=Math.sin(pitch)*y+Math.cos(pitch)*rz;
    const perspective=3/(3+depth);
    const sx=(rx*zoom*perspective/(width/height)+1)*width/2;
    const sy=(ry*zoom*perspective+1)*height/2;
    const distance=(sx-px)**2+(sy-py)**2;
    if(distance<bestDistance){best=i/6;bestDistance=distance;}
  }
  if(best<0){$("inspector").textContent="No point nearby. Zoom in or change the filter.";return;}
  const selectedCondition=condition,version=loadVersion;
  const data=await (await fetchOK(`/api/neuron/${selectedCondition}/${best}`)).json();
  if(version!==loadVersion)return;
  selectedNeuron={data,index:best};renderNeuron();
}

canvas.addEventListener("pointerdown",event=>{
  dragging={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove",event=>{
  if(!dragging)return;
  yaw+=(event.clientX-dragging.x)*0.006;pitch=Math.max(-1.4,Math.min(1.4,pitch+(event.clientY-dragging.y)*0.006));
  dragging.x=event.clientX;dragging.y=event.clientY;draw();
});
canvas.addEventListener("pointerup",event=>{
  if(!dragging)return;
  const moved=Math.hypot(event.clientX-dragging.startX,event.clientY-dragging.startY);
  dragging=null;
  if(moved<4){const rect=canvas.getBoundingClientRect();inspectAt(event.clientX-rect.left,event.clientY-rect.top).catch(fail);}
});
canvas.addEventListener("pointercancel",()=>{dragging=null;});
canvas.addEventListener("wheel",event=>{event.preventDefault();zoom=Math.max(0.4,Math.min(5,zoom*Math.exp(-event.deltaY*0.001)));draw();},{passive:false});
$("filter").addEventListener("change",()=>{filter=Number($("filter").value);draw();});
$("reset").addEventListener("click",()=>{yaw=0;pitch=-0.2;zoom=1;draw();});
$("condition").addEventListener("change",()=>selectCondition($("condition").value).catch(fail));
$("play").addEventListener("click",()=>playTimer===null ? startPlayback() : pausePlayback());
$("time").addEventListener("input",()=>{pausePlayback();setFrame(Number($("time").value));});
$("colorMode").addEventListener("change",updateColorMode);
document.addEventListener("visibilitychange",()=>{if(document.hidden)pausePlayback();});
window.addEventListener("resize",draw);
canvas.addEventListener("webglcontextlost",event=>{event.preventDefault();fail(new Error("Graphics context lost. Reload this page."));});

async function start(){
  meta=await (await fetchOK("/api/meta")).json();
  $("runLabel").textContent=`${meta.run_name} · ${meta.created_utc.slice(0,10)}`;
  $("dataset").textContent=`${format(meta.neurons)} neurons · Real anchor coordinates`;
  for(const name of Object.keys(meta.conditions)){
    const option=document.createElement("option");option.value=name;option.textContent=(labels[name]||[name])[0];$("condition").append(option);
  }
  initGL();
  const initial=Object.hasOwn(meta.conditions,"photoreceptor_pulse")?"photoreceptor_pulse":Object.keys(meta.conditions)[0];
  $("condition").value=initial;$("condition").disabled=false;
  await selectCondition(initial);
}
start().catch(fail);
