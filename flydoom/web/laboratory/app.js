"use strict";
const $=id=>document.getElementById(id);
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const fmt=(x,n=4)=>Number(x).toFixed(n),num=x=>Number(x).toLocaleString(),short=x=>x.replace('MOVE_','');
const lab={meta:null,map:null,latest:null,shown:null,follow:true,selected:null,detail:null,selectionRequest:0,generation:0,after:false,weights:null,skeletonLoading:false};
function fail(error){$('error').textContent=error?.message||String(error);$('error').hidden=false}
function safe(fn){return async(...args)=>{try{$('error').hidden=true;await fn(...args)}catch(e){fail(e)}}}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function row(values){const tr=el('tr');for(const value of values)tr.append(value instanceof Node?value:el('td',value));return tr}
function linkCell(id,text){const b=el('button',text||id);b.onclick=safe(()=>selectCell(id));return b}
function cellTd(id,text){const td=el('td');td.append(linkCell(id,text));return td}
function facts(target,values){target.replaceChildren();for(const [key,value] of values){target.append(el('dt',key),el('dd',value??'Not measured'))}}
function tab(id){for(const b of document.querySelectorAll('[data-pane]')){const on=b.dataset.pane===id;b.setAttribute('aria-selected',String(on));$(b.dataset.pane).hidden=!on}}
for(const b of document.querySelectorAll('[data-pane]'))b.onclick=()=>tab(b.dataset.pane);

function frame(){
  const s=lab.shown;if(!s?.frame)return;const after=lab.after&&s.after_frame;
  $('gameFrame').src=after?s.after_frame:s.frame;
  $('frameNote').textContent=after?'After action · inspector uses input':lab.after?'Outcome unavailable · input shown':'Policy input · RGB 320 × 240';
  $('before').setAttribute('aria-pressed',String(!lab.after));$('after').setAttribute('aria-pressed',String(lab.after));
}
function paintPolicy(){
  const s=lab.shown;if(!s?.sequence)return;frame();
  $('chosen').textContent=(s.manual_action?'MANUAL / ':'')+s.action;
  $('controllerTag').textContent=s.manual_action?'MANUAL OVERRIDE / BARS = STUDENT':'STUDENT ARGMAX / NO ONLINE TRAINING';
  $('decision').textContent='DECISION '+String(s.decision).padStart(3,'0');$('return').textContent='RETURN '+fmt(s.return,0);
  $('sequenceTag').textContent=`SEQUENCE ${s.sequence} / ${lab.follow?'LIVE':'REVIEW'}`;
  const winner=s.probabilities.indexOf(Math.max(...s.probabilities));$('actions').replaceChildren();
  lab.meta.actions.forEach((name,i)=>{const r=el('div');r.className='action-row'+(s.decision&&i===winner?' winner':'');r.append(el('span',short(name)),el('span',s.decision?fmt(100*s.probabilities[i],1)+'%':'—'));const bar=el('div'),fill=el('i');bar.className='bar';fill.style.width=(s.decision?100*s.probabilities[i]:0)+'%';bar.append(fill);r.append(bar);$('actions').append(r)});
  $('readoutGrid').replaceChildren();s.student_spikes.forEach((spikes,i)=>{const b=el('button',`S${String(i).padStart(2,'0')} · ${spikes}`);b.style.background=`rgba(77,135,101,${.07+.55*spikes/8})`;b.title=`Cell ${i}: ${spikes}/8 internal spikes`;b.onclick=safe(()=>readout(i));$('readoutGrid').append(b)});
}
async function paintSnapshot(snapshot){
  lab.shown=snapshot;paintPolicy();const generation=lab.generation,sequence=snapshot.sequence;
  await lab.map.update(snapshot);
  const signals=await api('/api/signals?sequence='+sequence);
  if(lab.generation!==generation||lab.shown.sequence!==sequence)return;
  $('clock').textContent=`${num(signals.simulated_ms)} ms simulated`;
  $('populationCount').textContent=`${num(signals.active_cell_count)} SPIKING CELLS`;
  $('voltageRange').textContent=`Current voltage range: ${fmt(signals.voltage_min_mv,2)} to ${fmt(signals.voltage_max_mv,2)} mV`;
  $('activeFly').replaceChildren(...signals.active_cells.map(c=>{const td=cellTd(c.id,c.label);td.append(el('small',c.id));td.lastChild.className='secondary';return row([td,c.spikes,fmt(c.voltage_mv,3)])}));
  $('mapNote').textContent=`Decision ${snapshot.decision} · ${num(signals.active_cell_count)} spiking cells · arrows are graph edges, not axon paths`;
  if(lab.selected)await inspectCell();
}
async function readout(i){
  const s=lab.shown,detail=await api(`/api/neuron?id=student:${i}&sequence=${s.sequence}`);
  if(lab.shown.sequence!==s.sequence)return;
  tab('activityPane');const box=$('readoutDetail');box.replaceChildren(el('h4',`S${String(i).padStart(2,'0')} / drive ${fmt(detail.drive)} / ${detail.spikes} spikes`));
  const table=el('table'),head=el('thead'),body=el('tbody');head.append(row(['Action','Weight','Mean activity','Δ logit']));
  lab.meta.actions.forEach((name,k)=>{const w=lab.weights.arrays['readout.weight'][k][i],a=s.student_spikes[i]/8;body.append(row([short(name),fmt(w),fmt(a),fmt(w*a)]))});table.append(head,body);box.append(table);
}
async function selectCell(id){lab.selected=id;lab.map.selected=id;lab.map.draw();tab('inspectPane');await inspectCell()}
async function inspectCell(){
  const id=lab.selected,s=lab.shown;if(!id||!s)return;const request=++lab.selectionRequest,generation=lab.generation;
  const detail=await api(`/api/neuron?id=${encodeURIComponent(id)}&sequence=${s.sequence}`);
  if(request!==lab.selectionRequest||generation!==lab.generation||lab.shown.sequence!==s.sequence)return;
  lab.detail=detail;lab.map.inspect(detail);$('cellName').textContent=detail.label;$('cellId').textContent=id;
  const annotation=detail.annotation||{};
  facts($('cellFacts'),[['Cell type',detail.cell_type||'Unassigned'],['Super-class',detail.super_class],['Transmitter',detail.transmitter],['Side',annotation.side||'Unspecified'],['Voltage',fmt(detail.voltage_mv,3)+' mV'],['Synaptic current',fmt(detail.synaptic_current_mv,4)+' mV eq.'],['Spikes / window',detail.spikes],['Refractory steps',detail.refractory_steps],['Incoming pairs',num(detail.incoming_count)],['Outgoing pairs',num(detail.outgoing_count)],['Image bin',detail.input_pixel_bin??'Not directly driven']]);
  $('connections').replaceChildren();
  for(const edge of detail.edges){
    const biological=lab.map.lookup.has(edge.source.id)&&lab.map.lookup.has(edge.target.id);
    if(!biological)continue;
    const incoming=edge.target.id===id,partner=incoming?edge.source.id:edge.target.id;
    const td=cellTd(partner,`${incoming?'IN ←':'OUT →'} ${partner}`);const group=el('td');group.append(el('span',edge.group||'fixed'));if(edge.plastic)group.firstChild.className='plastic-tag';
    $('connections').append(row([td,fmt(edge.base_weight??edge.weight,7),fmt(edge.weight,7),fmt(edge.gain??1,6)+'×',group]));
  }
  $('cellScope').textContent=`${detail.explanation} Displaying the strongest local connections for sequence ${s.sequence}.`;
  $('loadCell').disabled=false;$('loadPartners').disabled=false;
}
async function loadSkeleton(id){if(lab.map.skeletons.has(id))return;const data=await api('/api/skeleton?id='+encodeURIComponent(id));lab.map.addSkeleton(data)}
async function loadCells(ids){
  if(lab.skeletonLoading)return;lab.skeletonLoading=true;$('loadContext').disabled=true;$('loadCell').disabled=true;$('loadPartners').disabled=true;
  const errors=[];try{for(let i=0;i<ids.length;i++){try{$('branchCount').textContent=`Loading ${i+1}/${ids.length}…`;await loadSkeleton(ids[i])}catch(e){errors.push(ids[i]+': '+e.message)}}}
  finally{lab.skeletonLoading=false;$('loadContext').disabled=false;$('loadCell').disabled=!lab.selected;$('loadPartners').disabled=!lab.selected;$('branchCount').textContent=`${lab.map.skeletons.size} real cells / max 16`}
  if(errors.length)throw Error('Some skeletons could not load: '+errors.join('; '));
}
async function search(){const data=await api('/api/cells?q='+encodeURIComponent($('query').value));$('searchResults').replaceChildren(...data.cells.map(c=>{const b=linkCell(c.id,c.label);b.append(el('small',c.id));return b}));if(!data.cells.length)$('searchResults').textContent='No cells match this query.'}
$('searchForm').onsubmit=safe(async e=>{e.preventDefault();await search()});
$('loadContext').onclick=safe(()=>loadCells(lab.meta.morphology_examples));
$('loadCell').onclick=safe(()=>loadCells([lab.selected]));
$('loadPartners').onclick=safe(()=>{const ids=[lab.selected,...lab.detail.edges.flatMap(e=>[e.source.id,e.target.id]).filter(id=>id!==lab.selected&&lab.map.lookup.has(id))];return loadCells([...new Set(ids)].slice(0,4))});
$('clearBranches').onclick=()=>lab.map.clearSkeletons();
$('projection').onchange=()=>{const rotations={xy:[0,0],'3d':[.22,-.24],xz:[0,Math.PI/2],yz:[Math.PI/2,0]};[lab.map.yaw,lab.map.pitch]=rotations[$('projection').value];$('viewLabel').textContent=$('projection').value.toUpperCase();lab.map.reset()};
$('fit').onclick=()=>lab.map.reset();
function layers(on){$('anatomyPanel').classList.toggle('layers-collapsed',!on);$('layersToggle').setAttribute('aria-pressed',String(on));lab.map?.draw()}
$('layersToggle').onclick=()=>layers($('anatomyPanel').classList.contains('layers-collapsed'));
const narrow=matchMedia('(max-width:480px)');narrow.addEventListener('change',e=>layers(!e.matches));layers(!narrow.matches);
function expanded(on){$('anatomyPanel').classList.toggle('expanded',on);$('expand').textContent=on?'Close expanded view':'Expand ↗';document.body.style.overflow=on?'hidden':'';lab.map.draw()}
$('expand').onclick=()=>expanded(!$('anatomyPanel').classList.contains('expanded'));document.addEventListener('keydown',e=>{if(e.key==='Escape')expanded(false)});
$('before').onclick=()=>{lab.after=false;frame()};$('after').onclick=()=>{lab.after=true;frame()};

function history(){
  const s=lab.latest;if(!s?.retained_sequences)return;
  $('history').replaceChildren(...s.retained_sequences.map(sequence=>{const b=el('button',String(sequence-1).padStart(2,'0'));b.setAttribute('aria-pressed',String(sequence===lab.shown?.sequence));b.title='Inspect retained decision '+(sequence-1);b.onclick=safe(async()=>{await api('/api/control',{command:'pause'});lab.follow=false;$('liveFollow').setAttribute('aria-pressed','false');await paintSnapshot(await api('/api/snapshot?sequence='+sequence));history()});return b}));
  if(!lab.follow&&!s.retained_sequences.includes(lab.shown?.sequence))$('historyNote').textContent='Selected snapshot expired. Choose a retained decision or Follow live.';
  else $('historyNote').textContent=lab.follow?'Eight neural snapshots in RAM · full decision logs on disk':'Reviewing a retained decision · game paused';
}
$('liveFollow').onclick=safe(async()=>{lab.follow=true;$('liveFollow').setAttribute('aria-pressed','true');await paintSnapshot(lab.latest);history()});
for(const [id,command] of [['run','run'],['pause','pause'],['step','step'],['stop','stop']])$(id).onclick=safe(async()=>{if(command==='run'||command==='step'){lab.follow=true;$('liveFollow').setAttribute('aria-pressed','true')}await api('/api/control',{command})});
$('release').onclick=safe(async()=>{await api('/api/release-memory',{});$('historyNote').textContent='Only the latest snapshot is retained. No GPU teacher is loaded.'});
$('newRun').onclick=safe(async()=>{await api('/api/new-run',{seed:Number($('seed').value),episodes:1,max_decisions:Number($('limit').value),alpha:0});lab.generation++;lab.selectionRequest++;lab.latest=null;lab.shown=null;lab.detail=null;lab.map.activitySequence=-1;lab.map.inspected=null;lab.follow=true;$('liveFollow').setAttribute('aria-pressed','true');$('newRun').closest('details').open=false;lab.meta=await api('/api/meta')});

function chart(report){
  const svg=$('lossChart');svg.replaceChildren();const NS='http://www.w3.org/2000/svg';
  const add=(tag,attrs,text)=>{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e};
  const events=report.history.filter(h=>Number.isFinite(h.kl)),low=Math.min(...events.map(e=>e.kl))*.9,high=Math.max(...events.map(e=>e.kl))*1.08;
  const x=i=>50+i/(report.history.length-1)*525,y=v=>175-(v-low)/(high-low||1)*145;
  for(let i=0;i<4;i++){const v=low+(high-low)*i/3;add('line',{x1:50,x2:580,y1:y(v),y2:y(v),stroke:'#dce3d7'});add('text',{x:2,y:y(v)+4,fill:'#74816d','font-size':10,'font-family':'Consolas'},fmt(v,3))}
  add('polyline',{points:events.map(e=>`${x(e.evaluation)},${y(e.kl)}`).join(' '),fill:'none',stroke:'#739780','stroke-width':1.5});
  for(const e of events){const dot=add('circle',{cx:x(e.evaluation),cy:y(e.kl),r:3.3,fill:e.accepted?'#315d41':'#b3bfa9'});const title=document.createElementNS(NS,'title');title.textContent=`Candidate ${e.evaluation}: KL ${fmt(e.kl,6)}; gains ${e.gains.map(v=>fmt(v,3)).join(', ')}`;dot.append(title)}
  add('text',{x:50,y:204,fill:'#74816d','font-size':10,'font-family':'Consolas'},'Base graph');add('text',{x:450,y:204,fill:'#74816d','font-size':10,'font-family':'Consolas'},'Synaptic candidates →');
}
async function training(){
  const r=lab.meta.synaptic_checkpoint;
  $('edgeCount').textContent=num(r.changed_edges);$('checkpoint').textContent=r.trained_weights_sha256.slice(0,12);
  $('plasticSummary').textContent=`${num(r.changed_edges)} changed edge weights`;
  $('trainingScope').textContent=`${num(r.plastic_edges)} existing edges in the optimization mask; ${num(r.changed_edges)} changed. ${r.parameterization||'Four tied annotation-group gains'}. Fixed signs, fixed graph topology, unchanged decoder.`;
  $('trainingMetrics').replaceChildren();
  for(const [label,value] of [['Train KL / before → after',`${fmt(r.initial.train.kl,4)} → ${fmt(r.final.train.kl,4)}`],['Validation KL / before → after',`${fmt(r.initial.validation.kl,4)} → ${fmt(r.final.validation.kl,4)}`],['Teacher frames / train + validation',`${r.initial.train.samples} + ${r.initial.validation.samples}`],['Magnitude bounds','0.75–1.25×']]){const box=el('div');box.append(el('small',label),el('b',value));$('trainingMetrics').append(box)}
  $('groupRows').replaceChildren(...r.group_labels.map((name,i)=>row([name,num(r.group_edge_counts[i]),fmt(r.gains[i],6)+'×'+(r.group_gain_ranges?' ['+r.group_gain_ranges[i].map(v=>fmt(v,5)).join('–')+']':'')])));chart(r);
  const games=r.development_gameplay,base=games.base_graph.filter(e=>e.kills>0).length,trained=games.trained_graph.filter(e=>e.kills>0).length;
  $('trainingResult').textContent=`Development smoke comparison: base ${base}/2 target hits; trained ${trained}/2, 12 decisions maximum. These familiar starts are not a held-out benchmark. Validation did not select the gains. ${r.scope}`;
  const changes=await api('/api/plastic-edges');$('changedRows').replaceChildren(...changes.edges.map(e=>row([cellTd(e.source),cellTd(e.target),fmt(e.base_weight,7),fmt(e.weight,7),Number(e.delta).toExponential(3),e.group])));
  facts($('provenance'),[['Synaptic patch SHA256',r.synapses_sha256],['Trained graph SHA256',r.trained_weights_sha256],['Frozen student SHA256',r.student_sha256],['Locked training plan SHA256',r.plan_sha256],['Directed graph pairs',num(r.total_edges)],['Morphology',lab.meta.anatomy_source],['Live output directory',lab.meta.output]]);
}
async function poll(){
  const generation=lab.generation;
  try{
    const s=await api('/api/state');if(generation!==lab.generation)return;
    const previous=lab.latest;lab.latest=s;$('connection').textContent='LOCAL / CONNECTED';
    const ended=['completed','stopped','error'].includes(s.phase);$('runState').textContent=s.busy?'Computing…':ended?s.phase:s.paused?'Paused':'Running';
    for(const id of ['run','step'])$(id).disabled=ended||s.busy;$('pause').disabled=ended;$('stop').disabled=ended;$('newRun').disabled=!ended;$('release').disabled=!s.paused||s.busy;
    for(const b of $('motors').children)b.disabled=ended||s.busy||!s.paused;
    if(s.error)throw Error(s.error);
    if(lab.follow&&s.sequence&&(!previous||previous.sequence!==s.sequence||!lab.shown))await paintSnapshot(s);
    history();
  }catch(e){fail(e)}finally{setTimeout(poll,750)}
}
async function init(){
  lab.meta=await api('/api/meta');lab.weights=await api('/api/weights');
  $('seed').value=lab.meta.seed;
  for(const action of lab.meta.actions){const b=el('button',short(action));b.dataset.action=action;b.disabled=true;b.onclick=safe(async()=>{lab.follow=true;await api('/api/manual',{action})});$('motors').append(b)}
  lab.map=new LaboratoryMap(safe(selectCell));await lab.map.load(lab.meta);await lab.map.loadSurfaces();
  await training();await search();poll();
  // Bounded real skeleton context; failures remain visible rather than fabricated.
  await loadCells(lab.meta.morphology_examples);
  if(lab.shown&&!lab.selected)await selectCell(lab.meta.morphology_examples[0]);
}
safe(init)();
