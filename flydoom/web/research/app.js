"use strict";
const $=id=>document.getElementById(id),el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node};
const ACTIONS=['WAIT','MOVE_LEFT','MOVE_RIGHT','ATTACK'],COLORS=['#849089','#527cab','#5b9481','#bd784f'];
const short=a=>a.replace('MOVE_',''),fixed=x=>Number(x).toFixed(5),pct=x=>(100*x).toFixed(1)+'%',argmax=p=>p.indexOf(Math.max(...p));
const view={mode:'archive',record:null,signals:null,index:0,tic:0,playing:false,timer:null,request:0,selected:null,action:0,latest:null,meta:null,weights:null,map:null,runId:null,inspectRequest:0};
function error(message){$('error').hidden=!message;$('error').textContent=message}
async function api(path,body){const r=await fetch(path,body===undefined?undefined:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}
function rowNow(){return view.mode==='archive'?view.record?.decisions[view.index]:view.latest}
function signalNow(){return view.mode==='archive'?view.signals?.decisions[view.index]:null}
function activityNow(){const row=rowNow();return signalNow()?.activity||row?.student_spikes?.map(n=>n/8)||[]}
function weightsNow(){return view.mode==='archive'?view.signals?.readout_weight:view.weights?.arrays['readout.weight']}
function positiveChoice(activity,weights,i){const values=weights.map(w=>w[i]*activity);return Math.max(...values)>0?argmax(values):null}

class ObservatoryMap extends IntegratedMap {
  events(){
    super.events();let rotating=false;
    this.overlay.addEventListener('pointerdown',event=>{rotating=!event.shiftKey});
    this.overlay.addEventListener('pointermove',event=>{if(rotating&&event.buttons)$('brainView').value='3d'});
    for(const name of ['pointerup','pointercancel'])this.overlay.addEventListener(name,()=>{rotating=false});
  }
  geometry(){
    const key=`${this.yaw}:${this.pitch}`;
    if(this.boundsKey!==key){
      const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
      let left=Infinity,right=-Infinity,top=Infinity,bottom=-Infinity;
      for(let i=0;i<this.positions.length;i+=3){const x=cy*this.positions[i]+sy*this.positions[i+2],z=-sy*this.positions[i]+cy*this.positions[i+2],y=cp*this.positions[i+1]-sp*z,depth=sp*this.positions[i+1]+cp*z,f=3/(3+depth);left=Math.min(left,x*f);right=Math.max(right,x*f);top=Math.min(top,y*f);bottom=Math.max(bottom,y*f)}
      this.projectedBounds={left,right,top,bottom};this.boundsKey=key;
    }
    const b=this.projectedBounds,area={left:18,right:this.brainFocus?this.w-18:this.w*.50-18,top:66,bottom:this.h-34};
    const scale=.92*Math.min((area.right-area.left)/(b.right-b.left),(area.bottom-area.top)/(b.bottom-b.top))*this.zoom;
    return {x:(area.left+area.right)/2-(b.left+b.right)/2*scale+this.pan.x,y:(area.top+area.bottom)/2-(b.top+b.bottom)/2*scale+this.pan.y,s:scale};
  }
  focusBrain(enabled){this.brainFocus=enabled;this.anatomyOpacity=enabled?.30:.25;this.filter=0;$('mapFilter').value='0';this.zoom=1;this.pan={x:0,y:0};$('focusBrain').textContent=enabled?'Show circuit':'Enlarge brain';$('focusBrain').setAttribute('aria-pressed',String(enabled));this.draw()}
  reset(){this.yaw=0;this.pitch=-.2;this.zoom=1;this.pan={x:0,y:0};$('brainView').value='3d';this.filter=0;$('mapFilter').value='0';this.draw()}
  layout(){
    this.nodes.clear();if(this.brainFocus)return;const step=Math.min(19,Math.max(10,this.w*.025)),x=this.w*.66,top=78;
    const bottom=top+7*step,memoryTop=bottom+53,previousTop=memoryTop+step+54;
    this.labelPositions={x,memory:memoryTop-20,previous:previousTop-20};
    for(let i=0;i<this.meta.hidden;i++)this.nodes.set(`student:${i}`,{x:x+(i%8-3.5)*step,y:top+Math.floor(i/8)*step,kind:'student',index:i,r:Math.max(3.5,step*.27)});
    this.meta.memory_features.forEach((name,i)=>this.nodes.set(`memory:${name}`,{x:x+(i%9-4)*step,y:memoryTop+Math.floor(i/9)*step,kind:'memory',index:i,r:3.5}));
    this.meta.actions.forEach((name,i)=>{this.nodes.set(`action:${name}`,{x:this.w-32,y:86+i*66,kind:'action',index:i,r:8});this.nodes.set(`previous:${name}`,{x:x+(i-1.5)*step,y:previousTop,kind:'previous',index:i,r:3.5})});
  }
  drawLabels(ctx){
    if(this.brainFocus){ctx.fillStyle='#47596a';ctx.textAlign='center';ctx.font='13px Segoe UI';ctx.fillText('Fly brain / complete anchor cloud',this.w/2,30);ctx.font='10px Segoe UI';ctx.fillText(`${this.data.ids.length.toLocaleString()} neuron positions / click to inspect`,this.w/2,48);return}
    const {x,memory,previous}=this.labelPositions;
    ctx.fillStyle='#47596a';ctx.textAlign='center';ctx.font='12px Segoe UI';
    ctx.fillText('Fly brain',this.w*.26,30);ctx.fillText(`${this.meta.hidden} trained cells`,x,30);
    ctx.font='10px Segoe UI';ctx.fillText(`${this.data.ids.length.toLocaleString()} anchors`,this.w*.26,48);ctx.fillText('Spiking readout',x,48);
    if(this.meta.memory_features.length)ctx.fillText('Action memory',x,memory);
    ctx.fillText('Previous action',x,previous);
  }
  async load(meta){this.anatomyOpacity=.25;await super.load(meta);this.liveMeta=meta;this.liveEdges=this.data.edges;this.mode='empty'}
  upload(){const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.countBuffer);gl.bufferData(gl.ARRAY_BUFFER,this.counts,gl.DYNAMIC_DRAW)}
  clearRecord(){if(!this.data)return;this.mode='empty';this.snapshot=null;this.inspected=null;this.selected=null;this.counts.fill(0);this.data.edges=[];this.upload();this.draw();$('mapNote').textContent='Loading the recording’s own neural data…'}
  archived(row,signals,data){
    if(!this.data)return;this.mode='archive';this.meta={...this.liveMeta,hidden:data.activity.length,memory_features:signals.memory_feature_names};
    this.snapshot={sequence:`${view.runId}:${view.index}`,decision:row.decision,student_spikes:row.student_spikes,probabilities:row.applied_probabilities,memory:data.memory};
    this.counts.fill(0);signals.output_root_ids.forEach((id,i)=>{const index=this.lookup.get(id);if(index!==undefined)this.counts[index]=Math.abs(data.descending_voltage[i])});this.upload();
    const ranked=data.activity.map((a,i)=>({i,product:a*signals.readout_weight[view.action][i]})).filter(x=>data.activity[x.i]>0).sort((a,b)=>Math.abs(b.product)-Math.abs(a.product)).slice(0,8);
    this.data.edges=ranked.map(({i,product})=>({source:`student:${i}`,target:`action:${ACTIONS[view.action]}`,weight:product}));
    for(const {i} of ranked.slice(0,4))for(const term of data.cells[i].top_inputs.slice(0,2)){const feature=signals.feature_names[term.feature];this.data.edges.push({source:this.featureId(feature),target:`student:${i}`,weight:term.product})}
    this.selected=view.selected;this.inspected=null;
    if(view.selected?.startsWith('student:')){const i=Number(view.selected.split(':')[1]);this.inspected={id:view.selected,sequence:this.snapshot.sequence,edges:[...data.cells[i].top_inputs.map(term=>({source:{id:this.featureId(signals.feature_names[term.feature])},target:{id:view.selected},weight:term.product})),...ACTIONS.map((a,k)=>({source:{id:view.selected},target:{id:`action:${a}`},weight:data.activity[i]*signals.readout_weight[k][i]}))]}}
    this.draw();$('mapNote').textContent=`EP ${row.episode} / decision ${row.decision} · recorded descending |voltage feature| + student spikes. Other fly activity unavailable.`;
  }
  featureId(feature){return feature.channel==='action memory'?'memory:'+feature.id:feature.channel==='previous action'?'previous:'+feature.id:feature.id}
  line(ctx,edge,selected=false){super.line(ctx,edge,selected||this.mode==='archive')}
  async live(snapshot){
    this.mode='live';this.meta=this.liveMeta;this.data.edges=this.liveEdges;this.snapshot={...snapshot,probabilities:snapshot.applied_probabilities||snapshot.probabilities};this.selected=view.selected;
    if(this.activitySequence!==snapshot.sequence){this.counts.fill(0);this.upload();this.draw();const sequence=snapshot.sequence;const data=await api('/api/map-activity?sequence='+sequence);if(this.mode!=='live'||this.snapshot.sequence!==sequence)return;this.counts=Float32Array.from(this.decode(data.counts_u16,Uint16Array));this.activitySequence=sequence;this.upload()}
    this.draw();$('mapNote').textContent=`LIVE · EP ${snapshot.episode} / decision ${snapshot.decision} · full simulated spike counts aligned to this observation.`;
  }
  draw(){
    super.draw();if(!this.data||!this.snapshot)return;
    const weights=this.mode==='archive'?view.signals?.readout_weight:view.weights?.arrays['readout.weight'];if(!weights)return;
    const ctx=this.overlay.getContext('2d');
    for(const [id,node] of this.nodes){if(node.kind!=='student')continue;const a=(this.snapshot.student_spikes[node.index]||0)/8,p=positiveChoice(a,weights,node.index);ctx.beginPath();ctx.arc(node.x,node.y,node.r,0,Math.PI*2);ctx.fillStyle=a>0&&p!==null?COLORS[p]:'#e0e6e3';ctx.globalAlpha=a>0?.5+.5*a:.6;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=id===this.selected?'#223e3a':'#8ca29b';ctx.lineWidth=id===this.selected?2.5:1;ctx.stroke()}
    const applied=rowNow()?.action,node=this.nodes.get('action:'+applied);if(node){ctx.beginPath();ctx.arc(node.x,node.y,node.r+3,0,Math.PI*2);ctx.strokeStyle=COLORS[ACTIONS.indexOf(applied)];ctx.lineWidth=2;ctx.stroke()}
  }
}

function pause(){clearTimeout(view.timer);view.timer=null;view.playing=false;$('play').textContent='▶ Play run'}
function frame(){const row=rowNow();if(!row)return;
  if(view.mode==='archive'){$('gameFrame').src=row.frames[view.tic];const final=view.tic===row.frames.length-1;$('frameTiming').textContent=view.tic===0?'Observation before the decision':`After game tic ${view.tic} / ${row.effect.game_tics}`;$('outcomeBadge').textContent=final&&row.effect.kill_delta?'TARGET HIT':final&&row.effect.episode_finished?'EPISODE ENDED':'';$('outcomeBadge').classList.toggle('hit',final&&row.effect.kill_delta>0);if(final&&row.effect.episode_finished&&!row.effect.terminal_image_available)$('frameTiming').textContent='Last available frame · episode ended (engine counter)'}
  else{$('gameFrame').src=row.frame;$('frameTiming').textContent='Live observation before action · outcome after action';$('outcomeBadge').textContent=row.effect?.kill_delta?'TARGET HIT':row.effect?.episode_finished?'EPISODE ENDED':'';$('outcomeBadge').classList.toggle('hit',row.effect?.kill_delta>0)}
  $('episodeLabel').textContent=`EP ${row.episode} / DECISION ${row.decision}`;$('actionBadge').textContent=short(row.action);$('returnLabel').textContent=`RETURN ${Number(row.return).toFixed(0)}`;
}
function paintDecision(){const row=rowNow();if(!row?.decision&&view.mode==='live'&&!row?.sequence)return;if(!row)return;
  frame();const probs=row.applied_probabilities||row.probabilities,vision=view.mode==='archive'?row.vision_probabilities:row.vision?.probabilities;
  $('actionChoices').replaceChildren(...ACTIONS.map((name,i)=>{const button=el('button');button.setAttribute('aria-pressed',String(view.action===i));button.append(el('span',short(name)),el('strong',pct(probs[i])),el('small',row.action===name?'APPLIED':'INSPECT'));const bar=el('div');bar.className='prob-bar';const v=el('i'),s=el('i');v.style.width=pct(vision?row.alpha*vision[i]:0);s.className='student';s.style.width=pct((vision?1-row.alpha:1)*row.probabilities[i]);bar.append(v,s);button.append(bar);button.onclick=()=>{pause();view.action=i;paintDecision()};return button}));
  $('controllerLabel').textContent=!vision||row.alpha===0?'STUDENT CONTROL':row.alpha===1?'VISION CONTROL':'MIXED CONTROL';
  $('visionRole').textContent=vision?`Vision ${short(ACTIONS[argmax(vision)])} ${pct(Math.max(...vision))} · student ${short(row.student_action||ACTIONS[argmax(row.probabilities)])} · mix ${row.alpha.toFixed(2)} / ${(1-row.alpha).toFixed(2)}`:'Vision was not executed. The student selected every recorded action.';
  const distilled=view.mode==='archive'?view.signals?.distilled_from_vision:view.meta.student_distilled_from_vision;
  $('teachingRole').textContent=distilled?'Learned from Laya Vision offline. All weights are fixed during this run.':'Vision and student meet at the action mixture; this checkpoint predates Vision distillation.';
  const weights=weightsNow(),activity=activityNow();$('activeCells').replaceChildren();$('contributionHeading').textContent=`${short(ACTIONS[view.action])} Δlogit`;
  if(weights){const ranked=activity.map((a,i)=>({a,i,value:a*weights[view.action][i]})).filter(x=>x.a>0).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));$('activeCount').textContent=`${ranked.length} ACTIVE / ${activity.length}`;
    for(const {a,i,value} of ranked.slice(0,6)){const row=el('tr');row.classList.toggle('selected',view.selected===`student:${i}`);const td=el('td'),button=el('button',`S${String(i).padStart(2,'0')}`),choice=positiveChoice(a,weights,i),dot=el('i');dot.className='cell-dot';dot.style.setProperty('--cell-color',choice===null?'#9aa89d':COLORS[choice]);button.prepend(dot);button.onclick=()=>select(`student:${i}`);td.append(button);const term=el('td',(value>=0?'+':'')+fixed(value));term.className=value<0?'negative-text':'';row.append(td,el('td',Math.round(a*8)),el('td',choice===null?'No positive contribution':short(ACTIONS[choice])),term);$('activeCells').append(row)}}
  else $('activeCount').textContent='VERIFYING RECORD';
  if(view.mode==='archive'){
    $('position').textContent=`${view.index+1} / ${view.record.decisions.length}`;$('seek').value=view.index;$('previous').disabled=view.index===0;$('next').disabled=view.index===view.record.decisions.length-1;
    [...$('decisionStrip').children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===view.index)));
    if(view.signals)view.map?.archived(row,view.signals,signalNow());
  }
  if(view.selected)inspect();
}
function seek(index){if(!view.record)return;view.index=Math.max(0,Math.min(view.record.decisions.length-1,index));view.tic=0;view.action=ACTIONS.indexOf(rowNow().action);paintDecision()}
function schedule(){const length=rowNow().frames.length,pace=Number($('pace').value);view.timer=setTimeout(tick,view.tic===length-1?pace*.5:pace*.5/Math.max(1,length-1))}
function tick(){if(!view.playing||view.mode!=='archive')return;const row=rowNow();if(view.tic<row.frames.length-1){view.tic++;frame()}else if(view.index<view.record.decisions.length-1)seek(view.index+1);else{pause();return}schedule()}

async function loadRecord(id){pause();const token=++view.request;view.runId=id;view.signals=null;view.selected=null;view.inspectRequest++;$('inspector').hidden=true;view.map?.clearRecord();$('play').disabled=true;error('');
  try{const record=await api('/api/replay?id='+encodeURIComponent(id));if(token!==view.request)return;view.record=record;$('seek').max=record.decisions.length-1;$('runSummary').textContent=`${record.decisions.length} decisions / ${record.episodes.reduce((n,e)=>n+e.kills,0)} kills`;$('recordPath').textContent=record.source_run;$('recordScope').textContent=record.note;
    $('decisionStrip').replaceChildren(...record.decisions.map((row,i)=>{const b=el('button',String(i+1).padStart(2,'0'));b.style.setProperty('--cell-color',COLORS[ACTIONS.indexOf(row.action)]);b.title=`Episode ${row.episode}, decision ${row.decision}: ${short(row.action)}`;b.onclick=()=>{pause();seek(i)};return b}));
    if(view.mode==='archive')seek(0);const signals=await api('/api/replay-signals?id='+encodeURIComponent(id));if(token!==view.request)return;
    view.signals=signals;$('play').disabled=false;if(view.mode==='archive'){paintDecision();$('checkpointLabel').textContent='RECORD '+signals.checkpoint_sha256.slice(0,12)}
  }catch(e){if(token===view.request){error(e.message);$('mapNote').textContent='Neural data unavailable; no activity is inferred.'}}
}
async function refresh(){try{const list=await api('/api/replays'),previous=$('runSelect').value;const preferred=list.find(r=>r.checkpoint_sha256===view.meta.checkpoint_sha256&&r.student_only);$('runSelect').replaceChildren(...list.map(r=>{const option=el('option',r.title);option.value=r.id;return option}));if(!list.length){$('runSummary').textContent='Complete a live run to create a recording.';return}$('runSelect').value=list.some(r=>r.id===previous)?previous:preferred?.id||list[0].id;await loadRecord($('runSelect').value)}catch(e){error(e.message)}}
function select(id){pause();if(view.map.brainFocus&&!view.map.lookup.has(id))view.map.focusBrain(false);view.selected=id;if(id.startsWith('action:'))view.action=ACTIONS.indexOf(id.split(':')[1]);paintDecision()}
function tableRow(values){const row=el('tr');for(const value of values)row.append(el('td',value));return row}
async function inspect(){const row=rowNow(),key=view.selected;if(!row||!key)return;const token=++view.inspectRequest;$('inspector').hidden=false;$('inspectorTitle').textContent=key;$('inputRows').replaceChildren();$('outputRows').replaceChildren();$('inspectorNote').textContent='';
  if(view.mode==='archive'){
    const signals=view.signals,data=signalNow();if(!data)return;
    if(key.startsWith('student:')){const i=Number(key.split(':')[1]),cell=data.cells[i];$('inspectorTitle').textContent=`Student ${String(i).padStart(2,'0')} / ${Math.round(data.activity[i]*8)} spikes`;
      $('inspectorSummary').textContent=`Input drive ${fixed(cell.drive)} = fly ${fixed(cell.fly_sum)} + memory ${fixed(cell.memory_sum)} + previous action ${fixed(cell.previous_sum)} + bias ${fixed(cell.bias)}`;
      for(const term of cell.top_inputs){const f=signals.feature_names[term.feature],tr=tableRow(['',fixed(term.weight),fixed(term.normalized),fixed(term.product)]),b=el('button',f.id+' / '+f.channel);b.onclick=()=>select(view.map.featureId(f));tr.firstChild.append(b);$('inputRows').append(tr)}
      ACTIONS.forEach((a,k)=>$('outputRows').append(tableRow([short(a),fixed(signals.readout_weight[k][i]),fixed(data.activity[i]),fixed(signals.readout_weight[k][i]*data.activity[i])])));
      $('inspectorNote').textContent='Four largest input products. The drive sums all inputs. Output contribution = weight × mean spike activity.';
    }else if(key.startsWith('action:')){const a=ACTIONS.indexOf(key.split(':')[1]);$('inspectorSummary').textContent=`Student logit ${fixed(data.logits[a])}; bias ${fixed(signals.readout_bias[a])}. Applied probability ${pct(row.applied_probabilities[a])}.`;
      data.activity.map((v,i)=>({i,v,w:signals.readout_weight[a][i]})).sort((a,b)=>Math.abs(b.v*b.w)-Math.abs(a.v*a.w)).slice(0,8).forEach(t=>{$('outputRows').append(tableRow([`S${t.i}`,fixed(t.w),fixed(t.v),fixed(t.w*t.v)]))});$('inspectorNote').textContent='Strongest student products, before softmax and any Vision mixture.';
    }else{const index=signals.output_root_ids.indexOf(key);if(index>=0){$('inspectorSummary').textContent=`Recorded descending cell · voltage feature ${fixed(data.descending_voltage[index])} · spike-rate feature ${fixed(data.descending_rate[index])}`;let matches=[];data.cells.forEach((c,i)=>c.top_inputs.forEach(t=>{if(t.feature===index||t.feature===index+signals.descending_count)matches.push({i,t})}));matches.sort((a,b)=>Math.abs(b.t.product)-Math.abs(a.t.product)).slice(0,8).forEach(({i,t})=>$('inputRows').append(tableRow([`→ student:${i}`,fixed(t.weight),fixed(t.normalized),fixed(t.product)])));$('inspectorNote').textContent='Links available in the retained strongest-input list. Unlisted links may still exist.'}else{$('inspectorSummary').textContent=key.startsWith('memory:')?'Recorded memory input: '+fixed(data.memory[signals.memory_feature_names.indexOf(key.slice(7))]):key.startsWith('previous:')?'Previous-action indicator: '+data.previous[ACTIONS.indexOf(key.slice(9))]:'Anatomical anchor only. This cell’s individual activity was not saved in this recording.';$('inspectorNote').textContent='Use Live experiment for full current biological activity and connectivity inspection.'}}
    if(!key.startsWith('student:')){view.map.selected=key;view.map.inspected=null;view.map.draw()}
  }else{
    try{const detail=await api(`/api/neuron?id=${encodeURIComponent(key)}&sequence=${row.sequence}`);if(token!==view.inspectRequest||view.mode!=='live')return;view.map.inspect(detail);$('inspectorTitle').textContent=detail.label+' / '+key;$('inspectorSummary').textContent=detail.kind==='student'?`Input drive ${fixed(detail.drive)} · spikes ${detail.spikes} · reconstructed drive ${fixed(detail.reconstructed_input_drive)}`:detail.explanation;
      for(const edge of detail.edges.slice(0,12))$('inputRows').append(tableRow([edge.source.id+' → '+edge.target.id,fixed(edge.weight),edge.normalized_value===undefined?'—':fixed(edge.normalized_value),edge.input_drive_contribution===undefined?'—':fixed(edge.input_drive_contribution)]));
      if(key.startsWith('student:')){const i=Number(key.split(':')[1]),a=(row.student_spikes[i]||0)/8;ACTIONS.forEach((name,k)=>$('outputRows').append(tableRow([short(name),fixed(weightsNow()[k][i]),fixed(a),fixed(weightsNow()[k][i]*a)])))}$('inspectorNote').textContent='Live data for this exact decision. Biological edges show weights; engineered inputs also show available normalized products.';
    }catch(e){if(token===view.inspectRequest)error(e.message)}
  }
}
async function mode(next){pause();view.mode=next;view.selected=null;view.inspectRequest++;$('inspector').hidden=true;$('archiveMode').setAttribute('aria-pressed',String(next==='archive'));$('liveMode').setAttribute('aria-pressed',String(next==='live'));$('archiveBar').hidden=next==='live';$('liveBar').hidden=next==='archive';$('transport').hidden=next==='live';$('decisionStrip').hidden=next==='live';$('mapSource').textContent=next==='archive'?'RECORDED OUTPUTS':'LIVE SPIKES';view.map.clearRecord();error('');
  if(next==='archive'){if(view.record){view.action=ACTIONS.indexOf(rowNow().action);paintDecision()}$('checkpointLabel').textContent='RECORD '+(view.signals?.checkpoint_sha256.slice(0,12)||'loading')}else{view.map.activitySequence=-1;$('checkpointLabel').textContent='LIVE '+view.meta.checkpoint_sha256.slice(0,12);if(view.latest?.sequence){view.action=Math.max(0,ACTIONS.indexOf(view.latest.action));paintDecision();await view.map.live(view.latest)}}
}
async function metadata(){view.meta=await api('/api/meta');for(const [id,key] of [['seed','seed'],['episodes','episodes'],['limit','max_decisions'],['alpha','alpha']])$(id).value=view.meta[key];const m=view.meta.distillation_metrics;$('trainingSummary').textContent=m?`Vision distillation, epoch ${m.selected_epoch}. Validation teacher agreement: ${pct(m.initial.validation.teacher_agreement)} → ${pct(m.final.validation.teacher_agreement)}. This is imitation accuracy, not a win rate.`:'The live checkpoint retains its earlier training history.';$('provenance').replaceChildren();for(const [key,value] of [['Vision revision',view.meta.vision.revision],['Live student SHA',view.meta.checkpoint_sha256],['Graph',`${view.meta.neurons} cells / ${view.meta.connections} directed pairs`],['Live run',view.meta.output]])$('provenance').append(el('dt',key),el('dd',value))}
async function poll(){try{const old=view.latest;view.latest=await api('/api/state');$('connection').textContent='LOCAL / CONNECTED';$('liveStatus').textContent=view.latest.busy?'Computing decision':view.latest.phase==='ready'?(view.latest.paused?'Paused':'Running'):view.latest.phase;const ended=['completed','stopped','error'].includes(view.latest.phase);for(const id of ['runLive','pauseLive','stepLive','stopLive'])$(id).disabled=ended||((id==='stepLive'||id==='runLive')&&view.latest.busy);$('newRun').disabled=!ended;
    if(view.mode==='live'&&view.latest.sequence&&(!old||old.sequence!==view.latest.sequence||old.phase!==view.latest.phase)){view.action=Math.max(0,ACTIONS.indexOf(view.latest.action));paintDecision();await view.map.live(view.latest)}if(view.latest.error&&view.mode==='live')error(view.latest.error);
  }catch(e){$('connection').textContent='DISCONNECTED';error(e.message)}setTimeout(poll,700)}
for(const [id,command] of [['runLive','run'],['pauseLive','pause'],['stepLive','step'],['stopLive','stop']])$(id).onclick=async()=>{try{await api('/api/control',{command});error('')}catch(e){error(e.message)}};
$('newRun').onclick=async()=>{try{await api('/api/new-run',{seed:Number($('seed').value),episodes:Number($('episodes').value),max_decisions:Number($('limit').value),alpha:Number($('alpha').value)});view.latest=null;view.map.activitySequence=-1;view.map.inspected=null;view.selected=null;$('inspector').hidden=true;await metadata();$('runSettings').open=false;error('')}catch(e){error(e.message)}};
$('archiveMode').onclick=()=>mode('archive');$('liveMode').onclick=()=>mode('live');$('resetView').onclick=()=>view.map?.reset();$('refreshRuns').onclick=refresh;$('runSelect').onchange=()=>loadRecord($('runSelect').value);
$('focusBrain').onclick=()=>view.map?.focusBrain(!view.map.brainFocus);
$('brainView').onchange=()=>{if(!view.map)return;const rotations={'3d':[0,-.2],xy:[0,0],xz:[0,Math.PI/2],yz:[Math.PI/2,0]};[view.map.yaw,view.map.pitch]=rotations[$('brainView').value];view.map.zoom=1;view.map.pan={x:0,y:0};view.map.draw()};
$('previous').onclick=()=>{pause();seek(view.index-1)};$('next').onclick=()=>{pause();seek(view.index+1)};$('seek').oninput=()=>{pause();seek(Number($('seek').value))};
$('play').onclick=()=>{if(view.playing){pause();return}if(!view.record||!view.signals)return;if(view.index===view.record.decisions.length-1&&view.tic===rowNow().frames.length-1)seek(0);view.playing=true;$('play').textContent='Ⅱ Pause';schedule()};
$('closeInspector').onclick=()=>{view.selected=null;view.inspectRequest++;$('inspector').hidden=true;view.map.inspected=null;view.map.selected=null;paintDecision();view.map.draw()};
(async()=>{try{await metadata();view.weights=await api('/api/weights');view.map=new ObservatoryMap(select);await view.map.load(view.meta);await poll();await refresh()}catch(e){error(e.message)}})();
