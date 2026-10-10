"use strict";
const $=id=>document.getElementById(id),el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
let meta,shown,follow=true,outcome=false,polling=false,playbackRequest=0,historyKey="",latestStatus=null;
let restarting=false,episodeGeneration=0,connectionLost=false;
const displayedFrames=[];
async function api(path,body){
 const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),path==='/api/teacher'?180000:path==='/api/restart'?40000:path.startsWith('/api/state')?5000:30000);
 try{const response=await fetch(path,{signal:controller.signal,...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data}
 catch(e){if(e.name==='AbortError')throw Error('The server did not respond in time. State polling will retry automatically.');throw e}
 finally{clearTimeout(timeout)}
}
function episodeStatus(s){
 const box=$('episodeStatus'),terminal=s.phase==='completed',failed=s.phase==='error';
 box.hidden=!(terminal||failed||s.phase==='stopped');box.dataset.kind=failed?'error':terminal?'completed':'stopped';
 $('restart').classList.toggle('primary',terminal||failed);
 if(box.hidden)return;
 const reason=s.termination?.reason;
 const titles={target_eliminated:'Episode complete / target eliminated',time_limit:'Episode complete / time limit',decision_limit:'Episode complete / recording limit',player_dead:'Episode complete / player died',engine_finished:'Episode complete / engine finished'};
 $('episodeTitle').textContent=failed?'Live session error':terminal?(titles[reason]||'Episode complete'):'Run stopped';
 $('episodeDetail').textContent=failed?`The last saved decision is retained. ${s.error||'See diagnostics for details.'} Log: ${s.diagnostics_log||s.run}. Use Restart episode to try a fresh episode.`:terminal?`The game has ended normally; the server is still connected. ${s.termination?`Engine tick ${s.termination.game_tick}, time limit ${s.termination.timeout_tics} tics; recording limit ${s.termination.decision_limit} decisions. `:''}Use Restart episode to play again. Saved decisions and weights are retained.`:'Stopped by the user. Use Restart episode to play again; existing records are retained.';
}
function safe(fn){return async(...args)=>{try{$('error').hidden=true;await fn(...args)}catch(e){$('error').hidden=false;$('error').textContent=e.message}}}
function frame(){if(!shown)return;$('frame').src=outcome&&shown.after_frame?shown.after_frame:shown.frame;$('frameToggle').disabled=!shown.after_frame;$('frameToggle').textContent=!shown.after_frame?(shown.sequence?'Terminal input / no next frame':'Decision input'):outcome?'Show decision input':'Show action outcome'}

function stopPlayback(){cancelAnimationFrame(playbackRequest);playbackRequest=0}
function playFrames(s){
 stopPlayback();if(s.paused||!s.action_frames?.length)return;
 const started=performance.now(),interval=s.frame_interval_ms||1000/35;let previous=-1;
 const draw=now=>{if(shown!==s||!follow||latestStatus?.paused)return;const index=Math.min(s.action_frames.length-1,Math.floor((now-started)/interval));
  if(index!==previous){$('frame').src=s.action_frames[index];previous=index;$('frameToggle').textContent='Live action frames';displayedFrames.push(now);while(displayedFrames.length&&displayedFrames[0]<now-2000)displayedFrames.shift();if(displayedFrames.length>1)$('playbackRate').textContent=`Observed live frame updates: ${((displayedFrames.length-1)*1000/(now-displayedFrames[0])).toFixed(1)}/s. Real engine images; neural decisions still span four game tics.`;}
  if(index<s.action_frames.length-1)playbackRequest=requestAnimationFrame(draw);else playbackRequest=0;
 };playbackRequest=requestAnimationFrame(draw);
}
function trace(){const svg=$('trace');svg.replaceChildren();const rows=shown?.trace||[],kind=$('population').value;if(!rows.length)return;const vals=rows.map(r=>r.values[kind]),min=Math.min(0,...vals),max=Math.max(0,...vals),range=Math.max(max-min,1e-9),end=rows.at(-1).ms,ns='http://www.w3.org/2000/svg';const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;svg.append(n)};add('text',{x:10,y:16,fill:'#637568','font-size':10},`${kind} / dimensionless state / ${max.toExponential(2)}`);add('line',{x1:45,y1:130-(0-min)/range*100,x2:580,y2:130-(0-min)/range*100,stroke:'#c7d1c0'});add('polyline',{points:rows.map((r,i)=>`${45+535*r.ms/end},${130-(vals[i]-min)/range*100}`).join(' '),fill:'none',stroke:'#a36732','stroke-width':2});add('text',{x:45,y:154,fill:'#637568','font-size':10},`0 → ${end.toFixed(1)} ms / min ${min.toExponential(2)}`)}
function paint(s){stopPlayback();shown=s;frame();playFrames(s);$('decision').textContent='DECISION '+String(s.sequence).padStart(3,'0');$('action').textContent=s.action;
 $('actions').replaceChildren(...meta.actions.map((name,i)=>{const n=el('div');n.className='action'+(name===s.action?' selected':'');n.append(el('span',name.replace('MOVE_','')),el('span',(s.probabilities[i]*100).toFixed(1)+'%'));const bar=el('div'),fill=el('i');bar.className='bar';fill.style.width=(s.probabilities[i]*100)+'%';bar.append(fill);n.append(bar);return n}));
 $('selectionNote').textContent=s.sequence?(s.learning_at_decision?'Sampled from the neural output policy; reward learning enabled.':'Highest-probability neural action; weights frozen.'):'No action yet. The first step processes the initial image.';
 $('cells').replaceChildren(...(s.terms?.cells||[]).map(c=>{const row=el('tr'),name=el('td'),button=el('button',c.cell_type);button.onclick=safe(()=>selectIntegratedCell(c.root_id));name.append(button,el('small',c.root_id));row.append(name,...[c.state,c.weight,c.activation,c.contribution].map(v=>el('td',v.toExponential(3))));return row}));
 $('logit').textContent=s.terms?`Selected-action logit ${s.terms.logit.toFixed(5)} = all neural products ${s.terms.all_neural_terms.toFixed(5)} + bias ${s.terms.bias.toFixed(5)}. Products outside this table: ${s.terms.other_terms.toFixed(5)}.`:'Waiting for the first decision.';
 $('reward').textContent=s.sequence?(s.reward>0?'+':'')+s.reward:'—';$('return').textContent=`Episode return ${s.return} / kills ${s.kills}`;
 const f=s.feedback;$('delta').textContent=f?f.weight_delta_l2.toExponential(3):'Frozen';$('updates').textContent=f?`${f.updates} reward updates this episode`:'No optimizer update at this decision';$('probability').textContent=f?`${(100*f.action_probability_before).toFixed(2)}% → ${(100*f.action_probability_after).toFixed(2)}%`:'—';
 $('feedbackNote').textContent=f?'Reward changed the live decoder after this action. The next decision uses its new weights. Biological graph gains stay fixed during live play. The saved experiment checkpoint remains intact.':'Game reward is recorded. Enable Learn from reward while paused to apply stochastic policy updates.';
 $('timing').textContent=s.sequence&&s.timing?`Encoder ${s.encoding_ms.toFixed(1)} ms / archive ${s.timing.archive_ms.toFixed(1)} ms / ${s.timing.total_ms.toFixed(1)} ms before final JSON publication. ${s.action_frames?.length||0} genuine engine frames per action. Neural window remains 114.3 ms.`:s.sequence?`Encoder ${s.encoding_ms.toFixed(1)} ms / decision computation ${s.wall_ms.toFixed(1)} ms (excludes archive write). Neural clock ${s.neural_clock_ms.toFixed(1)} ms. State updates use a fixed 114.3 ms response window.`:'';
 $('runPath').textContent=s.run;trace();history(s.latest_sequence);paintIntegrated(s)
}
function history(count){const key=shown?.run+':'+count+':'+shown?.sequence;if(key===historyKey)return;historyKey=key;$('history').replaceChildren(...Array.from({length:count+1},(_,i)=>{const b=el('button',String(i).padStart(2,'0'));b.setAttribute('aria-pressed',String(i===shown?.sequence));b.onclick=safe(async()=>{await api('/api/control',{command:'pause'});follow=false;paint(await api('/api/state?sequence='+i))});return b}))}
for(const command of ['run','pause','step','stop'])$(command).onclick=safe(async()=>{if(command==='run'||command==='step')follow=true;await api('/api/control',{command});await poll()});
$('restart').onclick=safe(async()=>{
 if(restarting)return;
 restarting=true;episodeGeneration++;stopPlayback();
 $('restart').disabled=true;$('restart').textContent='Restarting...';$('phase').textContent='Restarting episode';
 for(const name of ['run','pause','step','stop','newRun','learning','compareTeacher'])$(name).disabled=true;
 try{
  const result=await api('/api/restart',{});
  follow=true;outcome=false;shown=null;latestStatus=null;historyKey='';displayedFrames.length=0;
  $('history').replaceChildren();$('seed').value=result.seed;
  $('playbackRate').textContent=`Fresh episode / seed ${result.seed}. Learned weights retained.`;
 }finally{
  restarting=false;episodeGeneration++;$('restart').textContent='Restart episode';await poll();
 }
});
$('learning').onchange=safe(async()=>{await api('/api/learning',{enabled:$('learning').checked});await poll()});
$('newRun').onclick=safe(async()=>{await api('/api/new-run',{seed:Number($('seed').value)});follow=true;shown=null;latestStatus=null;await poll()});
$('follow').onclick=safe(async()=>{follow=true;shown=null;latestStatus=null;await poll()});$('frameToggle').onclick=()=>{stopPlayback();outcome=!outcome;frame()};$('population').onchange=trace;
async function poll(){
 if(polling||restarting)return;
 polling=true;const generation=episodeGeneration;
 try{
  const previous=latestStatus;
  const s=await api('/api/state'+(previous?'?since='+previous.sequence+'&run='+encodeURIComponent(previous.run):''));
  if(generation!==episodeGeneration)return;
  latestStatus=s;
  $('testMode').textContent=s.learning?'EXPLORATION: actions are sampled and weights change':'FROZEN TEST: highest-probability action, weights fixed';
  if(connectionLost){connectionLost=false;$('error').hidden=true;}
  $('connection').textContent=s.busy?'COMPUTING':'CONNECTED';
  episodeStatus(s);
  $('phase').textContent=s.restarting?'Restarting episode':s.phase==='ready'?(s.busy?'Computing':s.paused?'Paused':'Running'):s.phase==='completed'?'Episode complete':s.phase;
  for(const name of ['run','step'])$(name).disabled=s.restarting||s.busy||s.teacher_busy||s.phase!=='ready';
  for(const name of ['pause','stop'])$(name).disabled=!!s.restarting;
  $('restart').disabled=s.restarting||s.teacher_busy||teacherRequest;
  $('learning').disabled=s.restarting||s.busy||s.teacher_busy||!s.paused;$('learning').checked=s.learning;
  $('newRun').disabled=s.restarting||s.busy||s.teacher_busy||!s.paused||teacherRequest;
  $('compareTeacher').disabled=s.restarting||s.busy||s.teacher_busy||!s.paused||teacherRequest;
  if(s.error){$('error').hidden=false;$('error').textContent=s.error;}
  if(follow&&!s.unchanged&&(!shown||s.sequence!==shown.sequence||s.run!==shown.run))paint(s);else history(s.latest_sequence);
  if(s.paused&&!previous?.paused){stopPlayback();frame()}
 }catch(e){if(generation===episodeGeneration){connectionLost=true;stopPlayback();$('connection').textContent='RECONNECTING';$('phase').textContent='Connection interrupted';$('error').hidden=false;$('error').textContent=e.message+' Retrying the connection; the last received frame is retained.'}}
 finally{polling=false}
}
safe(async()=>{meta=await api('/api/meta');$('neuronCount').textContent=meta.neurons.toLocaleString();$('readoutCount').textContent=meta.readout_cells;const b=$('benchmark');b.replaceChildren(...Object.entries(meta.benchmark).map(([name,r])=>el('div',`${name.replaceAll('_',' ')}: ${r.kills} total kills in ${r.episodes} episodes; mean return ${r.mean_return.toFixed(1)}`)));await startIntegrated();await poll();setInterval(poll,33)})();
