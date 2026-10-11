"use strict";
let adaptiveReady=false,adaptivePolling=false;
async function adaptiveStatus(){
 if(adaptivePolling)return;adaptivePolling=true;
 try{
  const s=await api('/api/adaptive');if(!s.enabled)return;
  if(!adaptiveReady){
   adaptiveReady=true;const panel=el('section');panel.className='adaptive-panel research-panel';panel.id='adaptivePanel';
   panel.innerHTML='<div class="title"><h2>Adaptive relay / paired reward experiment</h2><span id="relayMode"></span></div><p>Two episodes use the same seed and scenario with opposite connection perturbations. Their actual returns update existing relay gains. The action decoder and response times remain fixed.</p><div class="controls"><label>Scenario <select id="relayScenario"></select></label><label>Pairs <input id="relayPairs" type="number" min="1" max="10" value="2"></label><button id="relayStart" class="primary">Start paired learning</button><button id="relayPause">Pause learning</button></div><p id="relayStatus" class="formula"></p><p id="relayFeedback"></p><p id="relayGains" class="caption"></p><div id="relaySteps" class="research-stages"></div><p class="caption">Experimental checkpoint; not promoted. Previously used physiology means are development data. Independent recordings remain unavailable (download HTTP 401/403).</p><div class="table-scroll"><table><thead><tr><th>Policy</th><th>Scenario</th><th>Episodes</th><th>With kill</th><th>Mean return</th></tr></thead><tbody id="adaptiveResults"></tbody></table></div><details><summary>Persistence and interrupted trials</summary><p>Completed pair updates are saved to the file below. Pause stops automatic advancement. End run and Restart discard an incomplete pair. Server resume restores learned gains and starts a fresh pair; it does not restore a running game.</p><pre id="relayPath"></pre><pre>.\\.venv\\Scripts\\python.exe -m flydoom --adaptive --resume-relay "PATH_TO_RELAY_LEARNER_JSON"</pre></details>';
   $('researchStatus').closest('section').before(panel);
   const jump=el('button','Relay learning');jump.onclick=()=>panel.scrollIntoView({behavior:'smooth',block:'start'});document.querySelector('.controls').append(jump);
   for(const name of s.scenarios){const option=el('option',name.replaceAll('_',' '));option.value=name;$('relayScenario').append(option)}
   $('relayScenario').value=s.scenario;
   $('relayStart').onclick=safe(async()=>{await api('/api/adaptive',{pairs:Number($('relayPairs').value),scenario:$('relayScenario').value});follow=true;await poll();await adaptiveStatus()});
   $('relayPause').onclick=safe(async()=>{await api('/api/control',{command:'pause'});await poll();await adaptiveStatus()});
   $('learning').closest('label').hidden=true;
   $('activeModel').textContent='Experimental eight-action controller / not promoted';$('activeModel').dataset.experimental='true';
   $('readoutLabel').textContent='descending decoder inputs';$('graphRoute').textContent='R1–6 · visual relays · descending cells';
   $('controlMethod').textContent='Only descending-cell states enter the eight-action decoder. Existing anatomical contacts carry the visual response. Laya is an inherited offline teacher; new curriculum annotations are used only during training.';
   $('trainingMethod').textContent='Paired episode rewards update nonempty relay connection groups between episodes. Graph weights stay fixed within each episode. Decoder weights and response times remain fixed in this live mode.';
   $('evidenceScope').textContent='The selected anatomical subcircuit is simulated; the surrounding full-brain map is context. Optical assignments and biological response timing are not independently validated. Multiple-scenario tests are reported below; kills do not prove navigation success.';
   $('decisionCaption').textContent='Largest signed descending-cell contributions to this action. Cell identity is anatomical; the action readout is engineered.';
   $('selectionNote').textContent='The decoder always chooses its highest-probability action. Paired learning perturbs connections between episodes.';
   $('feedbackNote').textContent='The decoder stays fixed. Episode-level biological gain updates appear in Adaptive relay below. The selected checkpoint is retained.';
   const option=el('option','Descending');option.value='Descending';$('population').prepend(option);$('population').value='Descending';
   $('relaySteps').replaceChildren(...Object.entries(s.four_steps).map(([name,status])=>{const row=el('article');row.append(el('strong',name.replaceAll('_',' ')),el('small',status.replaceAll('_',' ')));return row}));
   for(const [policy,tasks] of Object.entries(s.benchmark))for(const [task,r] of Object.entries(tasks)){const row=el('tr');[policy,task,r.episodes,`${r.kills} kills / ${(100*r.kill_episode_rate).toFixed(1)}% episodes`,r.mean_return.toFixed(1)].forEach(v=>row.append(el('td',String(v))));$('adaptiveResults').append(row)}
   const details=el('details'),summary=el('summary','Ablation and timing evidence');details.append(summary);
   const a=s.evidence.ablation,t=s.evidence.timing;
   details.append(el('p',`${a.finding||'Retained development observations; not independent game tests.'} ${a.frames} frames; ${a.active_transmission_groups} nonempty transmission groups; ${a.prior_empty_transmission_parameters} unused transmission channels excluded.`));
   const table=el('table'),head=el('tr');['Biology condition','Decoder','Target agreement','Validation KL'].forEach(v=>head.append(el('th',v)));table.append(head);
   for(const [condition,actors] of Object.entries(a.matrix))for(const [decoder,r] of Object.entries(actors)){const row=el('tr');[condition,decoder,`${(r.agreement*100).toFixed(1)}%`,r.kl.toFixed(4)].forEach(v=>row.append(el('td',v)));table.append(row)}
   const scroll=el('div');scroll.className='table-scroll';scroll.append(table);details.append(scroll);
   details.append(el('p',`Timing proposal (not deployed): training NRMSE ${t.final.nrmse.toFixed(4)}; previously used low-luminance development NRMSE ${t.development_low_luminance.nrmse.toFixed(4)}. Waveform threshold: ${t.waveform_threshold_passed?'passed':'failed'}.`),el('p',Object.entries(t.tau_ms).map(([name,v])=>`${name}: ${v.toFixed(3)} ms`).join(' / ')),el('p',t.scope));panel.append(details);
   if(s.teacher_training){
    const teaching=s.teacher_training;
    const resumeCommand=$('relayPath').nextElementSibling;
    resumeCommand.textContent=resumeCommand.textContent.replace('--adaptive','--laya-student');
    panel.querySelector('h2').textContent='Laya teaching / eye-driven student';jump.textContent='Training evidence';
    panel.querySelector('p').textContent=`Laya taught existing relay connection gains and the action readout using image probabilities. During play, native Doom pixels enter the eye model and the anatomical neural circuit selects actions through its readout. Laya supplies no live policy input. Teaching scope: ${teaching.trained_scenarios.join(', ')}; teacher options: ${teaching.teacher_actions.join(', ')}.`;
    $('activeModel').textContent='Laya-taught brain / experimental';
    document.querySelector('.heading .intro').textContent='Laya teaches the neural circuit offline. During play, the trained circuit sees through its eye model and selects each action.';
    const feedbackRoute=document.querySelector('.path > span:last-child');
    feedbackRoute.replaceChildren(el('b','04 / OUTCOME'),document.createTextNode('Action outcome and recorded reward'));
    $('controlMethod').textContent='Native Doom image -> eye model -> anatomical visual and descending neuron states -> learned action readout. Laya is absent from this live decision path.';
    $('trainingMethod').textContent='The same frozen Laya image probabilities supervise both relay connection gains and the readout. The previous readout initializes training. No engine bounding boxes or scripted action targets are used in this teaching cycle. Visual physiology and time constants are retained.';
    $('evidenceScope').textContent='Only basic target shooting was taught in this cycle; the eight-output interface is retained. The table separates Laya alone, the taught brain, fixed connections, and the approved controller. Biological timing and general navigation are not validated.';
    $('selectionNote').textContent='The trained brain selects its highest-probability action using its own eye input. Compare with Laya is an optional same-image diagnostic.';
    $('feedbackNote').textContent='Frozen teacher-free play. Laya teaching happened before this session; optional live reward adaptation is a separate experiment.';
    document.querySelector('.new p').textContent='Restart plays a fresh episode with the selected trained brain. Start paused prepares an episode for inspection. Optional reward adaptation has its own saved relay state; its resume command appears below.';
    const optional=el('details');optional.append(el('summary','Optional reward adaptation / separate from Laya teaching'));
    optional.append(panel.querySelector('.controls'),$('relayStatus'),$('relayFeedback'),$('relayGains'));panel.append(optional);
    const command=el('pre','.\\.venv\\Scripts\\python.exe -m flydoom.laya_brain_training');
    const teachingDetails=el('details');teachingDetails.append(el('summary','Repeat Laya teaching'),el('p','Run another bounded training cycle from the retained parent, with fresh reserved evaluation openings. The teacher closes before student game tests. Existing checkpoints remain intact.'),command);panel.append(teachingDetails);
   }
   if(s.reward_training)showRewardTraining(panel,s.reward_training,s.game_gate_passed);
  }
  $('relayMode').textContent=s.running?'BOUNDED LEARNING':'PAUSED / FROZEN';
  $('relayStatus').textContent=`${s.completed_pairs} completed pairs · ${s.remaining_episodes} budgeted episodes remaining · ${s.active_groups} active gain groups · ${s.changed_groups} changed · live gain change from checkpoint (L2): ${s.gain_delta_l2.toExponential(3)}${s.trial?` · ${s.trial.side} trial / seed ${s.trial.seed}`:''}`;
  const f=s.last_feedback;$('relayFeedback').textContent=f?`Last pair: return ${f.plus_return.toFixed(2)} versus ${f.minus_return.toFixed(2)}; gain update ${f.gain_delta_l2.toExponential(3)}. ${f.updated?'An update occurred; benefit still requires evaluation.':'Equal returns produced no update.'}`:'No complete paired reward comparison yet.';
  $('relayGains').textContent=s.largest_live_gain_changes.map(g=>`${g.cell_type} / ${g.channel}: gain=${g.gain.toFixed(5)}`).join('; ');
  $('relayPath').textContent=s.live_state;
  $('relayStart').disabled=s.running||!latestStatus?.paused||latestStatus?.busy||latestStatus?.teacher_busy;
  $('relayScenario').disabled=s.running;$('relayPairs').disabled=s.running;
 }finally{adaptivePolling=false}
}
document.addEventListener('DOMContentLoaded',()=>{safe(adaptiveStatus)();setInterval(safe(adaptiveStatus),1000)});

function showRewardTraining(panel,r,gatePassed){
 panel.querySelector('h2').textContent='Laya initialization / learning from game rewards';
 panel.querySelector('p').textContent=`Laya first taught the eye-driven circuit. ${r.continuation?'This continuation added':'Then'} ${r.training_episodes_per_policy} training episodes per policy using real game rewards, with ${r.rehearsal_frames} retained teacher-labelled frames to limit forgetting. Both relay gains and the action decoder were trained. Run above is a frozen test of the resulting checkpoint.`;
 $('activeModel').textContent=`Laya + reward-trained brain / experimental / game gate ${gatePassed?'passed':'failed'}`;
 document.querySelector('.heading .intro').textContent='Initialized by Laya, trained from game outcomes, tested through its own eye and neural circuit.';
 $('trainingMethod').textContent=r.method+'. Updates occur between batches; rewards from later actions credit earlier decisions. Anatomy, visual rows and response times remain fixed.';
 $('evidenceScope').textContent='Matched tests compare joint relay/decoder reinforcement learning, decoder-only reinforcement learning, and the unchanged parent checkpoint. Training returns below include stochastic exploration; test results above use frozen greedy actions. Neither nonzero updates nor increasing training reward prove generalization.';
 $('relayPath').nextElementSibling.textContent=$('relayPath').nextElementSibling.textContent.replace('--laya-student','--reward-student');
 const legacy=$('relayStart').closest('details');legacy.querySelector('summary').textContent='Separate experiment / legacy paired relay perturbations';
 legacy.hidden=true;$('relayPath').closest('details').hidden=true;
 document.querySelector('.new p').textContent='Restart plays a fresh episode with this frozen reward-trained brain. Start paused prepares an episode for inspection. Training is a separate bounded experiment; both commands preserve the selected weights.';
 const evidence=el('section');evidence.id='rewardTrainingEvidence';
 if(r.continuation)evidence.append(el('p',`Continued checkpoint: ${r.parent}. ${r.continuation} Training caps: ${r.training_decisions} decisions / ${r.training_timeout_tics} tics per episode. Test caps: ${r.evaluation_decisions} decisions / ${r.evaluation_timeout_tics} tics. Longer training returns are not directly comparable to earlier shorter episodes.`));
 evidence.append(el('h3','Reward learning / measured batch updates'),el('p',`${r.accepted_batches}/${r.batches} candidate batches accepted by the training-loss and drift checks. ${r.changed_edges.toLocaleString()} existing edge weights changed; decoder change L2 ${r.decoder_change_l2.toFixed(5)}. These checks do not establish game improvement.`));
 const table=el('table'),head=el('tr');['Policy / batch','Train return','Hits','Relay change','Decoder change','Teacher KL'].forEach(v=>head.append(el('th',v)));table.append(head);
 for(const [name,history] of Object.entries(r.history))for(const row of history){const tr=el('tr');[`${name} / ${row.batch}`,row.mean_return.toFixed(1),`${row.kills}/${row.episodes}`,row.relay_change_l2.toExponential(2),row.decoder_change_l2.toExponential(2),row.teacher_train_kl_after.toFixed(4)].forEach(v=>tr.append(el('td',v)));table.append(tr)}
 const scroll=el('div');scroll.className='table-scroll';scroll.append(table);evidence.append(scroll);
 const credit=el('details');credit.id='rewardCredit';credit.append(el('summary','Inspect delayed reward credit / recorded training episodes'));
 const label=el('label','Training episode '),select=el('select');select.id='rewardEpisode';select.setAttribute('aria-label','Recorded reward training episode');
 const total=r.training_episodes_per_policy*Object.keys(r.history).length;
 for(let i=0;i<total;i++){const o=el('option',String(i+1));o.value=i;select.append(o)}label.append(select);credit.append(label);
 const facts=el('p'),trace=el('div');facts.id='rewardEpisodeFacts';trace.className='table-scroll';trace.style.maxHeight='360px';credit.append(facts,trace);
 const load=async()=>{const selected=select.value;const d=await api('/api/reward-episode?index='+selected);if(select.value!==selected)return;
  facts.textContent=`${d.policy} / seed ${d.seed} / ${d.kills} hits. ${d.scope}`;
  const t=el('table'),h=el('tr');['Decision','Action','Probability','Reward','Future return / 100','Predicted value','Credit before scaling'].forEach(v=>h.append(el('th',v)));t.append(h);
  for(const row of d.records){const tr=el('tr');[row.decision,row.action,`${(100*row.probability).toFixed(1)}%`,row.reward.toFixed(1),row.discounted_return.toFixed(4),row.predicted_value.toFixed(4),row.advantage_before_scaling.toFixed(4)].forEach(v=>tr.append(el('td',String(v))));t.append(tr)}trace.replaceChildren(t)};
 select.onchange=safe(load);credit.ontoggle=safe(async()=>{if(credit.open&&!trace.children.length)await load()});
 evidence.append(credit,el('p','Run and Restart test saved weights; they do not train. To continue learning from the latest completed weights, use the separate command below. It opens no browser. Episodes advance automatically within its finite batch budget; previous models are retained and the approved controller is unchanged.'),el('pre','.\\.venv\\Scripts\\python.exe -m flydoom.reward_training_v3'));
 panel.querySelector('.table-scroll').after(evidence);
}
