"use strict";

// Reported evidence belongs to a frozen cycle; live decoder updates are separate.
async function researchStatus(){
 const response=await api('/api/research'),latest=response.latest_report,active=response.active_report;
 const host=$('researchStatus');if(!host)return;
 const progress=response.progress;
 host.textContent=progress?`${progress.status.toUpperCase()} / stage ${progress.stage||'—'} of 5`:'No sensorimotor cycle yet';
 $('checkpointRole').textContent=response.active_checkpoint===response.champion.path?'ACTIVE: APPROVED CHECKPOINT':'ACTIVE: EXPERIMENTAL PREVIEW';
 $('activeModel').textContent=response.active_checkpoint===response.champion.path?'Approved checkpoint':'Experimental candidate / promotion gate not passed';
 $('activeModel').dataset.experimental=String(response.active_checkpoint!==response.champion.path);
 const stages=$('researchStages');stages.replaceChildren();
 const names=['Existing-edge learning','Measured timing','Descending pathway','Independent benchmark','Persistent learning'];
 names.forEach((name,i)=>{const row=el('article');row.append(el('span',`0${i+1}`),el('strong',name),el('small',latest?.phases_completed?.includes(i+1)?'Completed':progress?.stage===i+1?'Running':'Pending'));stages.append(row)});
 $('researchPaths').textContent=`Active: ${response.active_checkpoint}\nChampion: ${response.champion.path}\nLatest candidate: ${response.latest?.path||'none'}`;
 if(!latest){$('researchSummary').textContent=progress?.message||'Run one bounded cycle from the command below.';return}
 const timing=latest.timing,fit=latest.distillation;
 $('researchSummary').textContent=`Latest cycle: ${latest.plasticity.changed_edges.toLocaleString()} existing edges changed across ${latest.plasticity.changed_groups} groups. ${latest.neurons.toLocaleString()} simulated cells, ${latest.readout_cells} descending outputs. Teacher validation KL ${fit.final.validation.kl.toFixed(4)}, agreement ${(fit.final.validation.agreement*100).toFixed(1)}%. Timing development NRMSE ${timing.development_low_luminance.nrmse.toFixed(4)}; waveform threshold ${timing.waveform_threshold_passed?'passed':'failed'}. ${latest.promotion.accepted?'Candidate promoted.':'Candidate rejected by the game gate; champion retained.'}`;
 if(latest.aim_training)$('researchSummary').textContent=`Aiming correction: ${latest.aim_training.training_frames} training frames, selected round ${latest.aim_training.selected_round}. Training-only engine annotations supervise the action decoder; live inputs remain descending neuron states. Biological weights and timing are inherited unchanged. ${latest.promotion.accepted?'Basic-scenario gate passed.':'Basic-scenario gate failed; champion retained.'}`;
 const table=$('researchResults');table.replaceChildren();
 for(const [policy,tasks] of Object.entries(latest.test_by_task))for(const [task,result] of Object.entries(tasks)){
  const row=el('tr');[policy==='parent'&&latest.aim_training?'approved baseline':policy.replaceAll('_',' '),task.replaceAll('_',' '),result.episodes,result.kills,`${(result.kill_episode_rate*100).toFixed(1)}%`,result.mean_return.toFixed(1)].forEach(v=>row.append(el('td',String(v))));table.append(row)
 }
 $('researchScope').textContent=latest.scope+' The low-luminance recordings were used in earlier development; they are not new biological validation. Test results do not select the champion.';
 $('researchGate').textContent=Object.entries(latest.promotion.by_task).map(([name,g])=>`${name}: gate return Δ ${g.return_delta.toFixed(1)}, kills Δ ${g.kill_delta}`).join(' / ');
 const fitDetails=$('researchFit');fitDetails.replaceChildren();
 const tau=el('p',Object.entries(timing.tau_ms).map(([name,value])=>`${name}: ${value.toFixed(3)} ms`).join(' / '));
 fitDetails.append(tau,el('p',`Fitted input adaptation: gain ${timing.adaptation_gain.toFixed(3)}, time constant ${timing.adaptation_tau_ms.toFixed(2)} ms, dark-contrast gain ${timing.dark_gain.toFixed(3)}. Numerical half-step difference ${(timing.fine_dt_relative_rms*100).toFixed(2)}%. These are conditional model parameters.`));
 if(response.latest_timing){const d=response.latest_timing;
  for(let cell=0;cell<2;cell++){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 600 180');svg.setAttribute('role','img');svg.setAttribute('aria-label',`${cell?'L2':'L1'} measured and fitted flash response`);fitDetails.append(svg);
   const values=[...d.measured[cell].flat(),...d.predicted[cell].flat()],lo=Math.min(...values),hi=Math.max(...values),range=Math.max(hi-lo,1e-8),end=d.time_ms.at(-1);
   svgElement(svg,'text',{x:45,y:15,'font-size':11,fill:'#637568'},`${cell?'L2':'L1'} / high-luminance training / 0–${end.toFixed(0)} ms / negative baseline-centered dF/F`);
   svgElement(svg,'line',{x1:45,x2:575,y1:145-(0-lo)/range*115,y2:145-(0-lo)/range*115,stroke:'#c7d1c0'});
   for(let contrast=0;contrast<2;contrast++)for(const kind of ['measured','predicted'])svgElement(svg,'polyline',{points:d.time_ms.map((t,i)=>`${45+530*t/end},${145-(d[kind][cell][contrast][i]-lo)/range*115}`).join(' '),fill:'none',stroke:contrast?'#285940':'#a36732','stroke-width':1.8,'stroke-dasharray':kind==='predicted'?'5 4':'none'});
   svgElement(svg,'text',{x:45,y:173,'font-size':10,fill:'#637568'},'Solid: measured / dashed: fitted / orange: dark / green: light');
  }
 }
 const changed=latest.plasticity.groups.map((name,i)=>({name,gain:latest.plasticity.gains[i]})).sort((a,b)=>Math.abs(b.gain-1)-Math.abs(a.gain-1)).slice(0,12);
 fitDetails.append(el('p','Largest type-pair gain changes: '+changed.map(g=>`${g.name} × ${g.gain.toFixed(3)}`).join('; ')));
 $('researchActive').textContent=active?'The neural map and game above use the sensorimotor candidate shown in Active.':'The game above uses the approved checkpoint. Launch with --candidate to inspect the latest candidate.';
 if(active){$('readoutLabel').textContent='descending decoder inputs';$('graphRoute').textContent='R1–6 · visual relays · descending cells';
  $('decisionCaption').textContent='Largest signed descending-cell contributions to the chosen action. Each root ID belongs to the anatomical graph; the Doom readout is engineered.';
  $('controlMethod').textContent='Only selected descending-cell states enter the action decoder. The path follows existing FAFB contacts through visual and central relays. Laya supplies image-based teaching targets; no raw pixels or engine positions enter the decoder.';
  $('trainingMethod').textContent='Bounded existing-edge gains and population timing are trained between episodes. Live reward updates the engineered decoder only. A saved learner can be resumed explicitly; automatic cycles retain the champion unless their paired gate passes.';
  $('evidenceScope').textContent=active.scope;
  if(active.aim_training){$('controlMethod').textContent='Only descending-cell states enter the action decoder. The inherited Laya-trained circuit is retained. Aiming correction uses engine target annotations during training only; live inference has no access to those labels.';$('trainingMethod').textContent='Frozen evaluation is the default. Explore & learn from reward samples exploratory actions and updates a separate live decoder. Recorded game outcomes, not agreement with Laya, select the aiming correction.'}
  if(!$('population').querySelector('option[value="Descending"]')){const option=el('option','Descending');option.value='Descending';$('population').prepend(option);$('population').value='Descending'}
 }
}
document.addEventListener('DOMContentLoaded',()=>{safe(researchStatus)();setInterval(safe(researchStatus),10000)});
