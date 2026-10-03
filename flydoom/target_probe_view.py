"""Standalone inspection of the target-side diagnostic; no live policy control."""

import json


def write_view(output, report, records, brightness):
    data = {"analysis": report["analysis"], "records": records, "brightness": brightness.tolist()}
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Target-side diagnostic | Fly Doom</title>
<style>
*{box-sizing:border-box}body{font:16px system-ui;margin:0;background:#f4f2ed;color:#252b2b}
main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:8px}
p{line-height:1.55;max-width:900px}.muted{color:#555f5b}section{background:white;border:1px solid #d6dad5;padding:20px;margin:20px 0;border-radius:6px}
.views{display:grid;grid-template-columns:2fr 1fr;gap:22px}canvas{width:100%;height:auto;background:#14191b}
button,select{font:inherit;padding:9px 14px;border:1px solid #919b95;background:#fff;border-radius:4px}
button:disabled{opacity:.4}input{width:100%;margin:18px 0}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #deded8}
.table{overflow:auto}#detail{min-height:3em}#predictions{line-height:1.9}a{color:#165f4a}
@media(max-width:700px){main{padding:14px}.views{grid-template-columns:1fr}h1{font-size:24px}section{padding:14px}}
</style>
<main><div class="muted">FLY DOOM / EXPERIMENT INSPECTOR</div><h1>Where is the target?</h1>
<p>We measure whether a small diagnostic classifier can recover LEFT or RIGHT from three stages of the same observation. The playing model and fly connections were not trained or changed.</p>
<section><h2>Held-out episode results</h2><p class="muted">Balanced accuracy averages LEFT and RIGHT recall. A constant-side baseline scores 50%. Near-center targets and exact repeated RGB images are excluded from scoring. These are development results on one map, not game wins.</p>
<div class="table"><table><thead><tr><th>Representation</th><th>Features</th><th>Training</th><th>Validation</th><th>Shuffled labels (mean)</th></tr></thead><tbody id="metrics"></tbody></table></div><p id="counts"></p></section>
<section><div class="controls"><label for="split">Collection</label><select id="split"><option value="all">All observations</option><option value="train">Training episodes</option><option value="validation" selected>Validation episodes</option></select><button id="previous">Previous</button><button id="next">Next</button><span id="position"></span></div>
<input id="slider" type="range" min="0" value="0" aria-label="Observation">
<div class="views"><div><h2>Game image and scoring label</h2><canvas id="frame" width="640" height="480" aria-label="Recorded game frame with target bounding box"></canvas></div><div><h2>Actual fly input: 8 × 8</h2><canvas id="grid" width="320" height="320" aria-label="Coarse brightness features supplied to the fly network"></canvas><p class="muted">Only these 64 brightness values enter the fly graph. The green target box is an evaluation overlay; the network never receives it.</p></div></div>
<p id="detail"></p><div id="predictions"></div></section>
<p class="muted">Graph state persists for four observations and resets between episodes. Probes use no previous-action or memory features. Decodability does not show that the playing readout uses the information. Poor decoding does not prove information is absent. Similar backgrounds can remain across episodes.</p>
<p><a href="report.json">Full measured report</a> · <a href="plan.json">Plan recorded before collection</a> · <a href="https://vizdoom.farama.org/main/api/python/gameState/">ViZDoom label definitions</a></p></main>
<script>
const data=__DATA__;
const names={rgb_8x8:'RGB image bins',brightness_8x8:'Fly brightness input',descending_outputs:'Fly descending outputs'};
const el=id=>document.getElementById(id),pct=x=>(100*x).toFixed(1)+'%';
for(const [name,result] of Object.entries(data.analysis.representations)){
const row=document.createElement('tr');
for(const value of [names[name],result.dimensions,pct(result.train.balanced_accuracy),pct(result.validation.balanced_accuracy),pct(result.shuffled_training_label_control.mean_balanced_accuracy)]){
const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell)}el('metrics').appendChild(row)}
const counts=data.analysis.representations.brightness_8x8.validation.class_counts_left_right;
el('counts').textContent=`Scored validation images: ${counts[0]} LEFT, ${counts[1]} RIGHT. No classifier settings were selected using these results.`;
let indices=[],position=0,version=0;
function render(){const index=indices[position],r=data.records[index],token=++version;
el('slider').value=position;el('position').textContent=`${position+1} / ${indices.length}`;
el('previous').disabled=position===0;el('next').disabled=position===indices.length-1;
const scored=data.analysis.indices[r.split].includes(index),side=r.target.side<0?r.target.reason:(r.target.side===0?'LEFT':'RIGHT');
el('detail').textContent=`Seed ${r.seed} · observation ${r.step+1} · ${r.split} · target: ${side} · ${scored?'included in metrics':'excluded from metrics (center, missing target, or repeated image)'}`;
const canvas=el('frame'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,640,480);
const img=new Image();img.onload=()=>{if(token!==version)return;ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,640,480);
if(r.target.box){const [x,y,w,h]=r.target.box;ctx.strokeStyle='#74f2ac';ctx.lineWidth=3;ctx.strokeRect(x*2,y*2,w*2,h*2)}};img.onerror=()=>{el('detail').textContent+=' · IMAGE LOAD FAILED'};img.src=r.image;
const grid=el('grid').getContext('2d');data.brightness[index].forEach((value,i)=>{const v=Math.round(value*255);grid.fillStyle=`rgb(${v},${v},${v})`;grid.fillRect((i%8)*40,Math.floor(i/8)*40,40,40);grid.strokeStyle='#787878';grid.strokeRect((i%8)*40,Math.floor(i/8)*40,40,40)});
el('predictions').replaceChildren();for(const [name,result] of Object.entries(data.analysis.representations)){
const line=document.createElement('div');line.textContent=`${names[name]} → diagnostic prediction: ${result.scores[index]>=0?'RIGHT':'LEFT'}`;el('predictions').appendChild(line)}}
function filter(){indices=data.records.map((r,i)=>[r,i]).filter(([r])=>el('split').value==='all'||r.split===el('split').value).map(([,i])=>i);position=0;el('slider').max=indices.length-1;render()}
el('split').onchange=filter;el('slider').oninput=()=>{position=Number(el('slider').value);render()};
el('previous').onclick=()=>{if(position>0){position--;render()}};el('next').onclick=()=>{if(position<indices.length-1){position++;render()}};filter();
</script></html>'''
