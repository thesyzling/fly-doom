"use strict";
const $ = id => document.getElementById(id);
const element = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
let meta, state, selected, inspected, requestNumber = 0, lastSequence = -1;
let zoom = 1, pan = {x: 0, y: 0}, hitNodes = [], dragging;
const colors = {fly: "#6c9eae", input: "#74e2c0", descending: "#f2cb79", student: "#c2a4ff", action: "#f0f3f5", previous: "#899aa3", memory: "#e5ad79"};
const shorten = id => id.length > 17 ? id.slice(0, 6) + "…" + id.slice(-5) : id;
const number = n => Number(n).toLocaleString("en-US");
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}
async function control(command) {
  try { await api("/api/control", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({command})}); error(""); }
  catch (e) { error(e.message); }
}
for (const command of ["run", "pause", "step", "stop"]) $(command).onclick = () => control(command);
$("search").onsubmit = event => { event.preventDefault(); choose($("rootId").value.trim()); };
$("resetView").onclick = () => { zoom = 1; pan = {x: 0, y: 0}; drawGraph(); };
function choose(id) { selected = id; zoom = 1; pan = {x: 0, y: 0}; inspect(); highlight(); }
function highlight() {
  document.querySelectorAll("[data-node]").forEach(e => e.classList.toggle("selected", e.dataset.node === selected));
}
async function inspect() {
  if (!selected || !state?.sequence) return;
  const token = ++requestNumber;
  try {
    const result = await api(`/api/neuron?id=${encodeURIComponent(selected)}&sequence=${state.sequence}`);
    if (token !== requestNumber) return;
    inspected = result;
    $("neuronTitle").textContent = result.label;
    $("connectionNote").textContent = result.connection_note;
    $("explanation").textContent = result.explanation;
    const fields = [["ID", result.id]];
    for (const [key, label, suffix] of [["super_class", "Cell superclass", ""], ["transmitter", "Transmitter annotation", ""],
      ["voltage_mv", "Endpoint voltage", " mV"], ["rest_mv", "Rest", " mV"], ["threshold_mv", "Threshold", " mV"],
      ["spikes", "Spikes in window", ""], ["incoming_count", "All incoming edges", ""], ["outgoing_count", "All outgoing edges", ""],
      ["input_pixel_bin", "Direct input bin (0–63)", ""], ["drive", "Normalized input drive", ""], ["probability", "Action probability", ""],
      ["logit", "Action logit", ""], ["bias", "Action bias", ""], ["value", "Input value", ""]]) {
      if (result[key] !== undefined && result[key] !== null && result[key] !== "") {
        const value = typeof result[key] === "number" ? Number.isInteger(result[key]) ? number(result[key]) : result[key].toFixed(4) : result[key];
        fields.push([label, value + suffix]);
      }
    }
    $("details").replaceChildren(...fields.map(([label, value]) => { const box = element("div", undefined, "detail"); box.append(element("small", label), element("strong", value)); return box; }));
    $("edges").replaceChildren(...result.edges.map(edge => {
      const row = element("tr");
      for (const endpoint of [edge.source, edge.target]) {
        const td = element("td"), button = element("button", endpoint.id); button.title = endpoint.label; button.onclick = () => choose(endpoint.id); td.append(button); row.append(td);
      }
      row.append(element("td", edge.weight.toFixed(5)), element("td", `${edge.channel} / ${edge.unit}${edge.current_logit_contribution !== undefined ? `; contribution ${edge.current_logit_contribution.toFixed(4)}` : ""}`));
      return row;
    }));
    $("edgeCount").textContent = `(${result.edges.length} shown)`;
    drawGraph(); drawHistory();
  } catch (e) { if (token === requestNumber) error(e.message); }
}
function neuronList(id, neurons) {
  $(id).replaceChildren(...neurons.map(neuron => {
    const button = element("button", undefined, "neuron-row"), label = element("span", neuron.label);
    button.dataset.node = neuron.id; label.append(element("small", neuron.id));
    button.append(label, element("span", `${neuron.spikes} spikes · ${neuron.voltage_mv.toFixed(1)} mV`));
    button.onclick = () => choose(neuron.id); return button;
  }));
}
function render(next) {
  state = next;
  const ended = ["completed", "stopped", "error"].includes(next.phase);
  $("status").textContent = ended ? `Run ${next.phase} · inspection available` : next.busy ? (next.paused ? "Finishing decision · then paused" : "Computing neural response…") : next.paused ? "Paused · ready to inspect" : "Running";
  $("run").disabled = ended || !next.paused; $("pause").disabled = ended || next.paused;
  $("step").disabled = ended || next.busy; $("stop").disabled = ended;
  if (next.error) error(next.error);
  $("results").replaceChildren(...next.finished_episodes.map((e, i) => element("p", `Episode ${i + 1}: ${e.kills} target kills · return ${e.return} · ${e.end_reason.replaceAll("_", " ")}`)));
  if (!next.sequence || next.sequence === lastSequence) return;
  lastSequence = next.sequence;
  if (next.memory) document.querySelectorAll(".memory-input").forEach((button, i) => { button.lastChild.textContent = next.memory[i].toFixed(3); });
  $("doom").src = next.frame;
  $("episode").textContent = `EP ${next.episode}/${meta.episodes} · DECISION ${next.decision}`;
  $("chosen").textContent = next.action.replaceAll("_", " "); $("reward").textContent = `Return ${next.return}`;
  $("actions").replaceChildren(...meta.actions.map((action, i) => {
    const button = element("button", undefined, "action"), fill = element("i", undefined, "fill"); fill.style.width = `${next.probabilities[i] * 100}%`;
    button.dataset.node = "action:" + action; button.append(fill, element("span", action.replaceAll("_", " ")), element("span", next.decision ? `${(next.probabilities[i] * 100).toFixed(1)}%` : "—"));
    button.onclick = () => choose(button.dataset.node); return button;
  }));
  next.student_spikes.forEach((spikes, i) => {
    const cell = $("cells").children[i]; cell.style.backgroundColor = `rgb(${37 + spikes * 19}, ${32 + spikes * 14}, ${49 + spikes * 24})`;
    cell.title = `Added cell ${i + 1}: ${spikes}/8 spikes. Click to inspect its learned connections.`;
    cell.setAttribute("aria-label", cell.title);
  });
  neuronList("active", next.active); neuronList("descending", next.descending);
  const metrics = [["Whole-graph spikes", number(next.brain_spikes)], ["Neural window", `${next.brain_ms} ms`], ["Wall time / decision", `${next.brain_compute_seconds.toFixed(2)} s`], ["Lowest voltage in window", `${next.voltage_min_mv.toFixed(2)} mV`], ["Decision source", "Added spiking group"]];
  $("metrics").replaceChildren(...metrics.flatMap(([key, value]) => [element("dt", key), element("dd", value)]));
  if (!selected) selected = next.descending[0]?.id;
  highlight(); inspect();
}
function canvasContext(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {canvas, ctx, w: rect.width, h: rect.height};
}
function drawGraph() {
  const {ctx, w, h} = canvasContext("graph");
  if (!inspected) { ctx.fillStyle = "#92a9b4"; ctx.font = "13px Segoe UI"; ctx.fillText("Select a fly neuron or an added cell to explore.", 24, 40); return; }
  ctx.fillStyle = "#18303a";
  for (let x = 16; x < w; x += 22) for (let y = 16; y < h; y += 22) ctx.fillRect(x, y, 1, 1);
  const incoming = new Map(), outgoing = new Map(), id = inspected.id;
  for (const edge of inspected.edges) {
    if (edge.target.id === id && edge.source.id !== id) incoming.set(edge.source.id, edge.source);
    if (edge.source.id === id && edge.target.id !== id) outgoing.set(edge.target.id, edge.target);
  }
  const coords = new Map([[id, {x: 500, y: 240, node: inspected}]]);
  const place = (nodes, x) => Array.from(nodes.values()).forEach((node, i, all) => coords.set(node.id, {x, y: 45 + (i + .5) * 380 / Math.max(1, all.length), node}));
  place(incoming, 120); place(outgoing, 880);
  // A node occurring on both sides gets one position; all real directed edges remain.
  const scale = Math.min(w / 1000, h / 480) * zoom;
  ctx.save(); ctx.translate(w / 2 + pan.x, h / 2 + pan.y); ctx.scale(scale, scale); ctx.translate(-500, -240);
  ctx.font = "12px Segoe UI"; ctx.fillStyle = "#809ba6"; ctx.textAlign = "center";
  ctx.fillText("INCOMING", 120, 22); ctx.fillText("SELECTED", 500, 22); ctx.fillText("OUTGOING", 880, 22);
  for (const edge of inspected.edges) {
    const source = coords.get(edge.source.id), target = coords.get(edge.target.id);
    if (!source || !target) continue;
    ctx.strokeStyle = edge.weight > 0 ? "#5eb899" : edge.weight < 0 ? "#d17a84" : "#4b5c64"; ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1.4; ctx.globalAlpha = .65;
    if (source === target) { ctx.beginPath(); ctx.arc(source.x, source.y - 22, 28, .2, Math.PI * 1.8); ctx.stroke(); continue; }
    const dx = target.x - source.x, dy = target.y - source.y;
    const angle = Math.atan2(dy, dx), chosen = edge.target.id === id;
    const boundary = Math.min((chosen ? 114 : 99) / Math.max(.001, Math.abs(dx)), (chosen ? 39 : 14) / Math.max(.001, Math.abs(dy)));
    const end = {x: target.x - dx * boundary, y: target.y - dy * boundary};
    ctx.beginPath(); ctx.moveTo(source.x, source.y); ctx.lineTo(end.x, end.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(end.x, end.y); ctx.lineTo(end.x - 9 * Math.cos(angle - .4), end.y - 9 * Math.sin(angle - .4)); ctx.lineTo(end.x - 9 * Math.cos(angle + .4), end.y - 9 * Math.sin(angle + .4)); ctx.closePath(); ctx.fill();
  }
  ctx.globalAlpha = 1; hitNodes = [];
  for (const [key, point] of coords) {
    const isSelected = key === id, width = isSelected ? 222 : 192, height = isSelected ? 72 : 22;
    ctx.fillStyle = isSelected ? "#263541" : "#101f28"; ctx.strokeStyle = colors[point.node.kind] || colors.fly; ctx.lineWidth = isSelected ? 2.5 : 1;
    ctx.beginPath(); ctx.roundRect(point.x - width / 2, point.y - height / 2, width, height, 5); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#e5edf0"; ctx.font = isSelected ? "bold 15px Segoe UI" : "12px Segoe UI";
    ctx.fillText(point.node.label.slice(0, 26), point.x, point.y + (isSelected ? -12 : 4));
    if (isSelected) { ctx.fillStyle = colors[point.node.kind] || colors.fly; ctx.font = "11px Consolas"; ctx.fillText(shorten(key), point.x, point.y + 8); ctx.fillText(`${point.node.spikes ?? "—"} spikes`, point.x, point.y + 26); }
    hitNodes.push({id: key, label: point.node.label, x: w / 2 + pan.x + (point.x - 500) * scale, y: h / 2 + pan.y + (point.y - 240) * scale, width: width * scale, height: height * scale});
  }
  ctx.restore();
}
function drawHistory() {
  const {ctx, w, h} = canvasContext("history"), history = inspected?.history || [];
  ctx.fillStyle = "#92a9b4"; ctx.font = "10px Segoe UI";
  if (!history.length) { ctx.fillText("Select a neuron to see its spike history.", 5, 40); return; }
  const max = Math.max(1, ...history.map(s => s.spikes)), step = (w - 25) / history.length;
  history.forEach((s, i) => { const height = s.spikes / max * (h - 35); ctx.fillStyle = colors[inspected.kind] || colors.fly; ctx.fillRect(22 + i * step, h - 20 - height, Math.max(2, step - 3), Math.max(1, height)); if (history.length < 13 || i % 4 === 0) { ctx.fillStyle = "#92a9b4"; ctx.fillText(s.decision, 22 + i * step, h - 5); } });
  ctx.fillStyle = "#92a9b4"; ctx.fillText(max, 0, 15); ctx.fillText("0", 0, h - 20);
  $("historyLabel").textContent = `${history.length} observations · x: decision · y: spikes`;
}
const graph = $("graph");
graph.addEventListener("pointerdown", event => { dragging = {x: event.clientX, y: event.clientY, original: {...pan}, moved: false}; graph.setPointerCapture(event.pointerId); });
graph.addEventListener("pointermove", event => {
  const rect = graph.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  if (dragging) { const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y; dragging.moved ||= Math.abs(dx) + Math.abs(dy) > 5; pan = {x: dragging.original.x + dx, y: dragging.original.y + dy}; drawGraph(); }
  const hit = hitNodes.find(n => Math.abs(n.x - x) < n.width / 2 && Math.abs(n.y - y) < n.height / 2);
  graph.title = hit ? `${hit.label}\n${hit.id}\nClick to inspect` : "Drag to pan · Scroll to zoom";
});
graph.addEventListener("pointerup", event => {
  if (!dragging?.moved) { const rect = graph.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top; const hit = hitNodes.find(n => Math.abs(n.x - x) < n.width / 2 && Math.abs(n.y - y) < n.height / 2); if (hit) choose(hit.id); }
  dragging = null;
});
graph.addEventListener("pointercancel", () => { dragging = null; });
graph.addEventListener("wheel", event => { event.preventDefault(); zoom = Math.max(.5, Math.min(3, zoom * Math.exp(-event.deltaY * .001))); drawGraph(); }, {passive: false});
new ResizeObserver(() => { drawGraph(); drawHistory(); }).observe($("graph").parentElement);
async function poll() {
  try { render(await api("/api/state")); }
  catch (e) { $("status").textContent = "Connection lost"; error(`${e.message}. Keep the terminal command running, then reload this page.`); }
  setTimeout(poll, 500);
}
async function start() {
  try {
    meta = await api("/api/meta");
    $("graphSize").textContent = `${number(meta.neurons)} cells · ${number(meta.connections)} edges`;
    $("readoutSize").textContent = `${meta.outputs} outputs → ${meta.hidden} added cells`;
    $("outputPath").textContent = `Reports: ${meta.output}`;
    $("memoryPanel").hidden = !meta.memory_features?.length;
    for (const name of meta.memory_features || []) {
      const button = element("button", undefined, "neuron-row memory-input");
      button.dataset.node = `memory:${name}`;
      button.append(element("span", name.replaceAll("_", " ")), element("span", "0.000"));
      button.onclick = () => choose(button.dataset.node); $("memoryInputs").append(button);
    }
    for (let i = 0; i < meta.hidden; i++) { const button = element("button", String(i + 1).padStart(2, "0"), "cell"); button.dataset.node = `student:${i}`; button.onclick = () => choose(button.dataset.node); $("cells").append(button); }
    await poll();
  } catch (e) { error(e.message); }
}
start();
