const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('flydoom/web/laboratory/app.js', 'utf8');
const chartSource = source.slice(source.indexOf('function chart(report){'), source.indexOf('async function training(){'));
function node(tag) {
  return {tag, children: [], attributes: {}, replaceChildren() {this.children=[];},
    append(child) {this.children.push(child);}, setAttribute(key,value) {this.attributes[key]=value;}};
}
const svg = node('svg');
const sandbox = {document: {createElementNS: (ns,tag) => node(tag)}, $: () => svg,
  fmt: (value, places) => Number(value).toFixed(places)};
vm.createContext(sandbox); vm.runInContext(chartSource, sandbox);
for (const history of [
  [{evaluation: 0, kl: .3, accepted: true, gains: [1]}, {evaluation: 1, kl: .2, gains: [.99]}],
  [{evaluation: 0, kl: .3, accepted: true, method: 'champion', objective: .3},
   {evaluation: 1, kl: .299, accepted: true, method: 'sparse_consensus', objective: .299},
   {evaluation: 2, kl: .301, accepted: false, method: 'sparse_consensus', objective: .301}]
]) {
  sandbox.chart({history});
  const circles = svg.children.filter(n => n.tag === 'circle');
  assert.equal(circles.length, history.length);
  assert(circles.every(n => Number.isFinite(n.attributes.cx) && Number.isFinite(n.attributes.cy)));
  assert(circles.every(n => n.children[0].textContent.includes('KL')));
}
console.log('Legacy and sparse-consensus loss charts passed.');
