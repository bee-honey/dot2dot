// State-level UI regression checks. No browser, network, or external packages.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('dot2dot/web/static/index.html', 'utf8');
class Element {
  constructor(id='') { this.id=id; this.value=''; this.type=''; this.checked=false; this.disabled=false;
    this.style={}; this.children=[]; this.handlers={}; this.attrs={}; this.classList={add(){},remove(){}}; }
  addEventListener(name,cb) { this.handlers[name]=cb; }
  replaceChildren(...children) { this.children=[...children]; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  setAttribute(key,value) { this.attrs[key]=value; }
  getAttribute(key) { return this.attrs[key]; }
  querySelector() { return new Element(); }
  cloneNode() { const e=new Element(); e.attrs={...this.attrs}; return e; }
  setPointerCapture() {}
  getScreenCTM() { return {inverse(){return {};}}; }
}
const elements=new Map();
for (const tag of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
  const el=new Element(tag[1]);
  el.type=tag[0].match(/type="([^"]+)"/)?.[1] || '';
  el.value=tag[0].match(/value="([^"]+)"/)?.[1] || '';
  el.checked=/\bchecked\b/.test(tag[0]);
  elements.set(el.id,el);
}
const document={getElementById(id){assert.ok(elements.has(id),`Missing control ${id}`);return elements.get(id);},
  createElement:()=>new Element(),createElementNS:()=>new Element(),createTextNode:text=>({textContent:text}),
  querySelectorAll:()=>[],importNode:node=>node};
let response;
const sandbox={document,structuredClone,console,Blob,File:class{},
  FormData:class{constructor(){this.items=new Map();}append(k,v){this.items.set(k,v);}get(k){return this.items.get(k);}},
  URL:{createObjectURL:()=> 'blob:test'},crypto:{randomUUID:()=>String(Math.random())},
  performance:{now:()=>1000},indexedDB:{open(){throw Error('save unavailable in this unit harness');}},
  DOMParser:class{parseFromString(){return {documentElement:new Element()};}},
  DOMPoint:class{constructor(x,y){this.x=x;this.y=y;}matrixTransform(){return this;}},
  fetch:async(url)=>{if(url==='/api/config')return {json:async()=>({})};return {ok:true,json:async()=>structuredClone(response)};}
};
vm.createContext(sandbox);
vm.runInContext(html.split('<script>')[1].split('</script>')[0],sandbox);
const run=code=>vm.runInContext(code,sandbox);
function result(id) {return {id,width:100,height:100,source_png:'data:image/png;base64,x',edit_paths:[],warnings:[],
  puzzle_svg:`<svg>${id}</svg>`,solution_svg:`<svg>${id}</svg>`,check_png:'',style:'lineart',dots:20,lines:1,printed_details:0,
  quality:{overall:90,outline_coverage:1,detail_coverage:1,accuracy:1,crowded_labels:0},pdf_url:'/pdf',mystery_level:0,
  plan:null,guess_test:null,correction_checks:[],corrections:[]};}
(async()=>{
  sandbox.initial=result('original');
  run('chosenFile={name:"source.png"}; showResult(initial); appendVersion(initial,captureSettings());');
  // Mark a normalized region and add a description.
  await elements.get('feedback-canvas').handlers.pointerdown({clientX:20,clientY:30,pointerId:1,preventDefault(){}});
  await elements.get('feedback-canvas').handlers.pointermove({clientX:60,clientY:70});
  await elements.get('feedback-canvas').handlers.pointerup({});
  elements.get('feedback-note').value='Restore the eyes'; elements.get('feedback-action').value='restore';
  elements.get('add-feedback').onclick();
  assert.equal(run('feedbackRegions.length'),1);
  assert.equal(run('JSON.stringify(feedbackRegions[0].box)'),'[0.2,0.3,0.6,0.7]');
  response=result('revision');
  run('revisionRequest=true');
  await elements.get('form').handlers.submit({preventDefault(){}});
  assert.equal(run('lastResult.id'),'revision');
  assert.equal(run('versions.length'),1,'Unaccepted preview must not become a kept version');
  assert.ok(run('!!reviewBase'));
  elements.get('keep-revision').onclick();
  assert.equal(run('versions.length'),2); assert.equal(run('versions[1].feedback.length'),1);
  // Another revision is discarded; the accepted correction must survive.
  response=result('bad-revision'); run('revisionRequest=true');
  await elements.get('form').handlers.submit({preventDefault(){}});
  elements.get('discard-revision').onclick();
  assert.equal(run('lastResult.id'),'revision'); assert.equal(run('feedbackRegions.length'),1);
  elements.get('undo-revision').onclick();
  assert.equal(run('lastResult.id'),'original'); assert.equal(run('feedbackRegions.length'),0);
  assert.equal(run('reviewBase'),null);
  console.log('Feedback UI: region coordinates, whole-puzzle revision, keep, discard, and undo passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
