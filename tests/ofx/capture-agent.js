/* Process-only OFX observer. Load before the selected plug-in's setHost and
 * DescribeInContext/CreateInstance. rpc.exports.arm selects one render. Image
 * bytes are logical rows in ascending y, tightly packed little-endian FP32;
 * binary messages carry {event:'pixels',kind,offset,total}. Captures are
 * invasive: the host CUDA stream is synchronized before each image copy. */
'use strict';

const P = Process.pointerSize;
if (P !== 8 || Process.arch !== 'x64') throw new Error('capture requires Windows x64');
const types = new Map(), parameters = new Map(), clips = new Map(), effects = new Map();
const actions = new Map(), images = new Map(), installed = new Set();
let property = null, effectSuite = null, paramSuite = null, armed = null, serial = 0;
const key = p => p.toString();
const str = p => p.isNull() ? '' : p.readUtf8String();
const slot = (p, n) => p.add(n * P).readPointer();
const cstr = s => Memory.allocUtf8String(s);
const notifyError = (where, error) => send({event: 'error', where, error: String(error), stack:error.stack});
function attachOnce(p, callbacks) {
  const k = key(p); if (p.isNull() || installed.has(k)) return;
  const guarded={};
  for(const phase of ['onEnter','onLeave']) if(callbacks[phase]) guarded[phase]=function(value) {
    try { callbacks[phase].call(this,value); }
    catch(e) { const f=active(); if(f)f.failed=true; notifyError('hook '+phase,e); }
  };
  Interceptor.attach(p, guarded); installed.add(k);
}
function active() { return actions.get(Process.getCurrentThreadId()); }
function effectState(p) {
  const k = key(p); if (!effects.has(k)) effects.set(k, {params: new Map(), clips: new Map(), history: [], create_seen: false});
  return effects.get(k);
}
function getProperty(handle, name, kind, count = 1, required = true) {
  const out = [], n = cstr(name);
  const index = {pointer: 8, string: 9, double: 10, int: 11}[kind];
  const fn = new NativeFunction(slot(property, index), 'int', ['pointer','pointer','int','pointer']);
  for (let i = 0; i < count; ++i) {
    const v = Memory.alloc(8); v.writeU64(0);
    const status = fn(handle, n, i, v);
    if (status !== 0) { if (required) throw new Error(name + '[' + i + '] status ' + status); return null; }
    out.push(kind === 'pointer' ? v.readPointer() : kind === 'string' ? str(v.readPointer()) : kind === 'double' ? v.readDouble() : v.readS32());
  }
  return count === 1 ? out[0] : out;
}
function doubleBits(value) { const p = Memory.alloc(8); p.writeDouble(value); return p.readU64().toString(16).padStart(16, '0'); }
function hasProperty(handle, name) {
  const dimension=Memory.alloc(4); dimension.writeS32(0);
  const fn=new NativeFunction(slot(property,17),'int',['pointer','pointer','pointer']);
  return fn(handle,cstr(name),dimension)===0 && dimension.readS32()>0;
}
function snapshotParams(state) {
  const result = {}, get = slot(paramSuite, 4);
  for (const [name, param] of state.params) {
    const type = types.get(name);
    if (!type) throw new Error('parameter type not observed: ' + name);
    if (type === 'OfxParamTypePushButton' || type === 'OfxParamTypeGroup' || type === 'OfxParamTypePage') continue;
    const dimensions = type.endsWith('3D') || type === 'OfxParamTypeRGB' ? 3 : type.endsWith('2D') ? 2 : 1;
    const isDouble = type.startsWith('OfxParamTypeDouble') || type === 'OfxParamTypeRGB' || type === 'OfxParamTypeRGBA';
    const isString = type === 'OfxParamTypeString' || type === 'OfxParamTypeCustom' || type === 'OfxParamTypeStrChoice';
    const count = type === 'OfxParamTypeRGBA' ? 4 : dimensions;
    const slots = Array.from({length: count}, () => { const p = Memory.alloc(8); p.writeU64(0); return p; });
    const fn = new NativeFunction(get, 'int', ['pointer','...', ...slots.map(() => 'pointer')]);
    const frame = active(); if (frame) frame.observer_getter = true;
    let status;
    try { status = fn(param, ...slots); } finally { if (frame) frame.observer_getter = false; }
    const values = status === 0 ? slots.map(p => isDouble ? p.readDouble() : isString ? str(p.readPointer()) : p.readS32()) : [];
    result[name] = {type, values, status};
    if (status === 0 && isDouble) result[name].f64_bits = values.map(doubleBits);
  }
  return result;
}
function imageMetadata(h) {
  const g = (name, kind, count = 1, required = true) => getProperty(h, name, kind, count, required);
  return {bounds:g('OfxImagePropBounds','int',4), rod:g('OfxImagePropRegionOfDefinition','int',4),
    row_bytes:g('OfxImagePropRowBytes','int'), components:g('OfxImageEffectPropComponents','string'),
    depth:g('OfxImageEffectPropPixelDepth','string'), premultiplication:g('OfxImageEffectPropPreMultiplication','string'),
    pixel_aspect:g('OfxImagePropPixelAspectRatio','double'), scale:g('OfxImageEffectPropRenderScale','double',2),
    field:g('OfxImagePropField','string',1,false), unique_id:g('OfxImagePropUniqueIdentifier','string',1,false),
    data:g('OfxImagePropData','pointer')};
}
function copyImage(frame, kind, m) {
  if (m.depth !== 'OfxBitDepthFloat') throw new Error('image depth is not float');
  const channels = m.components === 'OfxImageComponentRGBA' ? 4 : m.components === 'OfxImageComponentRGB' ? 3 : 0;
  const width = m.bounds[2]-m.bounds[0], height = m.bounds[3]-m.bounds[1], row = width*channels*4;
  if (!channels || width <= 0 || height <= 0 || Math.abs(m.row_bytes) < row || m.data.isNull()) throw new Error('invalid image layout');
  const cuda = Process.getModuleByName('nvcuda.dll');
  const sync = new NativeFunction(cuda.getExportByName('cuStreamSynchronize'), 'int', ['pointer']);
  const ctxGet = new NativeFunction(cuda.getExportByName('cuCtxGetCurrent'), 'int', ['pointer']);
  const context = Memory.alloc(P), ctxStatus = ctxGet(context);
  if (ctxStatus !== 0 || context.readPointer().isNull()) throw new Error('no current CUDA context');
  const syncStatus = sync(frame.stream);
  if (syncStatus !== 0) throw new Error('cuStreamSynchronize status ' + syncStatus);
  const copy = new NativeFunction(cuda.getExportByName('cuMemcpyDtoH_v2'), 'int', ['pointer','pointer','size_t']);
  const rowsPerChunk = Math.max(1, Math.floor(4 * 1024 * 1024 / row));
  const total = row * height;
  for (let y = 0; y < height; y += rowsPerChunk) {
    const rows = Math.min(rowsPerChunk, height-y), buffer = Memory.alloc(rows * row);
    for (let j=0; j<rows; ++j) {
      const status = copy(buffer.add(j*row), m.data.add((y+j)*m.row_bytes), row);
      if (status !== 0) throw new Error('cuMemcpyDtoH status ' + status + ' row ' + (y+j));
    }
    send({event:'pixels', capture_id:frame.id, kind, offset:y*row, total}, buffer.readByteArray(rows*row));
  }
  const current=key(context.readPointer());
  if(frame.cuda_context && frame.cuda_context!==current) throw new Error('CUDA context changed between images');
  frame[kind + '_copied'] = true;
  frame.cuda_context = current;
}
function registerSuites(name, suite) {
  if (suite.isNull()) return;
  // Frida recycles the onLeave return-value wrapper after the callback.
  // Keep an independent pointer value for the lifetime of the host suites.
  suite = ptr(suite.toString());
  if (name === 'OfxPropertySuite') property = suite;
  if (name === 'OfxParameterSuite') {
    paramSuite = suite;
    attachOnce(slot(suite,0), {onEnter(a){this.frame=active();this.type=str(a[1]);this.name=str(a[2]);},onLeave(r){if(this.frame && r.toInt32()===0) types.set(this.name,this.type);}});
    attachOnce(slot(suite,1), {onEnter(a){this.name=str(a[1]);this.out=a[2];this.frame=active();},onLeave(r){if(r.toInt32()===0 && this.frame && !this.out.isNull()){const p=this.out.readPointer();parameters.set(key(p),this.name);effectState(this.frame.effect).params.set(this.name,p);}}});
    for (const index of [4,5]) attachOnce(slot(suite,index), {
      onEnter(a){
        this.frame=active(); this.name=parameters.get(key(a[0])); this.index=index;
        if (!this.frame || this.frame.observer_getter || !this.name) return;
        const type=types.get(this.name); this.type=type;
        if (!type) return;
        const count=type==='OfxParamTypeRGBA'?4:(type.endsWith('3D')||type==='OfxParamTypeRGB')?3:type.endsWith('2D')?2:1;
        this.outputs=[]; for(let i=0;i<count;++i)this.outputs.push(a[(index===4?1:2)+i]);
      },
      onLeave(r){
        if(!this.outputs || !this.frame.trace)return;
        const status=r.toInt32(), type=this.type;
        const isDouble=type.startsWith('OfxParamTypeDouble')||type==='OfxParamTypeRGB'||type==='OfxParamTypeRGBA';
        const isString=type==='OfxParamTypeString'||type==='OfxParamTypeStrChoice'||type==='OfxParamTypeCustom';
        try {
          const values=status===0?this.outputs.map(p=>isDouble?p.readDouble():isString?str(p.readPointer()):p.readS32()):[];
          const entry={action:this.frame.action,name:this.name,type,getter:this.index===4?'current':'at_time',status,values};
          if(status===0&&isDouble)entry.f64_bits=values.map(doubleBits);
          this.frame.trace.push(entry);
        } catch(e) { this.frame.failed=true; notifyError('parameter read trace',e); }
      }
    });
  }
  if (name === 'OfxImageEffectSuite') {
    effectSuite = suite;
    attachOnce(slot(suite,3), {onEnter(a){this.frame=active();this.effect=a[0];this.name=str(a[1]);this.out=a[2];},onLeave(r){if(r.toInt32()===0 && this.frame && !this.out.isNull()){const p=this.out.readPointer();clips.set(key(p),{effect:key(this.effect),name:this.name});effectState(this.effect).clips.set(this.name,p);}}});
    attachOnce(slot(suite,5), {onEnter(a){this.frame=active();this.clip=clips.get(key(a[0]));this.out=a[3];},onLeave(r){try {const f=this.frame;if(!f || !f.selected || !this.clip || this.clip.effect!==key(f.effect) || r.toInt32()!==0)return;const h=this.out.readPointer(), m=imageMetadata(h);images.set(key(h),{frame:f,kind:this.clip.name});const kind=this.clip.name==='Source'?'source':this.clip.name==='Output'?'output':null;if(!kind)return;f[kind]={...m,data:key(m.data)};if(kind==='source')copyImage(f,kind,m);}catch(e){this.frame.failed=true;notifyError('clipGetImage',e);}}});
    attachOnce(slot(suite,6), {onEnter(a){try{const entry=images.get(key(a[0]));if(!entry)return;images.delete(key(a[0]));if(entry.kind==='Output' && entry.frame.selected){const m=imageMetadata(a[0]);copyImage(entry.frame,'output',m);}}catch(e){const f=active();if(f)f.failed=true;notifyError('clipReleaseImage',e);}}});
  }
}
function registerHost(host) {
  const fetch = slot(host,1);
  attachOnce(fetch,{onEnter(a){this.name=str(a[1]);this.version=a[2].toInt32();},onLeave(r){try{if(this.version===1)registerSuites(this.name,r);}catch(e){notifyError('fetchSuite',e);}}});
}
function registerPlugin(plugin) {
  if (plugin.isNull()) return;
  const identifier = str(plugin.add(16).readPointer());
  attachOnce(plugin.add(32).readPointer(),{onEnter(a){try{registerHost(a[0]);}catch(e){notifyError('setHost',e);}}});
  attachOnce(plugin.add(40).readPointer(),{
    onEnter(a){try{
      const action=str(a[0]), f={effect:a[1],action,selected:false,trace:[]};
      this.previous=active(); actions.set(Process.getCurrentThreadId(),f);this.frame=f;
      if(action!=='OfxImageEffectActionRender' || !armed || !property || !effectSuite || !paramSuite)return;
      const args=a[2], time=getProperty(args,'OfxPropTime','double'), window=getProperty(args,'OfxImageEffectPropRenderWindow','int',4);
      if(armed.time!==undefined && time!==armed.time)return;
      if(armed.width!==undefined && window[2]-window[0]!==armed.width)return;
      if(armed.height!==undefined && window[3]-window[1]!==armed.height)return;
      if(armed.effect && key(f.effect)!==armed.effect)return;
      f.id=++serial;f.selected=true;armed=null;
      const streamPresent=hasProperty(args,'OfxImageEffectPropCudaStream');
      f.stream=streamPresent ? getProperty(args,'OfxImageEffectPropCudaStream','pointer') : ptr(0);
      const state=effectState(f.effect);
      if(!state.create_seen || state.params.size===0 || !state.clips.has('Source') || !state.clips.has('Output')) throw new Error('instance handles not observed from creation');
      f.params=snapshotParams(state);
      const effectPropsOut=Memory.alloc(P), effectPropsFn=new NativeFunction(slot(effectSuite,0),'int',['pointer','pointer']);
      if(effectPropsFn(f.effect,effectPropsOut)!==0)throw new Error('effect property set unavailable');
      const source=state.clips.get('Source'); let rod=null;
      if(source){const out=Memory.alloc(32), fn=new NativeFunction(slot(effectSuite,7),'int',['pointer','double','pointer']);if(fn(source,time,out)===0)rod=[0,8,16,24].map(i=>out.add(i).readDouble());}
      f.actionData={time,render_window:window,render_scale:getProperty(args,'OfxImageEffectPropRenderScale','double',2),field:getProperty(args,'OfxImageEffectPropFieldToRender','string',1,false),frame_rate:getProperty(effectPropsOut.readPointer(),'OfxImageEffectPropFrameRate','double',1,false),cuda_stream_present:streamPresent,cuda_stream_is_null:f.stream.isNull()};
      if(!rod || f.actionData.frame_rate===null)throw new Error('source region or frame rate unavailable');
      f.rod=rod; send({event:'begin',capture_id:f.id,identifier,effect:key(f.effect),action:f.actionData});
    }catch(e){if(this.frame)this.frame.failed=true;notifyError('mainEntry',e);}},
    onLeave(r){const f=this.frame;try{if(f){const state=effectState(f.effect);if(f.action==='OfxActionCreateInstance' && r.toInt32()===0)state.create_seen=true;if(f.selected){const trace=[...state.history,...f.trace], latest={};for(const entry of trace)if(entry.status===0)latest[entry.name]=entry;const complete=!f.failed && f.source_copied && f.output_copied && Object.keys(latest).length>0;send({event:'capture',capture_id:f.id,schema:'juicer-ofx-capture-v1',complete,status:r.toInt32(),parameters:f.params,latest_reads:latest,read_trace:trace,action:f.actionData,source:f.source,output:f.output,source_canonical_rod:f.rod,source_file:'source.f32',output_file:'output.f32',cuda_context:f.cuda_context});}else{state.history.push(...f.trace);}if(f.action==='OfxActionDestroyInstance')effects.delete(key(f.effect));}}catch(e){notifyError('mainEntry leave',e);}finally{if(this.previous)actions.set(Process.getCurrentThreadId(),this.previous);else actions.delete(Process.getCurrentThreadId());}}
  });
  send({event:'plugin',identifier});
}
function installModule(m, prime) {
    const get=m.getExportByName('OfxGetPlugin');
    attachOnce(get,{onLeave(r){try{registerPlugin(r);}catch(e){notifyError('OfxGetPlugin',e);}}});
    if(prime){const getPlugin=new NativeFunction(get,'pointer',['int']);registerPlugin(getPlugin(0));}
    return {path:m.path,base:key(m.base)};
}
let watching = false;
rpc.exports = {
  install(modulePath) {
    const modules=Process.enumerateModules().filter(m=>m.path.toLowerCase()===modulePath.toLowerCase());
    if(modules.length!==1)throw new Error('expected one exact loaded module');
    return installModule(modules[0],true);
  },
  watch(modulePath) {
    if(watching)throw new Error('module watch already installed');
    watching=true;
    const target=modulePath.toLowerCase();
    // Install after LoadLibrary returns, outside the loader's notification
    // callback. Resolve loads plug-ins concurrently during startup; performing
    // export lookup and interception inside that callback can stall startup.
    const loader=Process.getModuleByName('kernelbase.dll');
    for(const name of ['LoadLibraryW','LoadLibraryExW','LoadLibraryA','LoadLibraryExA']) {
      const address=loader.findExportByName(name);
      if(address)attachOnce(address,{onLeave(r){
        if(r.isNull())return;
        const m=Process.findModuleByAddress(r);
        if(m && m.path.toLowerCase()===target){const result=installModule(m,true);send({event:'module',...result});}
      }});
    }
    for(const m of Process.enumerateModules()) if(m.path.toLowerCase()===target) {
      const result=installModule(m,true);send({event:'module',...result});
    }
    return true;
  },
  arm(options) { if(armed)throw new Error('already armed'); armed=options || {}; return true; },
  disarm() { armed=null; return true; },
  state() { return {property:!!property,effect_suite:!!effectSuite,param_suite:!!paramSuite,types:types.size,effects:effects.size,armed:!!armed}; }
};
