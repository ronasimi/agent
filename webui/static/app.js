const $=(s)=>document.querySelector(s);
const messagesEl=$('#messages'),promptEl=$('#prompt'),appShell=$('#appShell');
let socket=null,activeTurn=null,assistantNode=null,activeUserMessageEl=null,attachments=[],workspacePath='',workspaceRows=[];
let followOutput=true;
const artifactCards=new Map();
const UI_STATE_KEYS={left:'agent.webui.leftSidebarCollapsed',right:'agent.webui.workspaceOpen',commandHistory:'agent.webui.commandHistory.v1',conversation:'agent.webui.activeConversation.v1'};
let activeConversationId=localStorage.getItem(UI_STATE_KEYS.conversation)||'default';
const commandHistory=new CommandHistoryBuffer({storage:localStorage,key:UI_STATE_KEYS.commandHistory,limit:100});

function setPromptValue(value,{history=false}={}){
  promptEl.value=String(value??'');promptEl.style.height='auto';promptEl.style.height=Math.min(promptEl.scrollHeight,190)+'px';
  promptEl.selectionStart=promptEl.selectionEnd=promptEl.value.length;
  $('#composer').classList.toggle('history-active',history);
}
function resetHistoryNavigation(){commandHistory.reset();$('#composer').classList.remove('history-active');}
function caretOnFirstLine(){const start=promptEl.selectionStart??0,end=promptEl.selectionEnd??start;if(start!==end)return false;return !promptEl.value.slice(0,start).includes('\n');}
function caretOnLastLine(){const start=promptEl.selectionStart??0,end=promptEl.selectionEnd??start;if(start!==end)return false;return !promptEl.value.slice(end).includes('\n');}
function navigateCommandHistory(direction){
  const value=commandHistory.move(direction,promptEl.value);if(value===null)return false;
  setPromptValue(value,{history:commandHistory.isBrowsing()});return true;
}

function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function safeLink(url){try{const u=new URL(url,location.origin);return ['http:','https:'].includes(u.protocol)?u.href:'#';}catch{return '#';}}
function renderMarkdown(text){return RichOutput.renderMarkdown(text);}
function clearTurnStatus(){
  if(activeUserMessageEl)activeUserMessageEl.querySelector('.turn-status')?.remove();
}
function setStatus(text,state='ok'){
  const value=String(text||'').trim();
  if(!activeUserMessageEl)return;
  let node=activeUserMessageEl.querySelector('.turn-status');
  if(!value||value==='Ready'){node?.remove();return;}
  if(!node){node=document.createElement('div');node.className='turn-status';node.setAttribute('role','status');node.setAttribute('aria-live','polite');node.innerHTML='<span class="turn-status-dot"></span><span class="turn-status-text"></span>';activeUserMessageEl.appendChild(node);}
  node.classList.toggle('error',state==='error');node.classList.toggle('busy',state==='busy');
  node.querySelector('.turn-status-text').textContent=value;scrollBottom();
}
function isNearBottom(threshold=120){return messagesEl.scrollHeight-messagesEl.scrollTop-messagesEl.clientHeight<=threshold;}
function updateScrollFollow(){followOutput=isNearBottom();$('#scrollLatest')?.classList.toggle('hidden',followOutput);}
function scrollBottom(force=false){if(!force&&!followOutput)return;requestAnimationFrame(()=>{messagesEl.scrollTop=messagesEl.scrollHeight;followOutput=true;$('#scrollLatest')?.classList.add('hidden');});}
function addMessage(role,content,{forceScroll=false}={}){const el=document.createElement('div');el.className=`message ${role}`;el.innerHTML='<div class="bubble"></div>';const bubble=el.querySelector('.bubble');bubble.dataset.raw=content||'';if(role==='assistant')bubble.innerHTML=renderMarkdown(content||'');else bubble.textContent=content||'';messagesEl.appendChild(el);scrollBottom(forceScroll);return bubble;}
function appendAssistant(content){if(!assistantNode)assistantNode=addMessage('assistant','');assistantNode.dataset.raw=(assistantNode.dataset.raw||'')+(content||'');assistantNode.innerHTML=renderMarkdown(assistantNode.dataset.raw);scrollBottom();}
function addTool(name,status,content){const wrap=document.createElement('details');wrap.className='tool-card';const badgeClass=['ok','error','partial','history'].includes(status)?status:'history';wrap.innerHTML=`<summary>${esc(name)} <span class="badge ${badgeClass}">${esc(status)}</span></summary><pre></pre>`;wrap.querySelector('pre').textContent=content||'';messagesEl.appendChild(wrap);scrollBottom();}
function addValidator(e){const el=document.createElement('div');el.className='validator';const recipe=e.suggested_recipe?` · recipe: ${e.suggested_recipe}`:'';const label=e.validator==='grounding'?'Grounding validator':'Fast validator';const missing=Array.isArray(e.missing_fact_types)&&e.missing_fact_types.length?` · missing: ${e.missing_fact_types.join(', ')}`:'';el.textContent=`${label}: ${e.decision||'—'}${e.diagnosis?' · '+e.diagnosis:''}${missing}${e.suggested_tool?' · '+e.suggested_tool:''}${recipe}`;messagesEl.appendChild(el);scrollBottom();}
function addRecipeSuggestion(e){const el=document.createElement('div');el.className='recipe-suggestion';el.innerHTML=`<div class="recipe-suggestion-text">${esc(e.message||'Save this successful workflow as a reusable recipe?')}</div><div class="recipe-suggestion-actions"><button data-answer="yes save it">Save recipe</button><button class="secondary" data-answer="no thanks">Not now</button></div>`;el.querySelectorAll('button').forEach(btn=>btn.addEventListener('click',()=>{if(socket?.readyState!==WebSocket.OPEN)return;setPromptValue(btn.dataset.answer||'');$('#composer').requestSubmit();el.querySelectorAll('button').forEach(b=>b.disabled=true);}));messagesEl.appendChild(el);scrollBottom();}
function artifactKindFromPath(path){const ext=String(path||'').toLowerCase().split('.').pop();if(['png','jpg','jpeg','webp','gif'].includes(ext))return'image';if(ext==='pdf')return'pdf';if(['md','markdown'].includes(ext))return'markdown';if(['txt','json','yaml','yml','csv','log','py','js','ts','html','css','sh','toml','ini','xml','rst','cfg','conf'].includes(ext))return'text';if(['mp3','wav','ogg','m4a','flac'].includes(ext))return'audio';if(['mp4','webm','mov','m4v'].includes(ext))return'video';return'download';}
function artifactRelative(item){if(item?.relative)return String(item.relative);const path=String(item?.path||'');return path.startsWith('/app/workspace/')?path.slice('/app/workspace/'.length):'';}
function artifactUrl(item,download=false){const rel=artifactRelative(item);return rel?`/api/files/${encodePath(rel)}${download?'?download=true':''}`:'#';}
function artifactPreviewUrl(item){const rel=artifactRelative(item);return rel?`/api/preview/${encodePath(rel)}`:'#';}
function artifactPdfPreviewUrl(item){const rel=artifactRelative(item);return rel?`/api/pdf-preview/${encodePath(rel)}`:'#';}
async function addArtifact(item){
  const path=String(item?.path||'');if(!path.startsWith('/app/workspace/'))return null;
  if(artifactCards.has(path))return artifactCards.get(path);
  const name=String(item?.name||path.split('/').pop()||'artifact');
  const kind=String(item?.preview_kind||artifactKindFromPath(path));
  const card=document.createElement('section');card.className='artifact-card';card.dataset.artifactPath=path;
  const size=item?.size!=null?formatSize(Number(item.size)):'';
  const meta=size?`${size} · ${kind}`:kind;
  card.innerHTML=`<div class="artifact-head"><div class="artifact-file-icon">${fileIcon({type:'file',name})}</div><div class="artifact-title"><strong title="${esc(name)}">${esc(name)}</strong><span>${esc(meta)}</span></div><div class="artifact-actions"><a href="${artifactUrl(item)}" target="_blank" rel="noopener" title="Open">Open</a><a class="artifact-download" href="${artifactUrl(item,true)}" download title="Download">↓ Download</a></div></div><div class="artifact-preview"></div>`;
  const preview=card.querySelector('.artifact-preview');
  messagesEl.appendChild(card);artifactCards.set(path,card);scrollBottom();
  if(kind==='image')preview.innerHTML=`<a href="${artifactUrl(item)}" target="_blank" rel="noopener"><img src="${artifactUrl(item)}" alt="Preview of ${esc(name)}" loading="lazy"></a>`;
  else if(kind==='pdf')preview.innerHTML=`<a href="${artifactUrl(item)}" target="_blank" rel="noopener" class="artifact-pdf-preview"><img src="${artifactPdfPreviewUrl(item)}" alt="First-page preview of ${esc(name)}" loading="lazy"><span>PDF · first page preview</span></a>`;
  else if(kind==='video')preview.innerHTML=`<video controls preload="metadata" src="${artifactUrl(item)}"></video>`;
  else if(kind==='audio')preview.innerHTML=`<audio controls preload="metadata" src="${artifactUrl(item)}"></audio>`;
  else if(kind==='text'||kind==='markdown'){
    preview.innerHTML='<div class="artifact-loading">Loading preview…</div>';
    try{const data=await api(artifactPreviewUrl(item));const content=String(data.content??'');if(kind==='markdown')preview.innerHTML=`<div class="artifact-markdown">${renderMarkdown(content)}</div>`;else{const pre=document.createElement('pre');pre.textContent=content;preview.innerHTML='';preview.appendChild(pre);}if(data.truncated){const note=document.createElement('div');note.className='artifact-truncated';note.textContent='Preview truncated — download the file for the complete contents.';preview.appendChild(note);}}
    catch(e){preview.innerHTML=`<div class="artifact-loading">Preview unavailable: ${esc(e.message)}</div>`;}
  }else preview.innerHTML='<div class="artifact-generic">Preview is not available for this file type. You can open or download the file.</div>';
  scrollBottom();return card;
}
function addMedia(refs){for(const ref of(refs||[])){if(!String(ref).startsWith('/app/workspace/'))continue;void addArtifact({path:ref,name:String(ref).split('/').pop(),preview_kind:artifactKindFromPath(ref)});}scrollBottom();}
function encodePath(path){return String(path||'').split('/').filter(Boolean).map(encodeURIComponent).join('/');}
async function api(path,opts={}){const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.json();}
function cssVar(name,value){document.documentElement.style.setProperty(name,value);}
async function loadTheme(){try{const t=await api('/api/theme');const c=t.colors||{};const map={foreground:'--xr-foreground',background:'--xr-background',cursorColor:'--xr-cursor'};for(const[k,v]of Object.entries(map))if(c[k])cssVar(v,c[k]);for(let i=0;i<16;i++)if(c[`color${i}`])cssVar(`--xr-color${i}`,c[`color${i}`]);}catch(e){console.warn('Theme load failed',e);}}
async function loadHealth(){const h=await api('/api/health');$('#mainModel').textContent=h.main_model;$('#fastModel').textContent=h.fast_model;$('#contextSize').textContent=(h.context/1024).toFixed(0)+'K';}
function refreshProfileImage(){const avatar=document.querySelector('.agent-avatar'),img=$('#userProfileImage');if(!avatar||!img)return;img.onload=()=>avatar.classList.add('has-image');img.onerror=()=>avatar.classList.remove('has-image');img.src=`/api/profile-image?v=${Date.now()}`;}
function conversationQuery(){return `conversation_id=${encodeURIComponent(activeConversationId)}`;}
async function loadConversations(){
  const rows=await api('/api/conversations');const host=$('#recentConversations');host.innerHTML='';
  if(!rows.some(row=>row.id===activeConversationId)){activeConversationId=rows[0]?.id||'default';localStorage.setItem(UI_STATE_KEYS.conversation,activeConversationId);}
  for(const row of rows){const btn=document.createElement('button');btn.className='recent-item';btn.dataset.panel='chat';btn.dataset.conversationId=row.id;btn.classList.toggle('active',row.id===activeConversationId);btn.title=row.title||'Conversation';btn.innerHTML=`<span>${esc(row.title||'Conversation')}</span>`;btn.onclick=async()=>{if(activeTurn)return;activeConversationId=row.id;localStorage.setItem(UI_STATE_KEYS.conversation,activeConversationId);await Promise.all([loadHistory(),loadState()]);showPanel('chat');await loadConversations();};host.appendChild(btn);}
}
async function loadHistory(){const rows=await api(`/api/history?${conversationQuery()}`);messagesEl.innerHTML='';artifactCards.clear();activeUserMessageEl=null;for(const m of rows){if(m.role==='user'||m.role==='assistant')addMessage(m.role,m.content);else if(m.role==='tool')addTool(m.name||'tool','history',m.content);}applyChatSearch();scrollBottom(true);}
async function loadState(){$('#stateView').textContent=JSON.stringify(await api(`/api/state?${conversationQuery()}`),null,2);}
async function loadJobs(){const rows=await api('/api/jobs');$('#jobsView').innerHTML=rows.length?rows.map(j=>`<div class="data-card"><strong>${esc(j.title)}</strong><div class="muted">${esc(j.job_type)} · ${esc(j.status)} · ${esc(j.id).slice(0,8)}</div>${j.error?`<pre>${esc(j.error)}</pre>`:''}</div>`).join(''):'<div class="muted">No jobs.</div>';}
async function loadReminders(){const rows=await api('/api/reminders');$('#remindersView').innerHTML=rows.length?rows.map(r=>`<div class="data-card"><strong>${esc(r.title)}</strong><div class="muted">${esc(r.when_iso)} · ${esc(r.repeat_mode)} · ${esc(r.status)}</div><div>${esc(r.message||'')}</div></div>`).join(''):'<div class="muted">No reminders.</div>';}

function fileIcon(item){if(item.type==='directory')return '▰';const ext=(item.name.split('.').pop()||'').toLowerCase();if(['png','jpg','jpeg','webp','gif'].includes(ext))return '▧';if(ext==='pdf')return '▤';if(['md','txt','log'].includes(ext))return '▥';if(['py','js','ts','sh','yaml','yml','json','toml'].includes(ext))return '⌘';if(['zip','gz','tar','7z'].includes(ext))return '◇';return '□';}
function formatSize(n){if(n==null)return 'Folder';if(n<1024)return `${n} B`;if(n<1048576)return `${(n/1024).toFixed(n<10240?1:0)} KB`;if(n<1073741824)return `${(n/1048576).toFixed(1)} MB`;return `${(n/1073741824).toFixed(1)} GB`;}
function fileHref(item,download=false){return `/api/files/${encodePath(item.relative)}${download?'?download=true':''}`;}
function attachWorkspaceFile(item){if(item.type!=='file')return;if(!attachments.some(a=>a.path===item.path))attachments.push({name:item.name,path:item.path,media:item.media,text:item.text});renderAttachments();promptEl.focus();}
function renderWorkspace(){const q=$('#workspaceFilter').value.trim().toLowerCase();const rows=workspaceRows.filter(x=>!q||x.name.toLowerCase().includes(q));$('#workspaceList').innerHTML=rows.length?rows.map((item,i)=>`<div class="workspace-item ${item.type==='directory'?'workspace-folder':''}" data-i="${i}" data-path="${esc(item.relative)}"><div class="workspace-icon">${fileIcon(item)}</div><div><div class="workspace-name" title="${esc(item.name)}">${esc(item.name)}</div><div class="workspace-meta">${item.type==='directory'?'Folder':formatSize(item.size)}</div></div><div class="workspace-actions">${item.type==='file'?`<button class="workspace-attach" title="Attach to message">＋</button><button class="workspace-open-file" title="Open">↗</button><button class="workspace-download" title="Download">↓</button>`:'<button class="workspace-open-folder" title="Open folder">›</button>'}</div></div>`).join(''):'<div class="workspace-empty">No matching files.</div>';
  [...$('#workspaceList').querySelectorAll('.workspace-item')].forEach((el)=>{const original=workspaceRows.find(x=>x.relative===el.dataset.path);if(!original)return;el.querySelector('.workspace-name').onclick=()=>original.type==='directory'?loadWorkspace(original.relative):window.open(fileHref(original),'_blank','noopener');const a=el.querySelector('.workspace-attach');if(a)a.onclick=(e)=>{e.stopPropagation();attachWorkspaceFile(original);};const o=el.querySelector('.workspace-open-file');if(o)o.onclick=(e)=>{e.stopPropagation();window.open(fileHref(original),'_blank','noopener');};const d=el.querySelector('.workspace-download');if(d)d.onclick=(e)=>{e.stopPropagation();window.open(fileHref(original,true),'_blank','noopener');};const f=el.querySelector('.workspace-open-folder');if(f)f.onclick=(e)=>{e.stopPropagation();loadWorkspace(original.relative);};});
}
async function loadWorkspace(path=workspacePath){try{const data=await api(`/api/workspace?path=${encodeURIComponent(path||'')}`);workspacePath=data.path||'';workspaceRows=data.items||[];$('#workspacePath').textContent='/' + workspacePath;$('#workspaceBack').disabled=!workspacePath;$('#workspaceBack').dataset.parent=data.parent??'';$('#workspaceFoot').textContent=`${workspaceRows.length}${data.truncated?'+':''} item${workspaceRows.length===1?'':'s'}${data.truncated?` · showing first ${data.limit}`:''}`;renderWorkspace();}catch(e){$('#workspaceList').innerHTML=`<div class="workspace-empty">${esc(e.message)}</div>`;}}
function persistUiState(key,value){try{localStorage.setItem(key,value?'1':'0');}catch{}}
function readUiState(key,fallback=false){try{const value=localStorage.getItem(key);return value===null?fallback:value==='1';}catch{return fallback;}}
function setLeftSidebarCollapsed(collapsed,{persist=true}={}){
  const isMobile=window.matchMedia('(max-width: 760px)').matches;
  if(isMobile){
    appShell.classList.toggle('mobile-sidebar-open',!collapsed);
    $('#mobileSidebar').setAttribute('aria-expanded',String(!collapsed));
    return;
  }
  appShell.classList.toggle('sidebar-collapsed',collapsed);
  $('#collapseSidebar').setAttribute('aria-expanded',String(!collapsed));
  $('#sidebarExpand').classList.toggle('hidden',!collapsed);
  $('#sidebarExpand').setAttribute('aria-expanded',String(!collapsed));
  if(persist)persistUiState(UI_STATE_KEYS.left,collapsed);
}
function toggleLeftSidebar(forceOpen){
  const isMobile=window.matchMedia('(max-width: 760px)').matches;
  if(isMobile){
    const open=forceOpen??!appShell.classList.contains('mobile-sidebar-open');
    appShell.classList.toggle('mobile-sidebar-open',open);
    $('#mobileSidebar').setAttribute('aria-expanded',String(open));
    return;
  }
  const currentlyCollapsed=appShell.classList.contains('sidebar-collapsed');
  const collapsed=forceOpen===undefined?!currentlyCollapsed:!forceOpen;
  setLeftSidebarCollapsed(collapsed);
}
function toggleWorkspace(force,{persist=true}={}){
  const open=force??!appShell.classList.contains('workspace-open');
  appShell.classList.toggle('workspace-open',open);
  $('#workspaceToggle').classList.toggle('active',open);
  $('#workspaceToggle').setAttribute('aria-expanded',String(open));
  if(persist)persistUiState(UI_STATE_KEYS.right,open);
  if(open)loadWorkspace(workspacePath);
}
function restoreSidebarState(){
  if(window.matchMedia('(max-width: 760px)').matches){
    appShell.classList.remove('mobile-sidebar-open');
    $('#mobileSidebar').setAttribute('aria-expanded','false');
    $('#sidebarExpand').classList.add('hidden');
  }else{
    setLeftSidebarCollapsed(readUiState(UI_STATE_KEYS.left,false),{persist:false});
  }
  toggleWorkspace(readUiState(UI_STATE_KEYS.right,false),{persist:false});
}

function showPanel(name){document.querySelectorAll('.panel').forEach(x=>x.classList.add('hidden'));$(`#${name}Panel`).classList.remove('hidden');document.querySelectorAll('.nav-item,.recent-item').forEach(b=>b.classList.toggle('active',b.dataset.panel===name));$('#panelTitle').textContent={chat:'Al Agent',state:'Working state',jobs:'Jobs',reminders:'Reminders'}[name]||'Al Agent';if(name==='state')loadState();if(name==='jobs')loadJobs();if(name==='reminders')loadReminders();}
function connect(){socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/chat`);socket.onopen=()=>{};socket.onclose=()=>{setStatus('Disconnected','error');setTimeout(connect,1200)};socket.onmessage=(ev)=>{const e=JSON.parse(ev.data);if(e.type==='accepted'){activeTurn=e.turn_id;setStatus('Working…','busy');$('#send').classList.add('hidden');$('#stop').classList.remove('hidden');assistantNode=null;}else if(e.type==='queue_wait'){setStatus('Queued for inference…','busy');}else if(e.type==='queue_acquired'){setStatus('Thinking…','busy');}else if(e.type==='recipe_check'){setStatus(e.best_match?`Recipe checked: ${e.best_match}`:'Recipes checked','busy');}else if(e.type==='assistant_delta'){appendAssistant(e.content||'');}else if(e.type==='tool_start'){assistantNode=null;setStatus(`Running ${e.name}…`,'busy');}else if(e.type==='tool_result'){addTool(e.name,e.status,e.content);addMedia(e.media);if(e.name==='set_profile_image'&&e.status==='ok')refreshProfileImage();setStatus('Thinking…','busy');if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}else if(e.type==='artifact_created'){assistantNode=null;void addArtifact(e.artifact||{});setStatus(`Created ${(e.artifact&&e.artifact.name)||'file'}`,'busy');if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}else if(e.type==='validator'){assistantNode=null;addValidator(e);}else if(e.type==='recipe_suggestion'){assistantNode=null;addRecipeSuggestion(e);}else if(e.type==='requirements'){setStatus('Completing requested checks…','busy');}else if(e.type==='media_attached'){setStatus('Inspecting media…','busy');}else if(e.type==='assistant_final'){if(!assistantNode&&e.content)assistantNode=addMessage('assistant',e.content);setStatus('Ready');}else if(e.type==='turn_cancelled'){setStatus('Cancelled');finishTurn();}else if(e.type==='turn_end'){finishTurn();}else if(e.type==='history_refresh'){loadState();loadConversations();loadJobs();if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}else if(e.type==='error'){addMessage('assistant',`Error: ${e.message}`);setStatus('Error','error');finishTurn();}};}
function finishTurn(){clearTurnStatus();activeTurn=null;assistantNode=null;activeUserMessageEl=null;$('#send').classList.remove('hidden');$('#stop').classList.add('hidden');}

async function uploadChatFiles(files){
  const items=Array.from(files||[]).filter(Boolean);
  if(!items.length)return;
  for(const file of items){
    const fd=new FormData();fd.append('file',file);setStatus(`Uploading ${file.name}…`,'busy');
    try{const uploaded=await api('/api/upload',{method:'POST',body:fd});if(!attachments.some(a=>a.path===uploaded.path))attachments.push(uploaded);}
    catch(e){alert(`Could not upload ${file.name}: ${e.message}`);}
  }
  setStatus('Ready');renderAttachments();promptEl.focus();
  if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);
}
async function uploadWorkspaceFiles(files){
  const items=Array.from(files||[]).filter(Boolean);
  if(!items.length)return;
  for(const file of items){
    const fd=new FormData();fd.append('file',file);setStatus(`Adding ${file.name} to workspace…`,'busy');
    try{await api(`/api/workspace/upload?path=${encodeURIComponent(workspacePath||'')}`,{method:'POST',body:fd});}
    catch(e){alert(`Could not add ${file.name}: ${e.message}`);}
  }
  setStatus('Ready');await loadWorkspace(workspacePath);
}
function renderAttachments(){$('#attachments').innerHTML=attachments.map((a,i)=>`<span class="attachment-chip">${esc(a.name)} <button data-i="${i}" aria-label="Remove">×</button></span>`).join('');$('#attachments').querySelectorAll('button').forEach(b=>b.onclick=()=>{attachments.splice(Number(b.dataset.i),1);renderAttachments();});}

$('#composer').addEventListener('submit',(ev)=>{ev.preventDefault();const text=promptEl.value.trim();if((!text&&!attachments.length)||activeTurn)return;commandHistory.push(text);const bubble=addMessage('user',text||attachments.map(a=>a.name).join(', '),{forceScroll:true});activeUserMessageEl=bubble.closest('.message');setStatus('Sending…','busy');const turn_id=crypto.randomUUID();socket.send(JSON.stringify({type:'message',turn_id,conversation_id:activeConversationId,content:text,attachments:attachments.map(a=>a.path),thinking:$('#thinking').checked}));setPromptValue('');resetHistoryNavigation();attachments=[];renderAttachments();});
$('#stop').addEventListener('click',()=>{if(activeTurn)socket.send(JSON.stringify({type:'cancel',turn_id:activeTurn}));});
$('#composerAddFile').addEventListener('click',()=>$('#fileInput').click());
$('#fileInput').addEventListener('change',async(ev)=>{await uploadChatFiles(ev.target.files);ev.target.value='';});
$('#workspaceAddFile').addEventListener('click',()=>$('#workspaceFileInput').click());
$('#workspaceFileInput').addEventListener('change',async(ev)=>{await uploadWorkspaceFiles(ev.target.files);ev.target.value='';});

const composerEl=$('#composer');
let composerDragDepth=0;
function fileDrag(ev){return Array.from(ev.dataTransfer?.types||[]).includes('Files');}
composerEl.addEventListener('dragenter',(ev)=>{if(!fileDrag(ev))return;ev.preventDefault();composerDragDepth++;composerEl.classList.add('drag-active');});
composerEl.addEventListener('dragover',(ev)=>{if(!fileDrag(ev))return;ev.preventDefault();ev.dataTransfer.dropEffect='copy';composerEl.classList.add('drag-active');});
composerEl.addEventListener('dragleave',(ev)=>{if(!fileDrag(ev))return;ev.preventDefault();composerDragDepth=Math.max(0,composerDragDepth-1);if(!composerDragDepth)composerEl.classList.remove('drag-active');});
composerEl.addEventListener('drop',async(ev)=>{if(!fileDrag(ev))return;ev.preventDefault();composerDragDepth=0;composerEl.classList.remove('drag-active');await uploadChatFiles(ev.dataTransfer.files);});
window.addEventListener('dragover',(ev)=>{if(fileDrag(ev))ev.preventDefault();});
window.addEventListener('drop',(ev)=>{if(fileDrag(ev))ev.preventDefault();});
messagesEl.addEventListener('scroll',updateScrollFollow,{passive:true});
$('#scrollLatest').addEventListener('click',()=>scrollBottom(true));
$('#newChat').addEventListener('click',async()=>{if(activeTurn)return;const created=await api('/api/conversations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});activeConversationId=created.id;localStorage.setItem(UI_STATE_KEYS.conversation,activeConversationId);await Promise.all([loadConversations(),loadHistory(),loadState()]);showPanel('chat');promptEl.focus();});
$('#refresh').addEventListener('click',()=>Promise.all([loadHistory(),loadState(),loadJobs(),loadReminders(),loadWorkspace(workspacePath)]));
async function copyEntireChat(){
  const button=$('#copyChat');const original=button.textContent;
  try{
    const response=await fetch(`/api/history/export?${conversationQuery()}`);if(!response.ok)throw new Error(await response.text());const text=await response.text();
    if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(text);
    else{const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove();}
    button.textContent='✓';button.title='Copied chat history';
  }catch(e){console.error('Copy chat history failed',e);button.textContent='!';button.title='Copy failed';}
  finally{setTimeout(()=>{button.textContent=original;button.title='Copy entire chat history';},1200);}
}
$('#copyChat').addEventListener('click',copyEntireChat);
document.querySelectorAll('[data-panel]').forEach(btn=>btn.onclick=()=>{showPanel(btn.dataset.panel);if(window.matchMedia('(max-width: 760px)').matches)toggleLeftSidebar(false);});
$('#workspaceToggle').onclick=()=>toggleWorkspace();$('#workspaceClose').onclick=()=>toggleWorkspace(false);$('#workspaceRefresh').onclick=()=>loadWorkspace(workspacePath);$('#workspaceBack').onclick=()=>loadWorkspace($('#workspaceBack').dataset.parent||'');$('#workspaceFilter').oninput=renderWorkspace;
$('#collapseSidebar').onclick=()=>toggleLeftSidebar(false);$('#sidebarExpand').onclick=()=>toggleLeftSidebar(true);$('#mobileSidebar').onclick=()=>toggleLeftSidebar();
$('#searchChat').onclick=()=>{$('#chatSearch').classList.remove('hidden');$('#chatSearchInput').focus();};$('#chatSearchClose').onclick=()=>{$('#chatSearch').classList.add('hidden');$('#chatSearchInput').value='';applyChatSearch();};$('#chatSearchInput').oninput=applyChatSearch;
function applyChatSearch(){const q=($('#chatSearchInput')?.value||'').trim().toLowerCase();let n=0;messagesEl.querySelectorAll('.message').forEach(el=>{const hit=!q||el.textContent.toLowerCase().includes(q);el.classList.toggle('hidden',!hit);if(q&&hit)n++;});if($('#chatSearchCount'))$('#chatSearchCount').textContent=q?`${n} match${n===1?'':'es'}`:'';}
promptEl.addEventListener('input',()=>{promptEl.style.height='auto';promptEl.style.height=Math.min(promptEl.scrollHeight,190)+'px';if(commandHistory.isBrowsing()){commandHistory.reset();$('#composer').classList.remove('history-active');}});
promptEl.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#composer').requestSubmit();return;}
  if(!e.altKey&&!e.metaKey&&!e.ctrlKey&&e.key==='ArrowUp'&&caretOnFirstLine()){if(navigateCommandHistory(-1))e.preventDefault();return;}
  if(!e.altKey&&!e.metaKey&&!e.ctrlKey&&e.key==='ArrowDown'&&caretOnLastLine()){if(navigateCommandHistory(1))e.preventDefault();return;}
  if(e.ctrlKey&&!e.altKey&&!e.metaKey&&(e.key==='p'||e.key==='P')){if(navigateCommandHistory(-1))e.preventDefault();return;}
  if(e.ctrlKey&&!e.altKey&&!e.metaKey&&(e.key==='n'||e.key==='N')){if(navigateCommandHistory(1))e.preventDefault();return;}
  if(e.key==='Escape'&&commandHistory.isBrowsing()){const draft=commandHistory.restoreDraft();setPromptValue(draft??'');resetHistoryNavigation();e.preventDefault();}
});

let onboardingRequired=false,onboardingReset=false;
async function openOnboarding({reset=false}={}){
  onboardingReset=reset;const state=await api('/api/onboarding');onboardingRequired=!state.completed;const id=state.identity||{},prefs=state.preferences||{};
  $('#oobeName').value=id.name||'';$('#oobeRole').value=id.role||'';$('#oobeTimezone').value=id.timezone||Intl.DateTimeFormat().resolvedOptions().timeZone||'';$('#oobeEmail').value=id.email||'';$('#oobeInterests').value=Array.isArray(id.interests)?id.interests.join(', '):'';$('#oobeStyle').value=prefs.response?.style||'concise';$('#oobeResearch').value=prefs.search_depth?.depth||'balanced';$('#oobeError').textContent='';
  const dialog=$('#onboardingDialog');if(!dialog.open)dialog.showModal();
}
async function checkOnboarding(){try{const state=await api('/api/onboarding');if(!state.completed)await openOnboarding();}catch(e){console.error('Onboarding check failed',e);}}
$('#onboardingDialog').addEventListener('cancel',e=>{if(onboardingRequired)e.preventDefault();});
$('#profileSetup').addEventListener('click',()=>openOnboarding({reset:true}));
$('#onboardingForm').addEventListener('submit',async e=>{e.preventDefault();const save=$('#oobeSave');save.disabled=true;$('#oobeError').textContent='';try{let profile_image_path='';const photo=$('#oobePhoto').files?.[0];if(photo){const fd=new FormData();fd.append('file',photo);const uploaded=await api('/api/upload',{method:'POST',body:fd});profile_image_path=uploaded.path||'';}const payload={name:$('#oobeName').value.trim(),role:$('#oobeRole').value.trim(),timezone:$('#oobeTimezone').value.trim()||'UTC',location:$('#oobeLocation').value.trim(),email:$('#oobeEmail').value.trim(),interests:$('#oobeInterests').value.split(',').map(x=>x.trim()).filter(Boolean),response_style:$('#oobeStyle').value,research_depth:$('#oobeResearch').value,profile_image_path,reset:onboardingReset};await api('/api/onboarding',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});onboardingRequired=false;onboardingReset=false;$('#onboardingDialog').close();$('#oobePhoto').value='';refreshProfileImage();}catch(err){$('#oobeError').textContent=err.message||String(err);}finally{save.disabled=false;}});

restoreSidebarState();
document.addEventListener('keydown',(e)=>{if(e.key!=='Escape')return;if(appShell.classList.contains('workspace-open')){toggleWorkspace(false);return;}if(appShell.classList.contains('mobile-sidebar-open'))toggleLeftSidebar(false);});
window.addEventListener('resize',()=>{
  if(window.matchMedia('(max-width: 760px)').matches){
    $('#sidebarExpand').classList.add('hidden');
  }else{
    appShell.classList.remove('mobile-sidebar-open');
    setLeftSidebarCollapsed(readUiState(UI_STATE_KEYS.left,false),{persist:false});
  }
});
refreshProfileImage();
Promise.all([loadTheme(),loadHealth(),loadConversations(),loadHistory(),loadState(),loadJobs(),loadReminders(),loadWorkspace('')]).finally(()=>{connect();checkOnboarding();});
