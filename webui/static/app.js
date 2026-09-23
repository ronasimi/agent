const $=(s)=>document.querySelector(s);
const messagesEl=$('#messages'),promptEl=$('#prompt'),appShell=$('#appShell');
let socket=null,activeTurn=null,assistantNode=null,activeUserMessageEl=null,attachments=[],workspacePath='',workspaceRows=[];
let assistantStreamBuffer='',assistantRenderFrame=0,thinkingNode=null,thinkingStreamBuffer='',thinkingRenderFrame=0;
let activityGroup=null,activityCount=0;
let followOutput=true,slashCommands=[],slashMenuIndex=0;
let conversationRows=[],turnStartedAt=0,turnTimer=null,activeThinkingEnabled=false;
const artifactCards=new Map();
const UI_STATE_KEYS={left:'agent.webui.leftSidebarCollapsed',right:'agent.webui.workspaceOpen',commandHistory:'agent.webui.commandHistory.v1',activeConversation:'agent.webui.activeConversation.v1'};
let activeConversationId='';
const commandHistory=new CommandHistoryBuffer({storage:localStorage,key:UI_STATE_KEYS.commandHistory,limit:100});
const recipeDecisionStore=new InteractionState.RecipeDecisionStore({storage:localStorage,limit:120});
const recipeCards=new Map();
let pendingRecipeSubmission=null;

function setPromptValue(value,{history=false}={}){
  promptEl.value=String(value??'');promptEl.style.height='auto';promptEl.style.height=Math.min(promptEl.scrollHeight,190)+'px';
  promptEl.selectionStart=promptEl.selectionEnd=promptEl.value.length;
  $('#composer').classList.toggle('history-active',history);renderSlashMenu();
}
function resetHistoryNavigation(){commandHistory.reset();$('#composer').classList.remove('history-active');}
function caretOnFirstLine(){const start=promptEl.selectionStart??0,end=promptEl.selectionEnd??start;if(start!==end)return false;return !promptEl.value.slice(0,start).includes('\n');}
function caretOnLastLine(){const start=promptEl.selectionStart??0,end=promptEl.selectionEnd??start;if(start!==end)return false;return !promptEl.value.slice(end).includes('\n');}
function navigateCommandHistory(direction){
  const value=commandHistory.move(direction,promptEl.value);if(value===null)return false;
  setPromptValue(value,{history:commandHistory.isBrowsing()});return true;
}

function slashMenuEl(){return $('#slashMenu');}
function slashToken(value=promptEl.value){
  const raw=String(value||'').trimStart();
  if(!raw.startsWith('/')||raw.includes('\n')||/\s/.test(raw))return null;
  return raw.toLowerCase();
}
function matchingSlashCommands(){
  const token=slashToken();if(token===null)return [];
  if(token==='/')return slashCommands.slice();
  return slashCommands.filter(c=>String(c.name||'').toLowerCase().startsWith(token));
}
function closeSlashMenu(){slashMenuEl()?.classList.add('hidden');slashMenuIndex=0;promptEl.removeAttribute('aria-activedescendant');promptEl.setAttribute('aria-expanded','false');}
function renderSlashMenu(){
  const menu=slashMenuEl();if(!menu)return;const matches=matchingSlashCommands();
  if(slashToken()===null){closeSlashMenu();return;}
  if(!matches.length){menu.innerHTML='<div class="slash-empty">No matching slash commands</div>';menu.classList.remove('hidden');promptEl.setAttribute('aria-expanded','true');slashMenuIndex=0;return;}
  slashMenuIndex=Math.max(0,Math.min(slashMenuIndex,matches.length-1));
  menu.innerHTML=matches.map((c,i)=>`<button type="button" class="slash-command${i===slashMenuIndex?' active':''}" role="option" aria-selected="${i===slashMenuIndex}" data-slash-index="${i}" id="slash-command-${i}"><span class="slash-command-main"><strong>${esc(c.name)}</strong><span>${esc(c.description)}</span></span><code>${esc(c.usage||c.name)}</code></button>`).join('');
  menu.classList.remove('hidden');promptEl.setAttribute('aria-expanded','true');promptEl.setAttribute('aria-activedescendant',`slash-command-${slashMenuIndex}`);
  menu.querySelectorAll('.slash-command').forEach(btn=>{
    btn.addEventListener('mousedown',e=>e.preventDefault());
    btn.addEventListener('click',()=>selectSlashCommand(Number(btn.dataset.slashIndex||0)));
  });
  menu.querySelector('.slash-command.active')?.scrollIntoView({block:'nearest'});
}
function selectSlashCommand(index=slashMenuIndex){
  const matches=matchingSlashCommands();const item=matches[index];if(!item)return false;
  setPromptValue(item.name+(item.accepts_arguments?' ':''));closeSlashMenu();promptEl.focus();return true;
}
async function loadSlashCommands(){
  try{const rows=await api('/api/commands');slashCommands=Array.isArray(rows)?rows:[];}catch(e){slashCommands=[];console.warn('Slash command catalog load failed',e);}
}

function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function mdiIcon(name,fallback='',extraClass=''){const extra=extraClass?` ${esc(extraClass)}`:'';return `<span class="mdi mdi-${esc(name)} mdi-ui${extra}" data-fallback="${esc(fallback)}" aria-hidden="true"></span>`;}
function setButtonIcon(button,name,fallback=''){if(button)button.innerHTML=mdiIcon(name,fallback);}
function safeLink(url){try{const u=new URL(url,location.origin);return ['http:','https:'].includes(u.protocol)?u.href:'#';}catch{return '#';}}
function renderMarkdown(text){return RichOutput.renderMarkdown(text);}
function clearTurnStatus(){
  if(activeUserMessageEl)activeUserMessageEl.querySelector('.turn-status')?.remove();
}
function elapsedLabel(milliseconds){
  const total=Math.max(0,Math.floor(milliseconds/1000));const hours=Math.floor(total/3600),minutes=Math.floor((total%3600)/60),seconds=total%60;
  if(hours)return `${hours}h ${minutes}m ${seconds}s`;
  if(minutes)return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}
function setStatus(text,state='ok'){
  let value=String(text||'').trim();
  if(!activeUserMessageEl)return;
  if(state==='busy'&&turnStartedAt)value=`Working for ${elapsedLabel(Date.now()-turnStartedAt)}`;
  let node=activeUserMessageEl.querySelector('.turn-status');
  if(!value||value==='Ready'){node?.remove();return;}
  if(!node){node=document.createElement('div');node.className='turn-status';node.setAttribute('role','status');node.setAttribute('aria-live','polite');node.innerHTML='<span class="turn-status-dot"></span><span class="turn-status-text"></span>';activeUserMessageEl.appendChild(node);}
  node.classList.toggle('error',state==='error');node.classList.toggle('busy',state==='busy');
  // Status/timer updates must never force layout or scrolling. The user message
  // is already visible when the turn starts, and repeated scrollHeight reads
  // during long reasoning runs can create needless main-thread work.
  node.querySelector('.turn-status-text').textContent=value;
}
function startTurnTimer(){
  if(!turnStartedAt)turnStartedAt=Date.now();
  if(turnTimer)clearInterval(turnTimer);
  setStatus('Working','busy');
  turnTimer=setInterval(()=>setStatus('Working','busy'),1000);
}
function stopTurnTimer(){if(turnTimer)clearInterval(turnTimer);turnTimer=null;turnStartedAt=0;}
function isNearBottom(threshold=120){return messagesEl.scrollHeight-messagesEl.scrollTop-messagesEl.clientHeight<=threshold;}
function updateScrollFollow(){followOutput=isNearBottom();$('#scrollLatest')?.classList.toggle('hidden',followOutput);}
function scrollBottom(force=false){if(!force&&!followOutput)return;requestAnimationFrame(()=>{messagesEl.scrollTop=messagesEl.scrollHeight;followOutput=true;$('#scrollLatest')?.classList.add('hidden');});}
function resetActivityGroup(){activityGroup=null;activityCount=0;}
function ensureActivityGroup(){
  if(activityGroup?.isConnected)return activityGroup;
  removeEmptyChat();
  const wrap=document.createElement('details');wrap.className='turn-activity';
  wrap.innerHTML=`<summary><span class="activity-chevron">${mdiIcon('chevron-right','›')}</span><span class="activity-label">Activity</span><span class="activity-count">0</span></summary><div class="activity-body"></div>`;
  messagesEl.appendChild(wrap);activityGroup=wrap;activityCount=0;return wrap;
}
function bumpActivity(){activityCount++;const count=activityGroup?.querySelector('.activity-count');if(count)count.textContent=String(activityCount);}
function removeEmptyChat(){messagesEl.querySelector('.empty-chat')?.remove();}
function renderEmptyChat(){
  if(messagesEl.querySelector('.message,.artifact-card,.turn-activity,.recipe-suggestion'))return;
  removeEmptyChat();const empty=document.createElement('div');empty.className='empty-chat';empty.innerHTML='<img src="/static/assets/agent-logo.png" alt="" /><h1>What can I help with?</h1>';messagesEl.appendChild(empty);promptEl.placeholder='Ask anything';
}
async function copyMessage(button,bubble){
  const text=String(bubble.dataset.raw||bubble.textContent||'');
  try{await navigator.clipboard.writeText(text);setButtonIcon(button,'check','✓');button.title='Copied';}catch{setButtonIcon(button,'alert-circle-outline','!');button.title='Copy failed';}
  setTimeout(()=>{setButtonIcon(button,'content-copy','⧉');button.title='Copy message';},1000);
}
function messageContent(content){return content&&typeof content==='object'?{text:content,attachments:[]}:RichOutput.extractWorkspaceAttachments(String(content??''));}
function renderMessageMedia(bubble,items){
  const lane=bubble.closest('.message-body')?.querySelector('.message-media');if(!lane)return;
  addMedia(items,{container:lane,dedupe:true});
}
function addMessage(role,content,{forceScroll=false,media=[]}={}){
  removeEmptyChat();if(role==='user')resetActivityGroup();
  const parsed=messageContent(content);const display=parsed.text;
  const el=document.createElement('div');el.className=`message ${role}`;el.innerHTML=`<div class="message-body"><div class="message-media"></div><div class="bubble"></div><div class="message-actions"><button type="button" title="Copy message" aria-label="Copy message">${mdiIcon('content-copy','⧉')}</button></div></div>`;
  const bubble=el.querySelector('.bubble');bubble.dataset.raw=typeof content==='string'?content:JSON.stringify(content??'');if(role==='assistant')bubble.innerHTML=renderMarkdown(display);else bubble.textContent=typeof display==='string'?display:JSON.stringify(display);
  if(!display)bubble.classList.add('attachment-only');
  el.querySelector('.message-actions button').onclick=event=>copyMessage(event.currentTarget,bubble);
  messagesEl.appendChild(el);renderMessageMedia(bubble,[...parsed.attachments,...(media||[])]);promptEl.placeholder='Follow up';scrollBottom(forceScroll);return bubble;
}
function paintAssistantStream(){
  assistantRenderFrame=0;if(!assistantNode)return;
  assistantNode.dataset.raw=assistantStreamBuffer;
  assistantNode.classList.remove('attachment-only');
  // Keep streaming paints intentionally cheap. Re-rendering the full Markdown
  // document for every token is O(n^2)-ish work and can starve Chromium's
  // paint loop when WebSocket deltas arrive in a burst. The canonical Markdown
  // render happens once in finalizeAssistantStream().
  assistantNode.textContent=assistantStreamBuffer;
  assistantNode.classList.add('streaming');
  scrollBottom();
}
function scheduleAssistantPaint(){
  if(assistantRenderFrame)return;
  assistantRenderFrame=requestAnimationFrame(paintAssistantStream);
}
function ensureAssistantComposite(){
  if(!assistantNode){assistantNode=addMessage('assistant','');assistantStreamBuffer='';}
  assistantNode.classList.remove('attachment-only');
  assistantNode.classList.add('assistant-composite');
  let thinking=assistantNode.querySelector('.assistant-thinking');
  let answer=assistantNode.querySelector('.assistant-answer');
  if(!thinking||!answer){
    assistantNode.innerHTML='';
    thinking=document.createElement('details');
    thinking.className='assistant-thinking';
    thinking.open=true;
    thinking.innerHTML=`<summary><span class="thinking-chevron">${mdiIcon('chevron-right','›')}</span><span>Thinking</span><span class="thinking-live">live</span></summary><pre></pre>`;
    answer=document.createElement('div');
    answer.className='assistant-answer';
    assistantNode.appendChild(thinking);
    assistantNode.appendChild(answer);
  }
  return {thinking,answer};
}
function paintAssistantStream(){
  assistantRenderFrame=0;if(!assistantNode)return;
  const shell=ensureAssistantComposite();
  assistantNode.dataset.raw=assistantStreamBuffer;
  shell.answer.classList.remove('attachment-only');
  // Keep streaming paints intentionally cheap. Re-rendering the full Markdown
  // document for every token is O(n^2)-ish work and can starve Chromium's
  // paint loop when WebSocket deltas arrive in a burst. The canonical Markdown
  // render happens once in finalizeAssistantStream().
  shell.answer.textContent=assistantStreamBuffer;
  shell.answer.classList.add('streaming');
  scrollBottom();
}
function scheduleAssistantPaint(){
  if(assistantRenderFrame)return;
  assistantRenderFrame=requestAnimationFrame(paintAssistantStream);
}
function ensureThinkingStream(){
  const shell=ensureAssistantComposite();
  thinkingNode=shell.thinking;
  if(!thinkingStreamBuffer){thinkingNode.open=true;}
  scrollBottom();
  return thinkingNode;
}
function paintThinkingStream(){
  thinkingRenderFrame=0;if(!thinkingNode)return;
  const pre=thinkingNode.querySelector('pre');if(pre)pre.textContent=thinkingStreamBuffer;
  scrollBottom();
}
function appendThinking(content){
  ensureThinkingStream();thinkingStreamBuffer+=String(content||'');
  if(!thinkingRenderFrame)thinkingRenderFrame=requestAnimationFrame(paintThinkingStream);
}
function finalizeThinkingStream({collapse=false}={}){
  if(thinkingRenderFrame){cancelAnimationFrame(thinkingRenderFrame);thinkingRenderFrame=0;}
  if(!thinkingNode)return;
  const pre=thinkingNode.querySelector('pre');if(pre)pre.textContent=thinkingStreamBuffer;
  const live=thinkingNode.querySelector('.thinking-live');if(live)live.textContent='done';
  if(collapse)thinkingNode.open=false;
  thinkingNode=null;thinkingStreamBuffer='';
}
function resetThinkingStream(){
  if(thinkingRenderFrame){cancelAnimationFrame(thinkingRenderFrame);thinkingRenderFrame=0;}
  if(thinkingNode){
    const wrap=thinkingNode.closest('.assistant-thinking');
    if(wrap)wrap.remove();
  }
  thinkingNode=null;thinkingStreamBuffer='';
  if(assistantNode&&!assistantStreamBuffer){assistantNode.closest('.message')?.remove();assistantNode=null;}
}
function appendAssistant(content){
  finalizeThinkingStream();
  ensureAssistantComposite();
  assistantStreamBuffer+=String(content||'');
  scheduleAssistantPaint();
}
function finalizeAssistantStream(content=''){
  const canonical=String(content||'');
  if(!assistantNode){
    if(!canonical)return null;
    assistantNode=addMessage('assistant',canonical,{forceScroll:true});
    assistantStreamBuffer=canonical;
    return assistantNode;
  }
  if(assistantRenderFrame){cancelAnimationFrame(assistantRenderFrame);assistantRenderFrame=0;}
  if(canonical)assistantStreamBuffer=canonical;
  // If this assistant bubble only contains inline reasoning for a tool turn,
  // leave it intact rather than replacing it with an empty answer shell.
  if(!assistantStreamBuffer){
    assistantNode.classList.remove('streaming');
    return assistantNode;
  }
  const shell=ensureAssistantComposite();
  assistantNode.dataset.raw=assistantStreamBuffer;
  const parsed=messageContent(assistantStreamBuffer);
  shell.answer.classList.remove('streaming');
  shell.answer.innerHTML=renderMarkdown(parsed.text);
  renderMessageMedia(shell.answer,parsed.attachments);
  scrollBottom(true);
  return assistantNode;
}
function resetAssistantStream(){
  if(assistantRenderFrame){cancelAnimationFrame(assistantRenderFrame);assistantRenderFrame=0;}
  if(assistantNode)assistantNode.closest('.message')?.remove();
  assistantNode=null;assistantStreamBuffer='';
}
function addTool(name,status,content){const group=ensureActivityGroup();const wrap=document.createElement('details');wrap.className='tool-card';const badgeClass=['ok','error','partial','history'].includes(status)?status:'history';wrap.innerHTML=`<summary><span class="tool-chevron">${mdiIcon('chevron-right','›')}</span><span>${esc(name)}</span><span class="badge ${badgeClass}">${esc(status)}</span></summary><pre></pre>`;wrap.querySelector('pre').textContent=content||'';group.querySelector('.activity-body').appendChild(wrap);bumpActivity();scrollBottom();}
function addValidator(e){const group=ensureActivityGroup();const el=document.createElement('div');el.className='validator';const recipe=e.suggested_recipe?` · recipe: ${e.suggested_recipe}`:'';const label=e.validator==='grounding'?'Grounding validator':'Fast validator';const missing=Array.isArray(e.missing_fact_types)&&e.missing_fact_types.length?` · missing: ${e.missing_fact_types.join(', ')}`:'';el.textContent=`${label}: ${e.decision||'—'}${e.diagnosis?' · '+e.diagnosis:''}${missing}${e.suggested_tool?' · '+e.suggested_tool:''}${recipe}`;group.querySelector('.activity-body').appendChild(el);bumpActivity();scrollBottom();}
function recipeSuggestionId(e){
  const candidate=String(e?.candidate?.candidate_id||'').replace(/[^A-Za-z0-9_.:-]/g,'').slice(0,80);
  const eventPart=String(e?.timestamp||Date.now()).replace(/[^A-Za-z0-9_.:-]/g,'').slice(0,80);
  return `${activeConversationId}:${candidate?'candidate-'+candidate:'event-'+eventPart}`;
}
function applyRecipeDecision(el,decision){
  el.dataset.decision=decision||'';el.classList.toggle('decided-up',decision==='up');el.classList.toggle('decided-down',decision==='down');
  el.querySelectorAll('.recipe-vote').forEach(button=>{button.setAttribute('aria-pressed',String(button.dataset.decision===decision));button.disabled=Boolean(decision);});
  const status=el.querySelector('.recipe-suggestion-status');if(status)status.textContent=decision==='up'?'Recipe save selected':decision==='down'?'Recipe save declined':'Choose whether to save this recipe';
}
function submitRecipeDecision(record,decision){
  const answer=decision==='up'?'save recipe':'no thanks';
  if(activeTurn){pendingRecipeSubmission={record,decision};const card=recipeCards.get(record.id);card?.querySelector('.recipe-suggestion-status')?.replaceChildren(document.createTextNode('Decision saved · waiting for the current turn'));return false;}
  pendingRecipeSubmission=null;return submitChatMessage(answer,[],{recordHistory:false,deferUntilIdle:true});
}
function addRecipeSuggestion(e,{restore=false}={}){
  removeEmptyChat();
  const candidate=restore?e:recipeDecisionStore.upsert({id:recipeSuggestionId(e),conversationId:activeConversationId,message:e.message||'Save this successful workflow as a reusable recipe?',decision:null,createdAt:e.timestamp||new Date().toISOString()});
  if(!candidate)return null;
  const existing=recipeCards.get(candidate.id);if(existing?.isConnected){applyRecipeDecision(existing,candidate.decision);return existing;}
  const el=document.createElement('section');el.className='recipe-suggestion';el.dataset.recipeSuggestionId=candidate.id;
  el.innerHTML=`<div class="recipe-suggestion-text">${esc(candidate.message)}</div><div class="recipe-suggestion-footer"><div class="recipe-suggestion-actions" role="group" aria-label="Save this recipe?"><button type="button" class="recipe-vote recipe-vote-up" data-decision="up" aria-pressed="false" title="Save recipe">${mdiIcon('thumb-up-outline','👍')}<span class="sr-only">Save recipe</span></button><button type="button" class="recipe-vote recipe-vote-down" data-decision="down" aria-pressed="false" title="Do not save recipe">${mdiIcon('thumb-down-outline','👎')}<span class="sr-only">Do not save recipe</span></button></div><div class="recipe-suggestion-status" role="status" aria-live="polite"></div></div>`;
  el.querySelectorAll('.recipe-vote').forEach(button=>button.addEventListener('click',()=>{const decision=button.dataset.decision;if(el.dataset.decision===decision)return;const updated=recipeDecisionStore.decide(candidate.id,decision);if(!updated)return;applyRecipeDecision(el,decision);submitRecipeDecision(updated,decision);}));
  applyRecipeDecision(el,candidate.decision);messagesEl.appendChild(el);recipeCards.set(candidate.id,el);scrollBottom();return el;
}
function restoreRecipeSuggestions(conversationId){
  const replies=Array.from(messagesEl.querySelectorAll('.message.user'));let replyIndex=0;
  for(const item of recipeDecisionStore.list(conversationId)){
    const card=addRecipeSuggestion(item,{restore:true});if(!card||!item.decision)continue;
    const answer=item.decision==='up'?'save recipe':'no thanks';
    for(;replyIndex<replies.length;replyIndex++){const row=replies[replyIndex];if(String(row.querySelector('.bubble')?.dataset.raw||'').trim().toLowerCase()!==answer)continue;messagesEl.insertBefore(card,row);replyIndex++;break;}
  }
}
function artifactKindFromPath(path){const ext=String(path||'').toLowerCase().split('.').pop();if(['png','jpg','jpeg','webp','gif'].includes(ext))return'image';if(ext==='pdf')return'pdf';if(['md','markdown'].includes(ext))return'markdown';if(['txt','json','yaml','yml','csv','log','py','js','ts','html','css','sh','toml','ini','xml','rst','cfg','conf'].includes(ext))return'text';if(['mp3','wav','ogg','m4a','flac'].includes(ext))return'audio';if(['mp4','webm','mov','m4v'].includes(ext))return'video';if(['docx','xlsx','pptx','odt','ods','odp','rtf'].includes(ext))return'document';return'download';}
function artifactRelative(item){if(item?.relative)return String(item.relative);const path=String(item?.path||'');return path.startsWith('/app/workspace/')?path.slice('/app/workspace/'.length):'';}
function artifactUrl(item,download=false){const rel=artifactRelative(item);return rel?`/api/files/${encodePath(rel)}${download?'?download=true':''}`:'#';}
function artifactPreviewUrl(item){const rel=artifactRelative(item);return rel?`/api/preview/${encodePath(rel)}`:'#';}
function artifactPdfPreviewUrl(item){const rel=artifactRelative(item);return rel?`/api/pdf-preview/${encodePath(rel)}`:'#';}
async function addArtifact(item,{container=messagesEl,dedupe=true}={}){
  const path=String(item?.path||'');if(!path.startsWith('/app/workspace/'))return null;
  if(dedupe&&artifactCards.has(path))return artifactCards.get(path);
  const localMatch=Array.from(container.querySelectorAll?.('[data-artifact-path]')||[]).find(node=>node.dataset.artifactPath===path);if(localMatch)return localMatch;
  removeEmptyChat();
  const name=String(item?.name||path.split('/').pop()||'artifact');
  const kind=String(item?.preview_kind||artifactKindFromPath(path));
  const card=document.createElement('section');card.className=`artifact-card${container!==messagesEl?' inline-artifact':''}`;card.dataset.artifactPath=path;
  const size=item?.size!=null?formatSize(Number(item.size)):'';
  const meta=size?`${size} · ${kind}`:kind;
  card.innerHTML=`<div class="artifact-head"><div class="artifact-file-icon">${fileIcon({type:'file',name})}</div><div class="artifact-title"><strong title="${esc(name)}">${esc(name)}</strong><span>${esc(meta)}</span></div><div class="artifact-actions"><a href="${artifactUrl(item)}" target="_blank" rel="noopener" title="Open">${mdiIcon('open-in-new','↗')}<span>Open</span></a><a class="artifact-download" href="${artifactUrl(item,true)}" download title="Download">${mdiIcon('download','↓')}<span>Download</span></a></div></div><div class="artifact-preview"></div>`;
  const preview=card.querySelector('.artifact-preview');
  container.appendChild(card);if(dedupe&&container===messagesEl)artifactCards.set(path,card);scrollBottom();
  if(kind==='image')preview.innerHTML=`<a href="${artifactUrl(item)}" target="_blank" rel="noopener"><img src="${artifactUrl(item)}" alt="Preview of ${esc(name)}" loading="lazy"></a>`;
  else if(kind==='pdf')preview.innerHTML=`<a href="${artifactUrl(item)}" target="_blank" rel="noopener" class="artifact-pdf-preview"><img src="${artifactPdfPreviewUrl(item)}" alt="First-page preview of ${esc(name)}" loading="lazy"><span>PDF · first page preview</span></a>`;
  else if(kind==='video')preview.innerHTML=`<video controls preload="metadata" src="${artifactUrl(item)}"></video>`;
  else if(kind==='audio')preview.innerHTML=`<audio controls preload="metadata" src="${artifactUrl(item)}"></audio>`;
  else if(kind==='text'||kind==='markdown'||kind==='document'){
    preview.innerHTML='<div class="artifact-loading">Loading preview…</div>';
    try{const data=await api(artifactPreviewUrl(item));const content=String(data.content??'');if(kind==='markdown')preview.innerHTML=`<div class="artifact-markdown">${renderMarkdown(content)}</div>`;else if(content){const pre=document.createElement('pre');pre.textContent=content;preview.innerHTML='';preview.appendChild(pre);}else preview.innerHTML='<div class="artifact-generic">This document has no extractable text preview. The original remains available above.</div>';if(data.truncated){const note=document.createElement('div');note.className='artifact-truncated';note.textContent='Preview truncated — download the file for the complete contents.';preview.appendChild(note);}}
    catch(e){preview.innerHTML=`<div class="artifact-loading">Preview unavailable: ${esc(e.message)}</div>`;}
  }else preview.innerHTML='<div class="artifact-generic">Preview is not available for this file type. You can open or download the file.</div>';
  scrollBottom();return card;
}
function addProfileMedia({container=messagesEl,dedupe=true}={}){
  const key='profile-image://current';const url=`/api/profile-image?v=${Date.now()}`;
  if(dedupe&&artifactCards.has(key)){const existing=artifactCards.get(key);for(const link of existing.querySelectorAll('a'))link.href=url;const img=existing.querySelector('img');if(img)img.src=url;return existing;}
  removeEmptyChat();
  const card=document.createElement('section');card.className=`artifact-card${container!==messagesEl?' inline-artifact':''}`;card.dataset.artifactPath=key;
  card.innerHTML=`<div class="artifact-head"><div class="artifact-file-icon">${mdiIcon('image-outline','▧')}</div><div class="artifact-title"><strong>Current profile picture</strong><span>image</span></div><div class="artifact-actions"><a href="${url}" target="_blank" rel="noopener" title="Open">${mdiIcon('open-in-new','↗')}<span>Open</span></a><a class="artifact-download" href="${url}" download="profile-picture.png" title="Download">${mdiIcon('download','↓')}<span>Download</span></a></div></div><div class="artifact-preview"><a href="${url}" target="_blank" rel="noopener"><img src="${url}" alt="Current profile picture" loading="lazy"></a></div>`;
  container.appendChild(card);if(dedupe&&container===messagesEl)artifactCards.set(key,card);scrollBottom();return card;
}
function addMedia(refs,{container=messagesEl,dedupe=true}={}){for(const ref of(refs||[])){const item=ref&&typeof ref==='object'?ref:{path:String(ref||'')};const value=String(item.path||item.reference||'');if(value==='profile-image://current'){addProfileMedia({container,dedupe});continue;}if(!value.startsWith('/app/workspace/'))continue;void addArtifact({...item,path:value,name:item.name||value.split('/').pop(),preview_kind:item.preview_kind||artifactKindFromPath(value)},{container,dedupe});}scrollBottom();}
function encodePath(path){return String(path||'').split('/').filter(Boolean).map(encodeURIComponent).join('/');}
async function api(path,opts={}){const r=await fetch(path,opts);if(!r.ok){const raw=await r.text();let message=raw||`Request failed (${r.status})`;try{const parsed=JSON.parse(raw);message=parsed?.detail?.message||parsed?.detail||message;}catch{}throw new Error(String(message));}return r.json();}
function cssVar(name,value){document.documentElement.style.setProperty(name,value);}
async function loadTheme(){try{const t=await api('/api/theme');const c=t.colors||{};const map={foreground:'--xr-foreground',background:'--xr-background',cursorColor:'--xr-cursor'};for(const[k,v]of Object.entries(map))if(c[k])cssVar(v,c[k]);for(let i=0;i<16;i++)if(c[`color${i}`])cssVar(`--xr-color${i}`,c[`color${i}`]);}catch(e){console.warn('Theme load failed',e);}}
async function loadHealth(){
  const main=$('#mainModel'),fast=$('#fastModel'),vision=$('#visionModel'),context=$('#contextSize');
  try{
    const h=await api('/api/health');
    const mainName=String(h.main_model||'Unknown model'),fastName=String(h.fast_model||'Unknown fast model'),visionName=String(h.vision_model||mainName);
    const contextTokens=Number(h.context);
    main.textContent=mainName;main.title=`Main model: ${mainName}`;
    fast.textContent=fastName;fast.title=`Fast model: ${fastName}`;
    vision.textContent=`Vision: ${visionName}`;vision.title=`Vision model: ${visionName}`;
    context.textContent=Number.isFinite(contextTokens)&&contextTokens>0?`${Math.round(contextTokens/1024)}K`:'—';
    context.title=Number.isFinite(contextTokens)&&contextTokens>0?`${contextTokens.toLocaleString()} token context`:'Context size unavailable';
    return h;
  }catch(e){
    main.textContent='Model status unavailable';main.title='Unable to load /api/health';
    fast.textContent='—';fast.title='';vision.textContent='—';vision.title='';context.textContent='—';context.title='';
    console.warn('Health load failed',e);return null;
  }
}
function refreshProfileImage(){const avatar=document.querySelector('.agent-avatar'),img=$('#userProfileImage');if(!avatar||!img)return;img.onload=()=>avatar.classList.add('has-image');img.onerror=()=>avatar.classList.remove('has-image');img.src=`/api/profile-image?v=${Date.now()}`;}
function conversationQuery(conversationId=activeConversationId){return `conversation_id=${encodeURIComponent(conversationId)}`;}
function displayConversationTitle(row){const title=String(row?.title||'').trim();return !title||title==='New conversation'||title==='Current conversation'?'New chat':title;}
function readActiveConversation(){try{return String(localStorage.getItem(UI_STATE_KEYS.activeConversation)||'');}catch{return '';}}
function setActiveConversation(conversationId,{persist=true}={}){
  activeConversationId=String(conversationId||'');
  if(persist){try{if(activeConversationId)localStorage.setItem(UI_STATE_KEYS.activeConversation,activeConversationId);else localStorage.removeItem(UI_STATE_KEYS.activeConversation);}catch{}}
  updateConversationHeading();
}
function currentConversation(){return conversationRows.find(row=>row.id===activeConversationId)||null;}
function updateConversationHeading(){
  const title=displayConversationTitle(currentConversation());const panel=$('#chatPanel');
  if(panel&&!panel.classList.contains('hidden'))$('#panelTitle').textContent=title;
  document.title=`${title} · Al Agent`;
}
async function createFreshConversation(){
  const created=await api('/api/conversations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
  setActiveConversation(created.id);
  return created;
}
async function activateConversation(conversationId){
  const target=String(conversationId||'');
  if(!target)return false;
  // Panel navigation must stay responsive while the current turn streams.
  // Re-opening the already-active conversation is presentation-only and does
  // not change the stream target, so it is safe during reasoning/generation.
  if(target===activeConversationId){
    showPanel('chat');
    promptEl.focus();
    return true;
  }
  // Switching the stream to a different conversation mid-turn would make the
  // global assistant/thinking nodes receive deltas for the wrong transcript.
  if(activeTurn)return false;
  setActiveConversation(target);
  // Load the clicked thread explicitly. Both loaders guard against a later
  // selection winning the race, so rapid sidebar clicks cannot render stale data.
  await Promise.all([loadHistory(target),loadState(target)]);
  if(activeConversationId!==target)return false;
  showPanel('chat');
  await loadConversations();
  promptEl.focus();
  return true;
}
async function renameSavedConversation(row){
  if(activeTurn)return;
  const current=displayConversationTitle(row);const title=window.prompt('Rename conversation',current);
  if(title===null||!title.trim()||title.trim()===current)return;
  await api(`/api/conversations/${encodeURIComponent(row.id)}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title.trim()})});
  await loadConversations();
}
async function deleteSavedConversation(row){
  if(activeTurn)return;
  const title=String(row?.title||'Conversation');
  if(!window.confirm(`Delete "${title}"? This permanently removes its saved messages and state.`))return;
  const deletingActive=row.id===activeConversationId;
  const result=await api(`/api/conversations/${encodeURIComponent(row.id)}`,{method:'DELETE'});
  if(!result?.ok)throw new Error('Conversation could not be deleted.');
  recipeDecisionStore.removeConversation(row.id);
  if(deletingActive){
    await createFreshConversation();
    await Promise.all([loadHistory(),loadState()]);
    showPanel('chat');
  }
  await loadConversations();
  promptEl.focus();
}
function closeConversationMenus(except=null){document.querySelectorAll('.conversation-menu.open').forEach(menu=>{if(menu!==except)menu.classList.remove('open');});}
function renderConversationList(){
  const host=$('#recentConversations'),query=String($('#conversationSearchInput')?.value||'').trim().toLowerCase();host.innerHTML='';
  const rows=conversationRows.filter(row=>!query||displayConversationTitle(row).toLowerCase().includes(query));
  if(!rows.length){host.innerHTML=`<div class="recent-empty">${query?'No matching chats':'No conversations yet'}</div>`;return;}
  for(const row of rows){
    const wrap=document.createElement('div');wrap.className='recent-conversation';wrap.dataset.conversationId=row.id;
    const title=displayConversationTitle(row);
    const btn=document.createElement('button');btn.type='button';btn.className='recent-item';btn.dataset.panel='chat';btn.dataset.conversationId=row.id;btn.classList.toggle('active',row.id===activeConversationId);btn.title=title;btn.innerHTML=`<span>${esc(title)}</span>`;btn.onclick=()=>activateConversation(row.id);
    const more=document.createElement('button');more.type='button';more.className='recent-more';more.title='Conversation options';more.setAttribute('aria-label',`Options for ${title}`);more.innerHTML=mdiIcon('dots-horizontal','⋯');
    const menu=document.createElement('div');menu.className='conversation-menu';menu.innerHTML=`<button type="button" class="conversation-rename">${mdiIcon('pencil-outline','✎')}<span>Rename</span></button><button type="button" class="recent-delete">${mdiIcon('delete-outline','×')}<span>Delete</span></button>`;
    more.onclick=e=>{e.preventDefault();e.stopPropagation();const opening=!menu.classList.contains('open');closeConversationMenus();menu.classList.toggle('open',opening);};
    menu.querySelector('.conversation-rename').onclick=async e=>{e.stopPropagation();closeConversationMenus();try{await renameSavedConversation(row);}catch(err){console.error(err);window.alert(err.message||String(err));}};
    menu.querySelector('.recent-delete').onclick=async e=>{e.stopPropagation();closeConversationMenus();try{await deleteSavedConversation(row);}catch(err){console.error(err);window.alert(err.message||String(err));}};
    wrap.append(btn,more,menu);host.appendChild(wrap);
  }
}
async function loadConversations(){
  const params=new URLSearchParams({limit:'200'});if(activeConversationId)params.set('active_conversation_id',activeConversationId);
  const rows=await api(`/api/conversations?${params}`);
  conversationRows=(Array.isArray(rows)?rows:[]).slice().sort((a,b)=>{
    if(a.id===activeConversationId)return -1;if(b.id===activeConversationId)return 1;
    const messageDelta=Number(b.last_message_id||0)-Number(a.last_message_id||0);if(messageDelta)return messageDelta;
    return String(b.updated_at||'').localeCompare(String(a.updated_at||''));
  });
  renderConversationList();updateConversationHeading();return conversationRows;
}
async function loadHistory(conversationId=activeConversationId){const cid=String(conversationId||'');const rows=await api(`/api/history?${conversationQuery(cid)}`);if(cid!==activeConversationId)return false;messagesEl.innerHTML='';artifactCards.clear();recipeCards.clear();activeUserMessageEl=null;assistantNode=null;assistantStreamBuffer='';thinkingNode=null;thinkingStreamBuffer='';if(assistantRenderFrame){cancelAnimationFrame(assistantRenderFrame);assistantRenderFrame=0;}if(thinkingRenderFrame){cancelAnimationFrame(thinkingRenderFrame);thinkingRenderFrame=0;}resetActivityGroup();for(const m of rows){if(m.role==='user'||m.role==='assistant')addMessage(m.role,m.content,{media:m.media||[]});else if(m.role==='tool'){addTool(m.name||'tool','history',m.content);addMedia(m.media||[]);}}restoreRecipeSuggestions(cid);renderEmptyChat();scrollBottom(true);return true;}
async function loadState(conversationId=activeConversationId){const cid=String(conversationId||'');const state=await api(`/api/state?${conversationQuery(cid)}`);if(cid!==activeConversationId)return false;$('#stateView').textContent=JSON.stringify(state,null,2);return true;}
async function loadJobs(){const rows=await api('/api/jobs');$('#jobsView').innerHTML=rows.length?rows.map(j=>`<div class="data-card"><strong>${esc(j.title)}</strong><div class="muted">${esc(j.job_type)} · ${esc(j.status)} · ${esc(j.id).slice(0,8)}</div>${j.error?`<pre>${esc(j.error)}</pre>`:''}</div>`).join(''):'<div class="muted">No jobs.</div>';}
async function loadReminders(){const rows=await api('/api/reminders');$('#remindersView').innerHTML=rows.length?rows.map(r=>`<div class="data-card"><strong>${esc(r.title)}</strong><div class="muted">${esc(r.when_iso)} · ${esc(r.repeat_mode)} · ${esc(r.status)}</div><div>${esc(r.message||'')}</div></div>`).join(''):'<div class="muted">No reminders.</div>';}

function fileIcon(item){if(item.type==='directory')return mdiIcon('folder','▰');const ext=(item.name.split('.').pop()||'').toLowerCase();if(['png','jpg','jpeg','webp','gif'].includes(ext))return mdiIcon('file-image-outline','▧');if(ext==='pdf')return mdiIcon('file-pdf-box','▤');if(['md','txt','log'].includes(ext))return mdiIcon('file-document-outline','▥');if(['py','js','ts','sh','yaml','yml','json','toml'].includes(ext))return mdiIcon('file-code-outline','⌘');if(['zip','gz','tar','7z'].includes(ext))return mdiIcon('folder-zip-outline','◇');return mdiIcon('file-outline','□');}
function formatSize(n){if(n==null)return 'Folder';if(n<1024)return `${n} B`;if(n<1048576)return `${(n/1024).toFixed(n<10240?1:0)} KB`;if(n<1073741824)return `${(n/1048576).toFixed(1)} MB`;return `${(n/1073741824).toFixed(1)} GB`;}
function fileHref(item,download=false){return `/api/files/${encodePath(item.relative)}${download?'?download=true':''}`;}
function attachWorkspaceFile(item){if(item.type!=='file')return;if(!attachments.some(a=>a.path===item.path))attachments.push({name:item.name,path:item.path,media:item.media,text:item.text});renderAttachments();promptEl.focus();}
function renderWorkspace(){const q=$('#workspaceFilter').value.trim().toLowerCase();const rows=workspaceRows.filter(x=>!q||x.name.toLowerCase().includes(q));$('#workspaceList').innerHTML=rows.length?rows.map((item,i)=>`<div class="workspace-item ${item.type==='directory'?'workspace-folder':''}" data-i="${i}" data-path="${esc(item.relative)}"><div class="workspace-icon">${fileIcon(item)}</div><div><div class="workspace-name" title="${esc(item.name)}">${esc(item.name)}</div><div class="workspace-meta">${item.type==='directory'?'Folder':formatSize(item.size)}</div></div><div class="workspace-actions">${item.type==='file'?`<button class="workspace-attach" title="Attach to message" aria-label="Attach to message">${mdiIcon('paperclip','＋')}</button><button class="workspace-open-file" title="Open" aria-label="Open file">${mdiIcon('open-in-new','↗')}</button><button class="workspace-download" title="Download" aria-label="Download file">${mdiIcon('download','↓')}</button>`:`<button class="workspace-open-folder" title="Open folder" aria-label="Open folder">${mdiIcon('chevron-right','›')}</button>`}</div></div>`).join(''):'<div class="workspace-empty">No matching files.</div>';
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

function showPanel(name){document.querySelectorAll('.panel').forEach(x=>x.classList.add('hidden'));$(`#${name}Panel`).classList.remove('hidden');document.querySelectorAll('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.panel===name));document.querySelectorAll('.recent-item').forEach(b=>b.classList.toggle('active',name==='chat'&&b.dataset.conversationId===activeConversationId));$('#panelTitle').textContent={chat:displayConversationTitle(currentConversation()),state:'Working state',jobs:'Jobs',reminders:'Reminders'}[name]||'Al Agent';if(name==='chat')updateConversationHeading();if(name==='state')loadState();if(name==='jobs')loadJobs();if(name==='reminders')loadReminders();}
function connect(){
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/chat`);
  socket.onopen=()=>{};
  socket.onclose=()=>{setStatus('Disconnected','error');setTimeout(connect,1200);};
  socket.onmessage=ev=>{
    const e=JSON.parse(ev.data);
    if(e.type==='accepted'){
      activeTurn=e.turn_id;startTurnTimer();$('#send').classList.add('hidden');$('#stop').classList.remove('hidden');assistantNode=null;assistantStreamBuffer='';thinkingNode=null;thinkingStreamBuffer='';
    }else if(e.type==='command_result'){
      assistantNode=null;assistantStreamBuffer='';resetThinkingStream();
      if(e.action==='clear_conversation'){messagesEl.innerHTML='';artifactCards.clear();recipeCards.clear();recipeDecisionStore.removeConversation(activeConversationId);activeUserMessageEl=null;followOutput=true;renderEmptyChat();}
      if(e.action==='set_thinking')$('#thinking').checked=!!e.data?.thinking;
      if(e.action==='open_profile')void openOnboarding({reset:true});
      if(['jobs_changed','show_jobs'].includes(e.action))void loadJobs();
      if(e.action==='show_reminders')void loadReminders();
      if(e.message)addMessage('assistant',e.message,{forceScroll:true});
      if(e.ok===false)setStatus('Command failed','error');else setStatus('Ready');
      finishTurn();
    }else if(e.type==='queue_wait')setStatus('Queued for inference…','busy');
    else if(e.type==='queue_acquired')setStatus(activeThinkingEnabled?'Thinking…':'Generating…','busy');
    else if(e.type==='recipe_check')setStatus(e.best_match?`Recipe checked: ${e.best_match}`:'Recipes checked','busy');
    else if(e.type==='thinking_delta')appendThinking(e.content||'');
    else if(e.type==='assistant_delta')appendAssistant(e.content||'');
    else if(e.type==='assistant_reset'){resetAssistantStream();resetThinkingStream();}
    else if(e.type==='tool_start'){finalizeThinkingStream();finalizeAssistantStream();assistantNode=null;assistantStreamBuffer='';setStatus(`Running ${e.name}…`,'busy');}
    else if(e.type==='tool_result'){addTool(e.name,e.status,e.content);addMedia(e.media);if(e.name==='set_profile_image'&&e.status==='ok')refreshProfileImage();setStatus(activeThinkingEnabled?'Thinking…':'Generating…','busy');if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}
    else if(e.type==='artifact_created'){finalizeThinkingStream();finalizeAssistantStream();assistantNode=null;assistantStreamBuffer='';void addArtifact(e.artifact||{});setStatus(`Created ${(e.artifact&&e.artifact.name)||'file'}`,'busy');if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}
    else if(e.type==='validator'){finalizeThinkingStream();finalizeAssistantStream();assistantNode=null;assistantStreamBuffer='';addValidator(e);}
    else if(e.type==='recipe_suggestion'){finalizeThinkingStream();finalizeAssistantStream();assistantNode=null;assistantStreamBuffer='';addRecipeSuggestion(e);}
    else if(e.type==='requirements')setStatus('Completing requested checks…','busy');
    else if(e.type==='media_attached')setStatus('Inspecting media…','busy');
    else if(e.type==='assistant_final'){finalizeThinkingStream();finalizeAssistantStream(e.content||'');setStatus('Ready');}
    else if(e.type==='turn_cancelled'){setStatus('Cancelled');finishTurn();}
    else if(e.type==='turn_end')finishTurn();
    else if(e.type==='history_refresh'){loadState();loadConversations();loadJobs();if(appShell.classList.contains('workspace-open'))loadWorkspace(workspacePath);}
    else if(e.type==='error'){addMessage('assistant',`Error: ${e.message}`);setStatus('Error','error');finishTurn();}
  };
}
function finishTurn(){
  finalizeThinkingStream();
  finalizeAssistantStream();
  stopTurnTimer();clearTurnStatus();activeTurn=null;activeThinkingEnabled=false;assistantNode=null;assistantStreamBuffer='';thinkingNode=null;thinkingStreamBuffer='';activeUserMessageEl=null;$('#send').classList.remove('hidden');$('#stop').classList.add('hidden');
  if(pendingRecipeSubmission){const queued=pendingRecipeSubmission;pendingRecipeSubmission=null;queueMicrotask(()=>submitRecipeDecision(queued.record,queued.decision));}
}

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
function renderAttachments(){$('#attachments').innerHTML=attachments.map((a,i)=>`<span class="attachment-chip">${esc(a.name)} <button data-i="${i}" aria-label="Remove attachment">${mdiIcon('close','×')}</button></span>`).join('');$('#attachments').querySelectorAll('button').forEach(b=>b.onclick=()=>{attachments.splice(Number(b.dataset.i),1);renderAttachments();});}

function submitChatMessage(text,mediaItems=[],{recordHistory=true,clearComposer=true,deferUntilIdle=false}={}){
  const content=String(text||'').trim();const items=Array.from(mediaItems||[]).filter(item=>item&&item.path);
  if((!content&&!items.length)||activeTurn||socket?.readyState!==WebSocket.OPEN)return false;
  closeSlashMenu();if(recordHistory&&content)commandHistory.push(content);
  const bubble=addMessage('user',content,{forceScroll:true,media:items});activeUserMessageEl=bubble.closest('.message');setStatus('Sending…','busy');
  const thinkingEnabled=Boolean($('#thinking').checked);activeThinkingEnabled=thinkingEnabled;
  const turn_id=crypto.randomUUID();socket.send(JSON.stringify({type:'message',turn_id,conversation_id:activeConversationId,content,attachments:items.map(item=>item.path),thinking:thinkingEnabled,defer_until_idle:Boolean(deferUntilIdle)}));
  if(clearComposer){setPromptValue('');resetHistoryNavigation();attachments=[];renderAttachments();}return true;
}
$('#composer').addEventListener('submit',(ev)=>{ev.preventDefault();submitChatMessage(promptEl.value,attachments.slice());});
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
$('#newChat').addEventListener('click',async()=>{if(activeTurn)return;await createFreshConversation();await Promise.all([loadConversations(),loadHistory(),loadState()]);showPanel('chat');promptEl.focus();});
$('#refresh').addEventListener('click',()=>Promise.all([loadHistory(),loadState(),loadJobs(),loadReminders(),loadWorkspace(workspacePath)]));
async function copyEntireChat(){
  const button=$('#copyChat');const original=button.innerHTML;
  try{
    const response=await fetch(`/api/history/export?${conversationQuery()}`);if(!response.ok)throw new Error(await response.text());const text=await response.text();
    if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(text);
    else{const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove();}
    setButtonIcon(button,'check','✓');button.title='Copied chat history';
  }catch(e){console.error('Copy chat history failed',e);setButtonIcon(button,'alert-circle-outline','!');button.title='Copy failed';}
  finally{setTimeout(()=>{button.innerHTML=original;button.title='Copy entire chat history';},1200);}
}
$('#copyChat').addEventListener('click',copyEntireChat);
document.querySelectorAll('[data-panel]').forEach(btn=>btn.onclick=()=>{showPanel(btn.dataset.panel);if(window.matchMedia('(max-width: 760px)').matches)toggleLeftSidebar(false);});
$('#workspaceToggle').onclick=()=>toggleWorkspace();$('#workspaceClose').onclick=()=>toggleWorkspace(false);$('#workspaceRefresh').onclick=()=>loadWorkspace(workspacePath);$('#workspaceBack').onclick=()=>loadWorkspace($('#workspaceBack').dataset.parent||'');$('#workspaceFilter').oninput=renderWorkspace;
$('#collapseSidebar').onclick=()=>toggleLeftSidebar(false);$('#sidebarExpand').onclick=()=>toggleLeftSidebar(true);$('#mobileSidebar').onclick=()=>toggleLeftSidebar();
$('#searchChat').onclick=()=>{$('#conversationSearch').classList.remove('hidden');$('#conversationSearchInput').focus();};
$('#conversationSearchClose').onclick=()=>{$('#conversationSearch').classList.add('hidden');$('#conversationSearchInput').value='';renderConversationList();};
$('#conversationSearchInput').oninput=renderConversationList;
promptEl.addEventListener('input',()=>{promptEl.style.height='auto';promptEl.style.height=Math.min(promptEl.scrollHeight,190)+'px';if(commandHistory.isBrowsing()){commandHistory.reset();$('#composer').classList.remove('history-active');}slashMenuIndex=0;renderSlashMenu();});
promptEl.addEventListener('keydown',e=>{
  const slashOpen=!slashMenuEl()?.classList.contains('hidden');
  if(slashOpen&&!e.altKey&&!e.metaKey&&!e.ctrlKey&&(e.key==='ArrowDown'||e.key==='ArrowUp')){const matches=matchingSlashCommands();if(matches.length){e.preventDefault();slashMenuIndex=(slashMenuIndex+(e.key==='ArrowDown'?1:-1)+matches.length)%matches.length;renderSlashMenu();}return;}
  if(slashOpen&&(e.key==='Tab'||(e.key==='Enter'&&!e.shiftKey))){const matches=matchingSlashCommands(),item=matches[slashMenuIndex];if(item){e.preventDefault();const exact=promptEl.value.trim()===item.name;if(e.key==='Enter'&&exact&&!item.accepts_arguments){closeSlashMenu();$('#composer').requestSubmit();}else selectSlashCommand(slashMenuIndex);return;}}
  if(slashOpen&&e.key==='Escape'){closeSlashMenu();e.preventDefault();return;}
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#composer').requestSubmit();return;}
  if(!e.altKey&&!e.metaKey&&!e.ctrlKey&&e.key==='ArrowUp'&&caretOnFirstLine()){if(navigateCommandHistory(-1))e.preventDefault();return;}
  if(!e.altKey&&!e.metaKey&&!e.ctrlKey&&e.key==='ArrowDown'&&caretOnLastLine()){if(navigateCommandHistory(1))e.preventDefault();return;}
  if(e.ctrlKey&&!e.altKey&&!e.metaKey&&(e.key==='p'||e.key==='P')){if(navigateCommandHistory(-1))e.preventDefault();return;}
  if(e.ctrlKey&&!e.altKey&&!e.metaKey&&(e.key==='n'||e.key==='N')){if(navigateCommandHistory(1))e.preventDefault();return;}
  if(e.key==='Escape'&&commandHistory.isBrowsing()){const draft=commandHistory.restoreDraft();setPromptValue(draft??'');resetHistoryNavigation();e.preventDefault();}
});
document.addEventListener('pointerdown',e=>{if(!$('#composer').contains(e.target)&&!slashMenuEl()?.contains(e.target))closeSlashMenu();if(!e.target.closest('.recent-conversation'))closeConversationMenus();});

let onboardingRequired=false,onboardingReset=false,oobeStep='profile',googleConnection=null;
const OOBE_COPY={
  profile:['Set up Al Agent','Personalize your local assistant. These details stay in the local harness.'],
  google:['Connect your tools','Grant only the read access you want Al Agent to use.'],
  finish:['Setup complete','Everything can be changed later from the sidebar.'],
};
function clearGoogleOAuthQuery(){const url=new URL(location.href);url.searchParams.delete('google_oauth');url.searchParams.delete('reason');history.replaceState({},'',url.pathname+url.search+url.hash);}
function showOnboardingStep(step){
  const order=['profile','google','finish'];oobeStep=order.includes(step)?step:'profile';const activeIndex=order.indexOf(oobeStep);
  document.querySelectorAll('[data-oobe-panel]').forEach(panel=>panel.classList.toggle('hidden',panel.dataset.oobePanel!==oobeStep));
  document.querySelectorAll('[data-oobe-marker]').forEach(marker=>{const index=order.indexOf(marker.dataset.oobeMarker);marker.classList.toggle('active',index===activeIndex);marker.classList.toggle('complete',index<activeIndex);});
  const copy=OOBE_COPY[oobeStep];$('#oobeTitle').textContent=copy[0];$('#oobeSubtitle').textContent=copy[1];$('#oobeClose').classList.toggle('hidden',onboardingRequired);
  $('#oobeError').textContent='';
}
function setGoogleStatus(status){
  googleConnection=status||null;const box=$('#googleConnectionStatus'),title=box.querySelector('strong'),detail=box.querySelector('small');
  const connected=Boolean(status?.connected),configured=Boolean(status?.configured);
  box.classList.toggle('connected',connected);box.classList.toggle('not-connected',!connected);
  title.textContent=connected?'Google Workspace connected':configured?'OAuth client ready':'Google Workspace is not connected';
  detail.textContent=connected?(status.email||'Read-only access is active.'):configured?'Select Connect Google to authorize read access.':'Upload the OAuth client JSON from Google Cloud.';
  $('#googleRedirectUri').textContent=status?.redirect_uri||'Unavailable';
  $('#googleDisconnect').classList.toggle('hidden',!connected);
  $('#googleForget').classList.toggle('hidden',!configured);
  $('#googleConnect').textContent=connected?'Reconnect Google':'Connect Google';
  $('#oobeGoogleContinue').textContent=connected?'Continue':'Skip for now';
  if(status?.configuration_error)$('#oobeError').textContent=status.configuration_error.message||'The Google OAuth redirect is not configured correctly.';
}
async function loadGoogleConnectionStatus(){
  const box=$('#googleConnectionStatus');box.querySelector('strong').textContent='Checking connection…';box.querySelector('small').textContent='';box.classList.remove('connected','not-connected');
  try{const status=await api('/api/integrations/google');setGoogleStatus(status);return status;}catch(err){setGoogleStatus(null);$('#oobeError').textContent=err.message||String(err);return null;}
}
async function openOnboarding({reset=false,step='profile'}={}){
  onboardingReset=reset;const state=await api('/api/onboarding');onboardingRequired=!state.completed;const id=state.identity||{},prefs=state.preferences||{};
  $('#oobeName').value=id.name||'';$('#oobeRole').value=id.role||'';$('#oobeTimezone').value=id.timezone||Intl.DateTimeFormat().resolvedOptions().timeZone||'';$('#oobeLocation').value=id.location||'';$('#oobeEmail').value=id.email||'';$('#oobeInterests').value=Array.isArray(id.interests)?id.interests.join(', '):'';$('#oobeStyle').value=prefs.response?.style||'concise';$('#oobeResearch').value=prefs.search_depth?.depth||'balanced';$('#oobeError').textContent='';
  showOnboardingStep(step);const dialog=$('#onboardingDialog');if(!dialog.open)dialog.showModal();if(step==='google')await loadGoogleConnectionStatus();
}
async function checkOnboarding(){
  try{const params=new URLSearchParams(location.search),oauthResult=params.get('google_oauth');if(oauthResult){await openOnboarding({step:'google'});if(oauthResult==='connected'){$('#oobeError').textContent='';}else{$('#oobeError').textContent=params.get('reason')==='access_denied'?'Google access was not granted. You can try again or skip for now.':'Google authorization did not complete. Please try connecting again.';}clearGoogleOAuthQuery();return;}const state=await api('/api/onboarding');if(!state.completed)await openOnboarding();}catch(e){console.error('Onboarding check failed',e);}
}
function closeOnboarding(){if(onboardingRequired)return;$('#onboardingDialog').close();clearGoogleOAuthQuery();}
$('#onboardingDialog').addEventListener('cancel',e=>{if(onboardingRequired)e.preventDefault();});
$('#oobeClose').addEventListener('click',closeOnboarding);
$('#profileSetup').addEventListener('click',()=>openOnboarding({reset:true,step:'profile'}));
$('#connectionsSetup').addEventListener('click',()=>openOnboarding({step:'google'}));
$('#oobeGoogleBack').addEventListener('click',()=>showOnboardingStep('profile'));
$('#oobeGoogleContinue').addEventListener('click',()=>{showOnboardingStep('finish');const target=$('#oobeFinishStatus');target.textContent=googleConnection?.connected?`Connected to ${googleConnection.email||'Google Workspace'} with read-only access.`:'Google Workspace was skipped. Connect it later from Connections.';});
$('#oobeFinish').addEventListener('click',closeOnboarding);
$('#googleConnect').addEventListener('click',async()=>{
  const button=$('#googleConnect');button.disabled=true;$('#oobeError').textContent='';
  try{const file=$('#googleClientFile').files?.[0];if(file){const fd=new FormData();fd.append('file',file);await api('/api/integrations/google/client',{method:'POST',body:fd});$('#googleClientFile').value='';}const result=await api('/api/integrations/google/authorize',{method:'POST'});const target=new URL(result.authorization_url);if(target.protocol!=='https:'||target.hostname!=='accounts.google.com')throw new Error('The authorization destination was invalid.');location.assign(target.href);}catch(err){$('#oobeError').textContent=err.message||String(err);button.disabled=false;await loadGoogleConnectionStatus();}
});
$('#googleDisconnect').addEventListener('click',async()=>{
  if(!confirm('Disconnect Google Workspace and revoke the stored token?'))return;const button=$('#googleDisconnect');button.disabled=true;$('#oobeError').textContent='';try{await api('/api/integrations/google/connection',{method:'DELETE'});await loadGoogleConnectionStatus();}catch(err){$('#oobeError').textContent=err.message||String(err);}finally{button.disabled=false;}
});
$('#googleForget').addEventListener('click',async()=>{
  if(!confirm('Remove the Google connection and encrypted OAuth client configuration from this harness?'))return;const button=$('#googleForget');button.disabled=true;$('#oobeError').textContent='';try{await api('/api/integrations/google/client',{method:'DELETE'});await loadGoogleConnectionStatus();}catch(err){$('#oobeError').textContent=err.message||String(err);}finally{button.disabled=false;}
});
$('#onboardingForm').addEventListener('submit',async e=>{
  e.preventDefault();if(oobeStep!=='profile')return;const save=$('#oobeSave');save.disabled=true;$('#oobeError').textContent='';
  try{let profile_image_path='';const photo=$('#oobePhoto').files?.[0];if(photo){const fd=new FormData();fd.append('file',photo);const uploaded=await api('/api/upload',{method:'POST',body:fd});profile_image_path=uploaded.path||'';}const payload={name:$('#oobeName').value.trim(),role:$('#oobeRole').value.trim(),timezone:$('#oobeTimezone').value.trim()||'UTC',location:$('#oobeLocation').value.trim(),email:$('#oobeEmail').value.trim(),interests:$('#oobeInterests').value.split(',').map(x=>x.trim()).filter(Boolean),response_style:$('#oobeStyle').value,research_depth:$('#oobeResearch').value,profile_image_path,reset:onboardingReset};await api('/api/onboarding',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});onboardingRequired=false;onboardingReset=false;$('#oobePhoto').value='';refreshProfileImage();showOnboardingStep('google');await loadGoogleConnectionStatus();}catch(err){$('#oobeError').textContent=err.message||String(err);}finally{save.disabled=false;}
});

restoreSidebarState();
document.addEventListener('keydown',(e)=>{if(e.key!=='Escape')return;if(!$('#conversationSearch').classList.contains('hidden')){$('#conversationSearchClose').click();return;}if(appShell.classList.contains('workspace-open')){toggleWorkspace(false);return;}if(appShell.classList.contains('mobile-sidebar-open'))toggleLeftSidebar(false);});
window.addEventListener('resize',()=>{
  if(window.matchMedia('(max-width: 760px)').matches){
    $('#sidebarExpand').classList.add('hidden');
  }else{
    appShell.classList.remove('mobile-sidebar-open');
    setLeftSidebarCollapsed(readUiState(UI_STATE_KEYS.left,false),{persist:false});
  }
});
refreshProfileImage();
async function bootstrapWebUi(){
  // Sidebar metadata and secondary panels must never prevent chat startup. A
  // failed health/jobs/reminders request used to reject this Promise and abort
  // conversation restoration + WebSocket setup entirely.
  await Promise.allSettled([loadTheme(),loadHealth(),loadSlashCommands(),loadJobs(),loadReminders(),loadWorkspace('')]);
  const restored=readActiveConversation();
  if(restored)setActiveConversation(restored,{persist:false});
  try{
    let rows=await loadConversations();
    if(!activeConversationId||!rows.some(row=>row.id===activeConversationId)){
      if(rows.length)setActiveConversation(rows[0].id);
      else await createFreshConversation();
      rows=await loadConversations();
    }else setActiveConversation(activeConversationId);
  }catch(err){
    console.error('Conversation index load failed',err);
    if(!activeConversationId)setActiveConversation('default');
  }
  await Promise.allSettled([loadHistory(),loadState()]);
  showPanel('chat');
  connect();
  await checkOnboarding();
}
bootstrapWebUi().catch(err=>{console.error('Web UI bootstrap failed',err);setStatus('Startup failed','error');if(!socket||socket.readyState===WebSocket.CLOSED)connect();});
