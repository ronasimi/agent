(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.RichOutput=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';

  function esc(v){
    return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function safeLink(url){
    try{
      const u=new URL(url,'http://localhost/');
      return ['http:','https:'].includes(u.protocol)?u.href:'#';
    }catch{return '#';}
  }

  function safeMediaLink(url){
    const value=String(url||'').trim();
    if(/^\/(?:api\/files\/|api\/profile-image(?:\?|$))/.test(value))return value;
    try{
      const parsed=new URL(value,'http://localhost/');
      return parsed.protocol==='https:'?parsed.href:'#';
    }catch{return '#';}
  }

  function inlineMd(v){
    const media=[];
    let raw=String(v??'').replace(/!\[([^\]\n]*)\]\(([^\s)]+)\)/g,(_,label,url)=>{
      const safe=safeMediaLink(url);if(safe==='#')return _;
      const id=media.length;media.push(`<a class="inline-media-link" href="${esc(safe)}" target="_blank" rel="noopener noreferrer"><img class="inline-markdown-media" src="${esc(safe)}" alt="${esc(label||'Inline image')}" loading="lazy"></a>`);return `@@MEDIA${id}@@`;
    });
    let s=esc(raw);
    s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
    s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
    s=s.replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
    s=s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,(_,label,url)=>`<a href="${esc(safeLink(url))}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    return s.replace(/@@MEDIA(\d+)@@/g,(_,idx)=>media[Number(idx)]||'');
  }

  function emailField(value){
    if(Array.isArray(value))return value.map(emailField).filter(Boolean).join(', ');
    if(value&&typeof value==='object'){
      const address=String(value.email||value.address||value.value||'').trim();const name=String(value.name||'').trim();
      return name&&address?`${name} <${address}>`:address||name;
    }
    return String(value??'').trim();
  }

  function emailBody(value){
    if(value&&typeof value==='object'&&!Array.isArray(value))value=value.text??value.plain??value.body??value.content??value.html??'';
    return String(value??'').trim();
  }

  function normalizeEmailObject(value){
    if(!value||typeof value!=='object'||Array.isArray(value))return null;
    let source=value;
    for(const key of ['email','draft','preview','message']){
      if(source[key]&&typeof source[key]==='object'&&!Array.isArray(source[key])){source=source[key];break;}
    }
    const lower={};for(const[key,item]of Object.entries(source))lower[String(key).toLowerCase()]=item;
    const email={
      from:emailField(lower.from||lower.sender),to:emailField(lower.to||lower.recipients),
      cc:emailField(lower.cc),bcc:emailField(lower.bcc),subject:emailField(lower.subject),
      body:emailBody(lower.body??lower.text??lower.content??lower.html??''),
    };
    return email.subject&&(email.to||email.from)?email:null;
  }

  function parseEmailContent(value){
    if(value&&typeof value==='object'&&!Array.isArray(value)){
      const email=normalizeEmailObject(value);return email?{prefix:'',email}:null;
    }
    const src=String(value??'').replace(/\r\n/g,'\n').trim();if(!src)return null;
    let candidate=src;
    const fenced=src.match(/^```(?:json|email)?\s*\n([\s\S]*?)\n```$/i);if(fenced)candidate=fenced[1].trim();
    if(candidate.startsWith('{')&&candidate.endsWith('}')){
      try{const email=normalizeEmailObject(JSON.parse(candidate));if(email)return{prefix:'',email};}catch{}
    }

    const lines=src.split('\n');let start=-1;
    const headerLine=line=>String(line||'').replace(/^\s{0,3}#{1,3}\s+/,'').replace(/\*\*/g,'').trim().match(/^(To|From|Cc|Bcc|Subject)\s*:\s*(.*)$/i);
    for(let i=0;i<lines.length;i++){if(headerLine(lines[i])){start=i;break;}}
    if(start<0)return null;
    const headers={};let cursor=start;
    while(cursor<lines.length){
      const match=headerLine(lines[cursor]);
      if(!match){if(!lines[cursor].trim())cursor++;break;}
      headers[match[1].toLowerCase()]=match[2].trim();cursor++;
    }
    if(!headers.subject||(!headers.to&&!headers.from))return null;
    const email={from:headers.from||'',to:headers.to||'',cc:headers.cc||'',bcc:headers.bcc||'',subject:headers.subject,body:lines.slice(cursor).join('\n').trim()};
    return{prefix:lines.slice(0,start).join('\n').trim(),email};
  }

  function renderEmailCard(email){
    const rows=[['From',email.from],['To',email.to],['Cc',email.cc],['Bcc',email.bcc]].filter(([,value])=>value);
    const meta=rows.map(([label,value])=>`<div class="email-meta-row"><span>${label}</span><strong>${esc(value)}</strong></div>`).join('');
    const body=email.body?renderMarkdownCore(email.body):'<p class="email-empty-body">No message body</p>';
    return `<section class="email-card" aria-label="Email preview"><div class="email-card-meta">${meta}<div class="email-subject"><span>Subject</span><h3>${esc(email.subject)}</h3></div></div><div class="email-card-body">${body}</div></section>`;
  }

  function extractWorkspaceAttachments(value){
    const refs=[];const seen=new Set();const add=(path,name='')=>{const clean=String(path||'').trim();const base=clean.split('/').pop().toLowerCase();if(!clean.startsWith('/app/workspace/')||base.endsWith('.lock')||seen.has(clean))return;seen.add(clean);refs.push({path:clean,name:String(name||clean.split('/').pop()||'attachment')});};
    let text=String(value??'').replace(/(?:^|\n)Attached text file `([^`\n]+)` \((\/app\/workspace\/[^)\n]+)\):\n\n```text\n[\s\S]*?\n```(?=\n|$)/g,(_,name,path)=>{add(path,name);return'\n';});
    text=text.replace(/(?:^|\n)Attached file:\s*(\/app\/workspace\/[^\n]+)(?=\n|$)/g,(_,path)=>{add(path.trim());return'\n';});
    return{text:text.replace(/\n{3,}/g,'\n\n').trim(),attachments:refs};
  }

  function plainText(value){
    return String(value??'')
      .replace(/[`*_>#~\[\]]/g,'')
      .replace(/\([^)]*\)/g,'')
      .replace(/\s+/g,' ')
      .trim();
  }

  function weatherIcon(value){
    const text=plainText(value).toLowerCase();
    if(!text)return'';
    if(/\b(thunder|thunderstorm|lightning)\b/.test(text))return'⛈️';
    if(/\b(snow|flurr|blizzard|sleet|ice pellet|freezing rain)\b/.test(text))return'❄️';
    if(/\b(rain|showers?|downpour)\b/.test(text))return'🌧️';
    if(/\b(drizzle|sprinkle)\b/.test(text))return'🌦️';
    if(/\b(fog|mist|haze)\b/.test(text))return'🌫️';
    if(/\b(partly cloudy|mostly sunny|sun and cloud|partly sunny)\b/.test(text))return'⛅';
    if(/\b(overcast|cloudy|clouds)\b/.test(text))return'☁️';
    if(/\b(sunny|sunshine|clear sky|clear)\b/.test(text))return'☀️';
    if(/\b(sunrise)\b/.test(text))return'🌅';
    if(/\b(sunset)\b/.test(text))return'🌇';
    if(/\b(wind|winds|windy|gust|gusts)\b/.test(text))return'💨';
    if(/\b(humidity|humid|dew point)\b/.test(text))return'💧';
    if(/\b(precip|chance of rain|chance of snow|pop\b)/.test(text))return'☔';
    if(/\b(temperature|temp\.?|feels like|high|low)\b/.test(text)||/[+-]?\d+(?:\.\d+)?\s*°\s*[cf]?\b/i.test(text))return'🌡️';
    if(/\b(weather|forecast)\b/.test(text))return'🌤️';
    return'';
  }

  function weatherLineMeta(value){
    const text=plainText(value);
    const lower=text.toLowerCase();
    const label=/^(?:temperature|temp\.?|feels like|conditions?|weather|forecast|humidity|wind(?:s| speed)?|gusts?|precipitation|precip\.?|rain|snow|sunrise|sunset|high|low)\s*[:—-]/i.test(text);
    const condition=/\b(?:sunny|clear|cloudy|overcast|rain|showers?|drizzle|snow|flurr|sleet|fog|mist|haze|thunder|lightning|windy)\b/i.test(text);
    const measurable=/\b(?:humidity|wind|precipitation|forecast|weather)\b/i.test(text)||/\d+\s*%/.test(text)||/[+-]?\d+(?:\.\d+)?\s*°\s*[cf]?\b/i.test(text);
    const icon=weatherIcon(text);
    return {icon,isWeather:Boolean(icon&&(label||condition||measurable||lower.startsWith('today')||lower.startsWith('tonight')||lower.startsWith('tomorrow')))};
  }

  function splitTableRow(line){
    let src=String(line??'').trim();
    if(!src.includes('|'))return null;
    if(src.startsWith('|'))src=src.slice(1);
    if(src.endsWith('|')&&!src.endsWith('\\|'))src=src.slice(0,-1);
    const cells=[];let buf='';let escaped=false;
    for(const ch of src){
      if(escaped){buf+=ch;escaped=false;continue;}
      if(ch==='\\'){escaped=true;buf+=ch;continue;}
      if(ch==='|'){cells.push(buf.trim().replace(/\\\|/g,'|'));buf='';continue;}
      buf+=ch;
    }
    cells.push(buf.trim().replace(/\\\|/g,'|'));
    return cells.length>=2?cells:null;
  }

  function separatorAlignment(cell){
    const src=String(cell??'').replace(/\s+/g,'');
    if(!/^:?-{2,}:?$/.test(src))return null;
    if(src.startsWith(':')&&src.endsWith(':'))return'center';
    if(src.endsWith(':'))return'right';
    return'left';
  }

  function tableHeaderIcon(value){
    const text=plainText(value).toLowerCase();
    if(!text)return'';
    if(/\b(date|day)\b/.test(text))return'📅';
    if(/\b(time|hour)\b/.test(text))return'🕒';
    if(/\b(temp|temperature|feels|high|low)\b/.test(text))return'🌡️';
    if(/\b(conditions?|weather|forecast)\b/.test(text))return'🌤️';
    if(/\b(precip|rain|snow|chance)\b/.test(text))return'☔';
    if(/\b(wind|gust)\b/.test(text))return'💨';
    if(/\b(humidity|dew)\b/.test(text))return'💧';
    if(/\b(status|state)\b/.test(text))return'●';
    return'';
  }

  function isNumericCell(value){
    const text=plainText(value);
    return /^[-+]?[$€£¥]?\s*\d[\d,.]*(?:\s*(?:%|°[CF]?|ms|s|min|h|KB|MB|GB|TB))?$/i.test(text);
  }

  function renderTable(lines,index){
    const headers=splitTableRow(lines[index]);
    const separator=splitTableRow(lines[index+1]);
    if(!headers||!separator||headers.length!==separator.length)return null;
    const alignments=separator.map(separatorAlignment);
    if(alignments.some(v=>v===null))return null;

    const rows=[];let cursor=index+2;
    while(cursor<lines.length){
      const row=splitTableRow(lines[cursor]);
      if(!row)break;
      const normalized=row.slice(0,headers.length);
      while(normalized.length<headers.length)normalized.push('');
      rows.push(normalized);cursor+=1;
    }

    const headerHtml=headers.map((cell,i)=>{
      const icon=tableHeaderIcon(cell);
      const iconHtml=icon?`<span class="table-header-icon" aria-hidden="true">${icon}</span>`:'';
      return `<th scope="col" style="text-align:${alignments[i]}">${iconHtml}${inlineMd(cell)}</th>`;
    }).join('');
    const bodyHtml=rows.map(row=>`<tr>${row.map((cell,i)=>{
      const classes=isNumericCell(cell)?' class="numeric"':'';
      const align=alignments[i]==='left'&&isNumericCell(cell)?'right':alignments[i];
      const header=plainText(headers[i]).toLowerCase();
      const conditionIcon=/\b(conditions?|weather|forecast)\b/.test(header)?weatherIcon(cell):'';
      const prefix=conditionIcon?`<span class="weather-emoji compact" aria-hidden="true">${conditionIcon}</span>`:'';
      return `<td${classes} style="text-align:${align}">${prefix}${inlineMd(cell)}</td>`;
    }).join('')}</tr>`).join('');
    return {
      html:`<div class="table-wrap" role="region" aria-label="Data table" tabindex="0"><table class="md-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`,
      nextIndex:cursor,
    };
  }

  function renderMarkdownCore(text){
    const src=String(text??'').replace(/\r\n/g,'\n');
    const blocks=[];
    const tokenized=src.replace(/```([^\n]*)\n?([\s\S]*?)```/g,(_,lang,code)=>{
      const id=blocks.length;
      blocks.push(`<pre><code data-lang="${esc(lang.trim())}">${esc(code.replace(/\n$/,''))}</code></pre>`);
      return `\n@@CODE${id}@@\n`;
    });
    const lines=tokenized.split('\n');let out=[],list=null;let i=0;
    const closeList=()=>{if(list){out.push(`</${list}>`);list=null;}};
    while(i<lines.length){
      const raw=lines[i];const line=raw.trimEnd();
      if(/^@@CODE\d+@@$/.test(line.trim())){closeList();out.push(line.trim());i+=1;continue;}
      if(!line.trim()){closeList();i+=1;continue;}

      const table=renderTable(lines,i);
      if(table){closeList();out.push(table.html);i=table.nextIndex;continue;}

      let m=line.match(/^(#{1,3})\s+(.+)$/);
      if(m){
        closeList();
        const level=m[1].length;const meta=weatherLineMeta(m[2]);
        const icon=meta.isWeather?`<span class="weather-emoji heading" aria-hidden="true">${meta.icon}</span>`:'';
        out.push(`<h${level}${meta.isWeather?' class="weather-heading"':''}>${icon}${inlineMd(m[2])}</h${level}>`);i+=1;continue;
      }
      m=line.match(/^[-*]\s+(.+)$/);
      if(m){if(list!=='ul'){closeList();list='ul';out.push('<ul>');}const meta=weatherLineMeta(m[1]);const icon=meta.isWeather?`<span class="weather-emoji compact" aria-hidden="true">${meta.icon}</span>`:'';out.push(`<li${meta.isWeather?' class="weather-list-item"':''}>${icon}${inlineMd(m[1])}</li>`);i+=1;continue;}
      m=line.match(/^\d+[.)]\s+(.+)$/);
      if(m){if(list!=='ol'){closeList();list='ol';out.push('<ol>');}out.push(`<li>${inlineMd(m[1])}</li>`);i+=1;continue;}
      m=line.match(/^>\s?(.*)$/);
      if(m){closeList();out.push(`<blockquote>${inlineMd(m[1])}</blockquote>`);i+=1;continue;}
      closeList();
      const meta=weatherLineMeta(line);
      const icon=meta.isWeather?`<span class="weather-emoji" aria-hidden="true">${meta.icon}</span>`:'';
      out.push(`<p${meta.isWeather?' class="weather-line"':''}>${icon}${inlineMd(line)}</p>`);
      i+=1;
    }
    closeList();
    return out.join('').replace(/@@CODE(\d+)@@/g,(_,idx)=>blocks[Number(idx)]||'');
  }

  function renderMarkdown(text){
    const parsed=parseEmailContent(text);
    if(!parsed)return renderMarkdownCore(text);
    const intro=parsed.prefix?renderMarkdownCore(parsed.prefix):'';
    return intro+renderEmailCard(parsed.email);
  }

  return {esc,inlineMd,renderMarkdown,renderEmailCard,parseEmailContent,extractWorkspaceAttachments,splitTableRow,separatorAlignment,weatherIcon,weatherLineMeta,tableHeaderIcon};
});
