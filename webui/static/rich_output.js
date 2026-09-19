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

  function inlineMd(v){
    let s=esc(v);
    s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
    s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
    s=s.replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
    s=s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,(_,label,url)=>`<a href="${esc(safeLink(url))}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    return s;
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

  function renderMarkdown(text){
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

  return {esc,inlineMd,renderMarkdown,splitTableRow,separatorAlignment,weatherIcon,weatherLineMeta,tableHeaderIcon};
});
