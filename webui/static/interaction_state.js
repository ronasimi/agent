(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.InteractionState=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';

  const VALID_DECISIONS=new Set(['up','down']);

  class RecipeDecisionStore{
    constructor({storage,key='agent.webui.recipeDecisions.v1',limit=100}={}){
      this.storage=storage||null;
      this.key=String(key||'agent.webui.recipeDecisions.v1');
      this.limit=Math.max(10,Math.min(Number(limit)||100,500));
      this.items=this._read();
    }

    _read(){
      if(!this.storage)return[];
      try{
        const parsed=JSON.parse(this.storage.getItem(this.key)||'[]');
        if(!Array.isArray(parsed))return[];
        return parsed.map(item=>this._normalize(item)).filter(Boolean).slice(-this.limit);
      }catch{return[];}
    }

    _normalize(item){
      if(!item||typeof item!=='object')return null;
      const id=String(item.id||'').slice(0,180);
      const conversationId=String(item.conversationId||'').slice(0,128);
      const message=String(item.message||'').slice(0,1000);
      if(!id||!conversationId||!message)return null;
      const decision=VALID_DECISIONS.has(item.decision)?item.decision:null;
      const createdAt=String(item.createdAt||new Date().toISOString()).slice(0,40);
      return{id,conversationId,message,decision,createdAt};
    }

    _write(){
      this.items=this.items.slice(-this.limit);
      if(!this.storage)return;
      try{this.storage.setItem(this.key,JSON.stringify(this.items));}catch{}
    }

    upsert(item){
      const normalized=this._normalize(item);if(!normalized)return null;
      const index=this.items.findIndex(row=>row.id===normalized.id);
      if(index>=0){
        normalized.decision=this.items[index].decision||normalized.decision;
        normalized.createdAt=this.items[index].createdAt||normalized.createdAt;
        this.items[index]=normalized;
      }else this.items.push(normalized);
      this._write();return{...normalized};
    }

    decide(id,decision){
      if(!VALID_DECISIONS.has(decision))return null;
      const index=this.items.findIndex(row=>row.id===String(id||''));
      if(index<0)return null;
      this.items[index]={...this.items[index],decision};this._write();
      return{...this.items[index]};
    }

    list(conversationId){
      const target=String(conversationId||'');
      return this.items.filter(item=>item.conversationId===target).map(item=>({...item}));
    }

    removeConversation(conversationId){
      const target=String(conversationId||'');
      this.items=this.items.filter(item=>item.conversationId!==target);this._write();
    }
  }

  return{RecipeDecisionStore};
});
