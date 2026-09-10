'use strict';
const $ = id => document.getElementById(id);
let state = null, busy = false, showNew = false;
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
function tokenText(text) { return text.replaceAll(' ', '·').replaceAll('\n','↵\n').replaceAll('\t','⇥'); }
async function refresh() {
  state = await api('/api/session');
  const {episodes} = await api('/api/episodes');
  $('episodes').replaceChildren();
  for (const episode of episodes) {
    const button=document.createElement('button'); button.className='episode';
    button.classList.toggle('active',episode.episode_id===state.episode?.episode_id);
    const title=document.createElement('strong'); title.textContent=episode.title || episode.initial_text.slice(0,40) || episode.episode_id.slice(0,8);
    const detail=document.createElement('small'); detail.textContent=`${episode.status} · ${episode.episode_id.slice(0,8)}`;
    button.append(title,detail); button.disabled=['completed','failed'].includes(episode.status);
    if(button.disabled) button.title='Sealed episode. Replay support is planned for a later preview.';
    button.onclick=()=>mutate('',{episode_id:episode.episode_id}); $('episodes').append(button);
  }
  const active=!!state.episode;
  $('welcome').hidden=active && !showNew; $('editor').hidden=!active || showNew;
  $('cancel-new').hidden=!active;
  $('live').hidden=true; $('no-choices').hidden=false;
  $('no-choices').textContent=active ? 'This episode is paused or ended.' : 'Open an episode to inspect the model’s choices.';
  if(!active) return;
  $('episode-label').textContent=`EPISODE ${state.episode.episode_id.slice(0,8)}`;
  $('status').textContent=state.ended ? 'Ended' : state.checkpointed ? 'Allowance reached' : 'Live';
  const prompt=document.createElement('span'); prompt.className='prompt-text'; prompt.textContent=state.episode.initial_text;
  $('context').replaceChildren(prompt,document.createTextNode(state.episode.visible_text));
  $('context').scrollTop=$('context').scrollHeight;
  $('boundary-label').textContent=`Boundary ${state.boundary}`;
  $('allowance').textContent=state.remaining===null?'Unlimited allowance':`${state.remaining} tokens remaining`;
  $('boundary').max=state.boundary; $('boundary').value=state.boundary;
  $('write-button').disabled=state.ended || state.checkpointed;
  $('rewind').disabled=state.ended; $('continue').disabled=state.ended; $('finish').disabled=state.ended || state.checkpointed;
  if(!state.ended && !state.checkpointed && !showNew) await choices();
}
async function choices() {
  const o=await api(`/api/observation?start=${encodeURIComponent($('rank').value)}&count=12`);
  if(o.revision!==state.revision) throw new Error('The session changed. Refresh to see current choices.');
  $('live').hidden=false; $('no-choices').hidden=true; $('proposal').textContent=tokenText(o.proposal.text);
  $('candidates').replaceChildren();
  for(const c of o.candidates) {
    const b=document.createElement('button'); b.className='candidate'; b.style.setProperty('--prob',`${Math.min(100,c.raw_probability*100)}%`);
    b.title=`Token ${c.token_id} · sampler probability ${(c.decoder_probability*100).toFixed(3)}%`;
    const r=document.createElement('span'); r.className='rank'; r.textContent=c.rank;
    const t=document.createElement('code'); t.textContent=c.is_eog?'[End generation]':tokenText(c.text);
    const p=document.createElement('small'); p.textContent=`${(c.raw_probability*100).toFixed(2)}%`;
    b.append(r,t,p); b.onclick=()=>act({kind:'select-raw-rank',rank:c.rank}); $('candidates').append(b);
  }
}
async function work(fn) {
  if(busy)return; busy=true; document.body.setAttribute('aria-busy','true');
  const controls=[...document.querySelectorAll('button,input,textarea,select')];
  const disabled=controls.map(c=>c.disabled); controls.forEach(c=>c.disabled=true);
  try { await fn(); } catch(e) { message(e.message,true); }
  finally { controls.forEach((c,i)=>c.disabled=disabled[i]);
    if(state?.episode){$('write-button').disabled=state.ended||state.checkpointed;$('rewind').disabled=state.ended;$('continue').disabled=state.ended;$('finish').disabled=state.ended;}
    busy=false; document.body.removeAttribute('aria-busy'); }
}
async function mutate(op, payload={}) {
  await work(async()=>{
    message('Working…');
    try {
      const result=await api(`/api/session${op?'/'+op:''}`, {...payload,revision:state.revision,request_id:crypto.randomUUID()});
      showNew=false; $('rank').value=1;
      await refresh();
      message(result.result?.handoff_reason || result.notices?.join(' ') || 'Saved.');
      if(op==='actions' && payload.action.kind==='write' && result.result?.outcomes?.length) $('text').value='';
    } catch(e) { await refresh(); throw e; }
  });
}
const act=action=>mutate('actions',{action});
$('create').onsubmit=e=>{e.preventDefault();mutate('',{prompt:$('prompt').value});};
$('write').onsubmit=e=>{e.preventDefault();act({kind:'write',text:$('text').value,mode:$('exact').checked?'exact':'continuation'});};
$('accept').onclick=()=>act({kind:'accept'});
$('hold').onsubmit=e=>{e.preventDefault();act({kind:'hold',limit:Number($('limit').value),boundary:$('stop').value||null});};
$('rank-form').onsubmit=e=>{e.preventDefault();work(choices);};
$('new').onclick=()=>{showNew=true;$('welcome').hidden=false;$('editor').hidden=true;$('live').hidden=true;$('no-choices').hidden=false;$('prompt').focus();};
$('cancel-new').onclick=()=>work(async()=>{showNew=false;await refresh();});
$('refresh').onclick=()=>work(refresh);
$('fork').onclick=()=>mutate('fork',{boundary:Number($('boundary').value)});
$('rewind').onclick=()=>{$('rewind-description').textContent=`Return to boundary ${$('boundary').value} of ${state.boundary}.`;$('rewind-dialog').showModal();};
$('cancel-rewind').onclick=()=>$('rewind-dialog').close();
$('confirm-rewind').onclick=()=>{$('rewind-dialog').close();mutate('rewind',{boundary:Number($('boundary').value)});};
$('continue').onclick=()=>mutate('settings');
$('close').onclick=()=>mutate('close');
$('finish').onclick=()=>mutate('end');
work(refresh);
