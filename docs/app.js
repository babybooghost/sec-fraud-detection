'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, places=3) => Number(value).toFixed(places);
const pct = value => `${(value * 100).toFixed(1)}%`;
let data, modelKey, page = 0;
const pageSize = 8;
const color = {accent:'var(--accent)', muted:'#acb8a4', warm:'#ac793f', grid:'#e0e3da', ink:'var(--ink)'};
const configs = {
  reddit: {
    title:'Reddit volatility', eyebrow:'48-hour volatility prediction',
    intro:'Compare three classifiers using simulated Reddit embeddings and synthetic prices.',
    repo:'reddit-volatility-pipeline', other:'sec-fraud-detection', otherName:'SEC fraud detection', number:'01',
    detail:'Synthetic posts & prices. Simulated embeddings. No live market data.',
    primary:'The next 48 hours', primaryDescription:'Realized volatility from synthetic hourly prices. A spike exceeds 1.5× the trailing 48-hour value.',
    tableDescription:'One decision per hour represented in the corpus. Labels use the full subsequent 48 hours. Scores come from the selected model.',
    method:'No future text. No future preprocessing. Training outcomes must finish before the holdout begins. Even with synthetic data, the time boundary stays intact.',
    steps:[['Observe, then embed','Four finance subreddits. Text and engagement are usable only after their recorded observation time. This demo simulates 1,536-dimensional vectors.'],['Preserve global variation','Full-SVD PCA fits on training data only. No whitening or local-neighborhood clustering. The 2D plot shows only the first two components.'],['Purge, then evaluate','Exclude overlapping 48-hour outcomes before training Logistic Regression, SVM, and Random Forest. Evaluate later observations once.']],
    limitations:'<p><strong>All inputs are generated.</strong> These are not scraped Reddit posts or actual market prices. Offline vectors have no semantic meaning.</p><p>The 0.55 price-only AUC is a supplied historical reference, not a baseline model reproduced on these rows. Differences are descriptive, not evidence of outperformance.</p><p>This model targets volatility spikes, not price direction. Overlapping holdout windows are not independent observations.</p>'
  },
  sec: {
    title:'SEC fraud detection', eyebrow:'Financial statement analysis',
    intro:'Inspect XGBoost predictions, accounting features, and errors on later synthetic company-years.',
    repo:'sec-fraud-detection', other:'reddit-volatility-pipeline', otherName:'Reddit volatility', number:'02',
    detail:'Synthetic companies, statements & labels. No real SEC findings.',
    primary:'What drives the model', primaryDescription:'The six largest held-out mean absolute SHAP contributions, compared with training. Contributions are in raw log-odds units.',
    tableDescription:'Generated company-years, ordered by model score. A flagged observation is a model decision, not a finding of fraud.',
    method:'Fit on older filings and inspect later years. Missing values stay missing. The model weighs positive-class loss ten times more heavily, then reports the cost of its actual mistakes.',
    steps:[['Read the original filing','Normalized annual statements and AAER-style labels. Consecutive years only. No cross-company lags or later-filed information.'],['Engineer the red flags','Eight Beneish components plus six nonlinear interactions capture receivables, margins, asset quality, leverage, and accruals.'],['Audit the forward years','Cost-sensitive XGBoost, fixed training rules, and one training-only SHAP background. Real AAER publication delays still require separate label-vintage controls.']],
    limitations:'<p><strong>No real companies are represented.</strong> Accounts and labels are generated with a fixed seed and noisy relationships. Scores are illustrative, not measured fraud-detection performance.</p><p>Weighted XGBoost scores are not calibrated fraud probabilities. A score of 0.80 does not mean an 80% chance of fraud.</p><p>AAER outcomes are retrospective. This implementation does not resolve enforcement-availability delays. SHAP describes model behavior, not causation; PSI drift alerts are descriptive.</p>'
  }
};
function svg(title, body, width=660, height=320) {
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${body}</svg>`;
}
function linePath(points) { return points.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' '); }
function text(x,y,label,extra='') { return `<text x="${x}" y="${y}" ${extra}>${esc(label)}</text>`; }
function plotAxes(width,height,maxY,yLabel,xLabel,steps=4) {
  const l=54, r=width-18, t=30, b=height-44;
  let body=text(l,14,yLabel)+text((l+r)/2,height-5,xLabel,'text-anchor="middle"');
  for(let i=0;i<=steps;i++) {
    const y=t+(b-t)*i/steps;
    body+=`<line x1="${l}" y1="${y}" x2="${r}" y2="${y}" stroke="${color.grid}"/>`+text(l-9,y+4,number(maxY*(1-i/steps),maxY<=1?2:1),'text-anchor="end"');
  }
  return {l,r,t,b,body};
}
function renderTimeline() {
  const rows=data.timeline, width=700, height=360;
  const maxY=Math.max(...rows.flatMap(r=>[r.past,r.future]))*1.12;
  const a=plotAxes(width,height,maxY,'48h realized volatility (%)','Decision date (UTC)');
  const x=i=>a.l+(a.r-a.l)*i/(rows.length-1), y=v=>a.b-(a.b-a.t)*v/maxY;
  const boundary=rows.findIndex(r=>r.split==='holdout');
  let body=`<rect x="${x(boundary)}" y="${a.t}" width="${a.r-x(boundary)}" height="${a.b-a.t}" fill="var(--light)" opacity=".7"/>`+a.body;
  body+=`<path d="${linePath(rows.map((r,i)=>[x(i),y(r.past)]))}" fill="none" stroke="${color.muted}" stroke-width="2"/>`;
  body+=`<path d="${linePath(rows.map((r,i)=>[x(i),y(r.future)]))}" fill="none" stroke="${color.accent}" stroke-width="2.4"/>`;
  body+=`<line x1="${x(boundary)}" y1="${a.t}" x2="${x(boundary)}" y2="${a.b}" stroke="${color.accent}" stroke-dasharray="4 4"/>`+text(x(boundary)+8,a.t+14,'HOLDOUT');
  [0,.25,.5,.75,1].forEach(f=>{const i=Math.round(f*(rows.length-1));body+=text(x(i),a.b+21,new Date(rows[i].time).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}),'text-anchor="middle"');});
  $('primary-chart').innerHTML=svg('Synthetic trailing and forward 48-hour volatility over time. Shaded region is held out.',body,width,height);
}
function renderAttribution() {
  const rows=[...data.attribution].sort((a,b)=>b.oot_mean_abs_shap-a.oot_mean_abs_shap).slice(0,6);
  const width=700,height=360,l=145,r=width-30,t=20,b=height-42;
  const max=Math.max(...rows.flatMap(r=>[r.train_mean_abs_shap,r.oot_mean_abs_shap]))*1.16;
  let body='';
  [0,.25,.5,.75,1].forEach(f=>{const x=l+(r-l)*f;body+=`<line x1="${x}" y1="${t}" x2="${x}" y2="${b}" stroke="${color.grid}"/>`+text(x,b+21,number(max*f,2),'text-anchor="middle"');});
  rows.forEach((row,i)=>{
    const y=t+i*48+8;
    body+=text(l-12,y+17,row.feature,'text-anchor="end"');
    body+=`<rect x="${l}" y="${y}" width="${(r-l)*row.train_mean_abs_shap/max}" height="9" fill="${color.muted}"/><rect x="${l}" y="${y+13}" width="${(r-l)*row.oot_mean_abs_shap/max}" height="9" fill="${color.accent}"><title>${esc(row.feature)}: holdout ${number(row.oot_mean_abs_shap)} log-odds</title></rect>`;
  });
  body+=text((l+r)/2,height-3,'Mean absolute SHAP contribution (log-odds)','text-anchor="middle"');
  $('primary-chart').innerHTML=svg('Training versus holdout SHAP importance for the six largest features',body,width,height);
}
function renderROC(model, counts) {
  const width=440,height=285,a=plotAxes(width,height,1,'True positive rate','False positive rate');
  const x=v=>a.l+(a.r-a.l)*v,y=v=>a.b-(a.b-a.t)*v;
  let body=a.body+`<path d="M${x(0)},${y(0)}L${x(1)},${y(1)}" stroke="#9aa191" stroke-dasharray="4 4" fill="none"/>`;
  body+=`<path d="${linePath(model.roc.map(p=>[x(p[0]),y(p[1])]))}" stroke="${color.accent}" stroke-width="2.3" fill="none"/>`;
  [0,.25,.5,.75,1].forEach(f=>body+=text(x(f),a.b+20,number(f,2),'text-anchor="middle"'));
  const fpr=counts.fp/(counts.fp+counts.tn)||0,tpr=counts.tp/(counts.tp+counts.fn)||0;
  body+=`<circle cx="${x(fpr)}" cy="${y(tpr)}" r="5" fill="var(--white)" stroke="${color.accent}" stroke-width="2"><title>Selected threshold: false positive rate ${number(fpr)}, true positive rate ${number(tpr)}</title></circle>`;
  $('roc-chart').innerHTML=svg('ROC curve on synthetic holdout observations, with the selected threshold marked',body,width,height);
}
function renderProjection() {
  const rows=data.projection,width=440,height=285,l=52,r=width-15,t=24,b=height-42;
  const xs=rows.map(p=>p.x),ys=rows.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const x=v=>l+8+(r-l-16)*(v-xmin)/(xmax-xmin||1),y=v=>b-8-(b-t-16)*(v-ymin)/(ymax-ymin||1);
  let body=text(l,12,`PC2 (${pct(data.variance[1])} variance)`)+text((l+r)/2,height-5,`PC1 (${pct(data.variance[0])} variance)`,'text-anchor="middle"');
  [0,.5,1].forEach(f=>{const xx=l+(r-l)*f,yy=b-(b-t)*f;body+=`<line x1="${xx}" y1="${t}" x2="${xx}" y2="${b}" stroke="${color.grid}"/><line x1="${l}" y1="${yy}" x2="${r}" y2="${yy}" stroke="${color.grid}"/>`+text(xx,b+20,number(xmin+(xmax-xmin)*f,2),'text-anchor="middle"')+text(l-8,yy+4,number(ymin+(ymax-ymin)*f,2),'text-anchor="end"');});
  rows.forEach(p=>body+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3.2" opacity=".78" fill="${p.split==='holdout'?color.accent:p.split==='purged'?color.warm:color.muted}"><title>${esc(p.split)} observation: PC1 ${number(p.x)}, PC2 ${number(p.y)}</title></circle>`);
  $('secondary-chart').innerHTML=svg('First two global PCA components of simulated vectors, colored by time split',body,width,height);
}
function renderHistogram(model) {
  const bins=Array.from({length:10},()=>[0,0]);
  model.scores.forEach((score,i)=>bins[Math.min(9,Math.floor(score*10))][data.truth[i]]++);
  const maxY=Math.max(...bins.flat())*1.15 || 1,width=440,height=285,a=plotAxes(width,height,maxY,'Company-years','Weighted model score (not probability)');
  const step=(a.r-a.l)/10;
  let body=a.body;
  bins.forEach((bin,i)=>{bin.forEach((count,c)=>{const h=(a.b-a.t)*count/maxY;body+=`<rect x="${a.l+i*step+c*step*.42+2}" y="${a.b-h}" width="${step*.38}" height="${h}" fill="${c?color.accent:color.muted}"><title>Score ${number(i/10,1)} to ${number((i+1)/10,1)}, ${c?'positive':'negative'} labels: ${count}</title></rect>`;});});
  [0,.25,.5,.75,1].forEach(f=>body+=text(a.l+(a.r-a.l)*f,a.b+20,number(f,2),'text-anchor="middle"'));
  const marker=a.l+(a.r-a.l)*Number($('threshold').value);
  body+=`<line x1="${marker}" y1="${a.t}" x2="${marker}" y2="${a.b}" stroke="${color.warm}" stroke-width="2" stroke-dasharray="4 3"/>`;
  $('secondary-chart').innerHTML=svg('Synthetic label distribution by weighted XGBoost score. Dashed line is the selected threshold.',body,width,height);
}
function countsFor(scores,truth,threshold) {
  const counts={tp:0,tn:0,fp:0,fn:0};
  scores.forEach((score,i)=>counts[score>=threshold?(truth[i]?'tp':'fp'):(truth[i]?'fn':'tn')]++);
  return counts;
}
function configureModel() {
  const model=data.models[modelKey],margin=modelKey==='svm';
  $('threshold').min=margin?Math.min(-2,Math.floor(Math.min(...model.scores))):0;
  $('threshold').max=margin?Math.max(2,Math.ceil(Math.max(...model.scores))):1;
  $('threshold').value=model.decision_threshold;
  $('range-min').textContent=number($('threshold').min,2);
  $('range-max').textContent=number($('threshold').max,2);
  $('score-note').textContent=margin?'SVM decision margin. Scores are not probabilities. Default boundary: 0.':data.kind==='sec'?'Cost-weighted XGBoost score, not a calibrated fraud probability.':'Classifier score from simulated embeddings. Larger values rank a spike more highly.';
  $('score-heading').textContent=margin?'Decision margin':'Model score';
  page=0; render();
}
function renderTable(model, threshold) {
  const positive=data.kind==='reddit'?'Spike':'Positive label',negative=data.kind==='reddit'?'No spike':'Negative label';
  const rows=model.scores.map((score,i)=>({...data.records[i],score,predicted:score>=threshold,index:i})).filter(r=>!$('errors-only').checked||Number(r.predicted)!==r.target);
  if(data.kind==='sec') rows.sort((a,b)=>b.score-a.score||a.index-b.index);
  const pageCount=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.min(page,pageCount-1);
  $('records').innerHTML=rows.slice(page*pageSize,(page+1)*pageSize).map(row=>{
    const error=Number(row.predicted)!==row.target;
    const result=error?(row.predicted?'False alarm':'Missed positive'):'Correct';
    return `<tr><td class="record-id">${esc(row.id)}${row.year?` <span class="muted">/ ${row.year}</span>`:''}</td><td>${row.target?positive:negative}</td><td><div class="score-cell">${number(row.score)}${modelKey!=='svm'?`<span class="score-track"><i style="width:${Math.max(0,Math.min(100,row.score*100))}%"></i></span>`:''}</div></td><td>${row.predicted?(data.kind==='reddit'?'Spike':'Flagged'):(data.kind==='reddit'?'No spike':'Not flagged')}</td><td><span class="result-tag ${error?'error':''}">${result}</span></td></tr>`;
  }).join('') || '<tr><td colspan="5">No errors at this threshold in this synthetic sample.</td></tr>';
  $('page-info').textContent=rows.length?`${page*pageSize+1}–${Math.min((page+1)*pageSize,rows.length)} of ${rows.length} synthetic observations`:'0 matching observations';
  $('previous').disabled=page===0;$('next').disabled=page>=pageCount-1;
}
function render() {
  const model=data.models[modelKey], threshold=Number($('threshold').value),counts=countsFor(model.scores,data.truth,threshold);
  $('threshold-value').textContent=number(threshold,2);
  $('auc').textContent=number(model.roc_auc);$('ap').textContent=number(model.pr_auc);
  $('auc-note').textContent=data.kind==='reddit'?`${model.roc_auc>=.55?'+':''}${number(model.roc_auc-.55)} vs. supplied 0.55 reference`:'Ranking on later synthetic company-years';
  $('ap-note').textContent=`Positive-label prevalence: ${pct(data.truth.reduce((a,b)=>a+b,0)/data.truth.length)}`;
  $('third').textContent=data.kind==='sec'?10*counts.fn+counts.fp:data.test_count;
  $('third-note').textContent=data.kind==='sec'?`${counts.fn} misses × 10 + ${counts.fp} false alarms`:`${data.train_count} training / ${data.purged_count} purged`;
  $('matrix').innerHTML=[['tn','Correct negatives','good'],['fp','False alarms','bad'],['fn','Missed positives','bad'],['tp','Correct positives','good']].map(([key,label,style])=>`<div class="${style}"><strong>${counts[key]}</strong><small>${label}</small></div>`).join('');
  const precision=counts.tp+counts.fp?pct(counts.tp/(counts.tp+counts.fp)):'N/A';
  $('decision-summary').innerHTML=`<span><b>${precision}</b>Precision${precision==='N/A'?' (no flags)':''}</span><span><b>${pct(counts.tp/(counts.tp+counts.fn)||0)}</b>Recall</span><span><b>${counts.tp+counts.fp}</b>Flagged</span>`;
  renderROC(model,counts);if(data.kind==='sec') renderHistogram(model);renderTable(model,threshold);
}
function initialize() {
  if(!data.synthetic||!configs[data.kind]||!data.truth.length) throw new Error('Expected a labeled synthetic dataset.');
  const c=configs[data.kind];document.body.classList.toggle('sec',data.kind==='sec');
  document.title=`${data.kind==='reddit'?'Reddit Volatility':'SEC Fraud Detection'} | Synthetic Research Demo`;
  for(const [id,value] of Object.entries({'title':c.title,'intro':c.intro,'eyebrow':c.eyebrow,'top-label':`${data.kind==='reddit'?'Reddit volatility':'SEC fraud detection'} / Demo`,'project-number':'Research projects','synthetic-detail':c.detail,'primary-title':c.primary,'primary-description':c.primaryDescription,'table-description':c.tableDescription,'method-intro':c.method,'data-provenance':data.provenance,'n-count':`n = ${data.test_count}`,'context-count':data.kind==='reddit'?`${data.source_count} generated posts / 4 subreddits`:`${data.source_count} generated company-years / 48 companies`,'context-time':data.kind==='reddit'?'Jan–Feb 2024 / hourly':'2013–2023 / annual filings','context-method':data.kind==='reddit'?'48-hour horizon':'14 accounting features','primary-tag':data.kind==='reddit'?'TIME-ORDERED':'TREE SHAP','primary-caption':data.kind==='reddit'?'Source: synthetic hourly prices, seed 7. RV is the square root of summed squared log returns, not annualized.':'Source: synthetic statements, seed 21. Same training-only background for both periods. SHAP is descriptive, not causal.','record-heading':data.kind==='reddit'?'Decision time (UTC)':'Synthetic company / fiscal year','third-label':data.kind==='reddit'?'HOLDOUT OBSERVATIONS':'ASYMMETRIC ERROR COST'})) $(id).textContent=value;
  $('source-link').href=`https://github.com/babybooghost/${c.repo}`;
  $('other-project').href=`https://babybooghost.github.io/${c.other}/`;
  $('other-project').innerHTML=`${esc(c.otherName)} <span aria-hidden="true">↗</span>`;
  $('limitations').innerHTML=c.limitations;
  $('method-steps').innerHTML=c.steps.map(([title,description],i)=>`<div class="method-step"><span>0${i+1}</span><div><h3>${esc(title)}</h3><p>${esc(description)}</p></div></div>`).join('');
  $('model').innerHTML=Object.entries(data.models).map(([id,m])=>`<option value="${esc(id)}">${esc(m.name)}</option>`).join('');
  if(data.kind==='reddit') {
    $('primary-legend').innerHTML='<span><i class="key accent"></i>Forward 48h RV</span><span><i class="key pale"></i>Trailing 48h RV</span><span>Shaded: holdout period</span>';
    $('secondary-tag').textContent=`${data.variance.length} COMPONENTS FITTED`;
    $('secondary-legend').innerHTML='<span><i class="key dot pale"></i>Training</span><span><i class="key dot warm"></i>Purged</span><span><i class="key dot accent"></i>Holdout</span>';
    $('secondary-caption').textContent=`Source: simulated vectors. First two components show ${pct(data.variance[0]+data.variance[1])} of training variance; all ${data.variance.length} retain ${pct(data.variance.reduce((a,b)=>a+b,0))}. This is not a semantic cluster map.`;
    renderTimeline();renderProjection();
  } else {
    $('primary-legend').innerHTML='<span><i class="key pale"></i>Training period</span><span><i class="key accent"></i>Holdout period</span>';
    $('secondary-eyebrow').textContent='SEPARATION & OVERLAP';$('secondary-title').textContent='Score distribution';$('secondary-tag').textContent='SYNTHETIC LABELS';
    $('secondary-legend').innerHTML='<span><i class="key pale"></i>Negative label</span><span><i class="key accent"></i>Positive label</span><span><i class="key warm"></i>Threshold</span>';
    $('secondary-caption').textContent='Held-out synthetic company-years, grouped into 0.1-wide score bins. Weighted scores are not calibrated probabilities.';
    renderAttribution();
  }
  modelKey=Object.keys(data.models)[0];configureModel();
  $('model').addEventListener('change',()=>{modelKey=$('model').value;configureModel();});
  $('threshold').addEventListener('input',()=>{page=0;render();});
  $('reset').addEventListener('click',()=>{$('errors-only').checked=false;configureModel();});
  $('errors-only').addEventListener('change',()=>{page=0;render();});
  $('previous').addEventListener('click',()=>{page--;render();});$('next').addEventListener('click',()=>{page++;render();});
  $('download').disabled=false;
  $('download').addEventListener('click',()=>{
    const report={...data, browser_inspection:{model:modelKey,threshold:Number($('threshold').value),counts:countsFor(data.models[modelKey].scores,data.truth,Number($('threshold').value)),note:'Synthetic demo. Threshold exploration is not a validated model change.'}};
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=`${c.repo}-synthetic-results.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  $('dashboard').setAttribute('aria-busy','false');
}
const dialog=$('data-dialog');
['provenance','method-data'].forEach(id=>$(id).addEventListener('click',()=>dialog.showModal()));
['close-dialog','dialog-done'].forEach(id=>$(id).addEventListener('click',()=>dialog.close()));
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
fetch(new URL('data.json',document.baseURI)).then(response=>{if(!response.ok)throw new Error(`Dataset request failed (${response.status})`);return response.json();}).then(result=>{data=result;initialize();}).catch(error=>{$('dashboard').hidden=true;$('load-error').hidden=false;$('load-error').textContent=`The demo dataset could not load. Refresh this page or check the repository's Pages deployment. ${error.message}`;console.error(error);});
