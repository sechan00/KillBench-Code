/* ---- mobile nav (shared by every page) ---- */
(function(){
  var btn=document.getElementById('navToggle'), menu=document.getElementById('navMenu');
  if(!btn||!menu) return;
  btn.addEventListener('click',function(){menu.classList.toggle('open');});
  menu.addEventListener('click',function(e){if(e.target.tagName==='A') menu.classList.remove('open');});
})();

/* ---- interactive result charts (results.html only) ---- */
(function(){
  if(!document.getElementById('mainChart') && !document.getElementById('benignChart')) return;

  var METHODS=["IPI","GCG","Warning","AutoGuard"];
  var COLOR={IPI:"#4285f4",GCG:"#9aa0a6",Warning:"#e08600",AutoGuard:"#0f9d58"};
  var DATA={
    agent:{groups:["C1 Text","C2 Sys-Bypass","C3 Multimodal","C4 Uncensored"],
      v:{IPI:[63.1,9.5,70.7,36.3],GCG:[3.8,2.8,2.7,0.0],Warning:[36.0,6.5,33.9,0.2],AutoGuard:[58.3,17.8,70.0,39.5]}},
    model:{groups:["Grok-4.3","GPT-5.2","Gemma4-31B","Qwen3.6-35B"],
      v:{IPI:[64.5,72.3,12.7,41.5],GCG:[4.6,5.2,0.5,2.1],Warning:[33.1,49.5,2.6,16.6],AutoGuard:[68.3,68.0,16.4,42.1]}},
    defense:{groups:["IPI","GCG","Warning","AutoGuard"],single:[46.9,2.9,23.5,48.0]}
  };
  var W=740,H=360,ML=46,MR=16,MT=18,MB=64;
  var pW=W-ML-MR, pH=H-MT-MB;
  function y(v){return MT+pH*(1-v/100);}
  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
  function axis(ylabel){
    var s='';
    for(var t=0;t<=100;t+=25){
      var yy=y(t);
      s+='<line x1="'+ML+'" y1="'+yy+'" x2="'+(W-MR)+'" y2="'+yy+'" stroke="#e5e7eb" stroke-width="1"/>';
      s+='<text x="'+(ML-8)+'" y="'+(yy+4)+'" text-anchor="end" font-size="11" fill="#6b7280">'+t+'</text>';
    }
    s+='<text transform="translate(14,'+(MT+pH/2)+') rotate(-90)" text-anchor="middle" font-size="11" fill="#6b7280">'+ylabel+'</text>';
    return s;
  }
  function render(view){
    var d=DATA[view], g=d.groups, n=g.length, gw=pW/n, s=axis("KSR_cond (%)");
    if(d.single){
      var bw=Math.min(72,gw*0.5);
      for(var i=0;i<n;i++){
        var cx=ML+gw*i+gw/2, val=d.single[i], col=COLOR[g[i]]||"#4285f4";
        s+='<rect x="'+(cx-bw/2)+'" y="'+y(val)+'" width="'+bw+'" height="'+(y(0)-y(val))+'" rx="4" fill="'+col+'"><title>'+esc(g[i])+': '+val+'%</title></rect>';
        s+='<text x="'+cx+'" y="'+(y(val)-7)+'" text-anchor="middle" font-size="13" font-weight="700" fill="#111418">'+val+'</text>';
        s+='<text x="'+cx+'" y="'+(H-MB+22)+'" text-anchor="middle" font-size="12" fill="#6b7280">'+esc(g[i])+'</text>';
      }
    } else {
      var m=METHODS.length, pad=gw*0.13, inner=gw-2*pad, bw=inner/m;
      for(var i=0;i<n;i++){
        var gx=ML+gw*i+pad;
        for(var j=0;j<m;j++){
          var meth=METHODS[j], val=d.v[meth][i], x=gx+bw*j;
          s+='<rect x="'+x+'" y="'+y(val)+'" width="'+(bw-2)+'" height="'+(y(0)-y(val))+'" rx="2.5" fill="'+COLOR[meth]+'"><title>'+esc(g[i])+' · '+meth+': '+val+'%</title></rect>';
          if(val>=14) s+='<text x="'+(x+(bw-2)/2)+'" y="'+(y(val)-4)+'" text-anchor="middle" font-size="9.5" fill="#6b7280">'+val+'</text>';
        }
        s+='<text x="'+(ML+gw*i+gw/2)+'" y="'+(H-MB+22)+'" text-anchor="middle" font-size="12" fill="#6b7280">'+esc(g[i])+'</text>';
      }
    }
    document.getElementById('mainChart').innerHTML=s;
    var leg=document.getElementById('chartLegend');
    if(d.single){leg.innerHTML='<span>Overall mean across every agent config, model, and scenario.</span>';}
    else{leg.innerHTML=METHODS.map(function(m){return '<span><i class="sw" style="background:'+COLOR[m]+'"></i>'+m+'</span>';}).join('');}
  }
  var tabs=document.querySelectorAll('.ctab');
  tabs.forEach(function(t){t.addEventListener('click',function(){
    tabs.forEach(function(x){x.classList.remove('active');}); t.classList.add('active'); render(t.dataset.view);
  });});
  if(document.getElementById('mainChart')) render('agent');

  /* benign collateral chart */
  var BEN=[{p:"IPI",BSR:0.8,KSR:72.0},{p:"Warning",BSR:1.6,KSR:48.4},{p:"GCG",BSR:48.4,KSR:4.9},{p:"AutoGuard",BSR:22.0,KSR:66.7}];
  function benign(){
    var n=BEN.length, gw=pW/n, s=axis("Rate (%)");
    var pad=gw*0.18, inner=gw-2*pad, bw=inner/2;
    for(var i=0;i<n;i++){
      var gx=ML+gw*i+pad, b=BEN[i];
      s+='<rect x="'+gx+'" y="'+y(b.BSR)+'" width="'+(bw-2)+'" height="'+(y(0)-y(b.BSR))+'" rx="3" fill="#4285f4"><title>'+b.p+' benign success (BSR): '+b.BSR+'%</title></rect>';
      s+='<text x="'+(gx+bw/2)+'" y="'+(y(b.BSR)-4)+'" text-anchor="middle" font-size="10" fill="#6b7280">'+b.BSR+'</text>';
      var x2=gx+bw;
      s+='<rect x="'+x2+'" y="'+y(b.KSR)+'" width="'+(bw-2)+'" height="'+(y(0)-y(b.KSR))+'" rx="3" fill="#0f9d58"><title>'+b.p+' malicious halt (KSR): '+b.KSR+'%</title></rect>';
      s+='<text x="'+(x2+bw/2)+'" y="'+(y(b.KSR)-4)+'" text-anchor="middle" font-size="10" fill="#6b7280">'+b.KSR+'</text>';
      s+='<text x="'+(ML+gw*i+gw/2)+'" y="'+(H-MB+22)+'" text-anchor="middle" font-size="12" fill="#6b7280">'+b.p+'</text>';
    }
    s+='<line x1="'+ML+'" y1="'+y(78.7)+'" x2="'+(W-MR)+'" y2="'+y(78.7)+'" stroke="#d93025" stroke-dasharray="5 4" stroke-width="1.5"/>';
    s+='<text x="'+(W-MR)+'" y="'+(y(78.7)-6)+'" text-anchor="end" font-size="10" fill="#d93025">clean benign success 78.7%</text>';
    document.getElementById('benignChart').innerHTML=s;
  }
  if(document.getElementById('benignChart')) benign();
})();
