"""Build a local, paginated photo viewer from a completed no_hand.csv."""
import csv
import json
from pathlib import Path


def build(csv_path):
    by_letter = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_letter.setdefault(row["letter"], []).append(Path(row["original_path"]).as_uri())
    data = json.dumps(by_letter).replace("<", "\\u003c")
    html = """<!doctype html><html lang="nl"><meta charset="utf-8">
<title>Foto’s zonder handdetectie</title>
<style>
body{max-width:1120px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#183034}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
img{display:block;width:100%;height:auto} a{color:#17685e} p,nav{margin-bottom:1.2rem}
select,button{padding:.4rem;font:inherit} #count{margin:0 .6rem}
</style><h1>Foto’s zonder handdetectie</h1>
<p>MediaPipe vond op deze originele trainingsfoto’s nul handen. Klik op een foto voor het origineel.</p>
<nav><label>Letter <select id="letter"></select></label>
<button id="prev">Vorige</button> <button id="next">Volgende</button>
<span id="count" aria-live="polite"></span></nav><main id="photos"></main>
<script>
const groups = DATA;
const letter = document.querySelector('#letter'), photos = document.querySelector('#photos');
let start = 0;
for (const key of Object.keys(groups).sort((a,b)=>groups[b].length-groups[a].length)) {
  letter.add(new Option(key+' ('+groups[key].length+')',key));
}
function render() {
  const urls = groups[letter.value] || [];
  photos.replaceChildren();
  for (const url of urls.slice(start,start+24)) {
    const link = document.createElement('a'), img = document.createElement('img');
    const name = decodeURIComponent(url.split('/').pop());
    img.src = url; img.alt = name; link.href = url; link.target = '_blank';
    link.append(img,name); photos.append(link);
  }
  document.querySelector('#count').textContent = urls.length ?
    (start+1)+'–'+Math.min(start+24,urls.length)+' van '+urls.length : 'Geen foto’s';
  document.querySelector('#prev').disabled = start===0;
  document.querySelector('#next').disabled = start+24>=urls.length;
}
letter.onchange = ()=>{start=0; render()};
document.querySelector('#prev').onclick = ()=>{start-=24; render()};
document.querySelector('#next').onclick = ()=>{start+=24; render()};
render();</script></html>"""
    csv_path.with_name("bekijk_fotos.html").write_text(html.replace("DATA", data), encoding="utf-8")
