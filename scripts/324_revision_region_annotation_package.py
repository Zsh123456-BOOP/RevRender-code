#!/usr/bin/env python3
"""Build a one-file page for marking *where* a student's revision ink is.

Everything this project has built so far treats a revision as a label on a whole
line. That was enough to prove the negative result -- three independently built
synthetic sources (313/315, 314, 316) all sit near chance on real revisions --
but it cannot say what a real revision looks like, and the learned generator of
318 stalls at bACC 0.618 against 0.779 for real positives.

Looking at 100 real revised lines settled why. The taxonomy every public
resource uses -- single, double, diagonal, cross, wave, zig-zag, scratch, all
inherited from Heil et al. and reused unchanged by HWG and by IJDAR 2026 --
assumes a stroke drawn *across a word*. Chinese mathematics candidates mostly do
something else: they blot out the two or three characters that are wrong. Wave,
zig-zag, double and cross are essentially absent from our data. Every generator
we have tried draws a full-width line, which is why none of them transfers.

So the annotation asked for here is region, not text. Three tools, and the tool
*is* the type, so choosing a category costs no extra click:

    1  blot     drag a box round scribbled-out or blackened ink
    2  line     click twice, start and end of a struck-through segment
    3  overwrite  drag a box where new ink was written on top of old

A line may carry several marks; many do. Stroke width is never asked for -- it
is recoverable from the ink beneath the drawn segment, and asking would cost a
click per mark for something a computer measures better.

Two escapes exist because the sample is drawn from lines labelled as revised. A
blind audit of the policy-assigned positives found 34% of them wrong, and the
annotator's own earlier pass found 9.7% even among human-backed rows, so the
page must let someone say a line carries no revision at all, or cannot be read.
Without that the form would force an invented mark.

Sampling uses the same hash namespace as the 100-line type survey that motivated
this design, so those 100 lines are exactly the first 100 here: the work already
done reading them is not repeated.

This script deliberately does NOT:
  * run a model, use a GPU, reach the network, or call a paid API;
  * draw a policy-assigned label -- only human-backed positives;
  * draw from the frozen evaluation fold, the external blind set, or any FTV
    test split;
  * show the annotator any existing label;
  * overwrite anything -- the output directory is create-only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
MATCHED_SCRIPT = ROOT / "scripts" / "313_real_matched_synthetic_gap_control.py"
AUTHORITATIVE_MATCHED_SHA256 = (
    "5310a04355007549105c774a44d12165828b3ea517c98a98c962650b08f94f71"
)

# Identical to the namespace used for the 100-line type survey, so that sample
# is a prefix of this one rather than a second, overlapping draw.
SAMPLE_NAMESPACE = "hme-typesurvey-v1"
DEFAULT_SAMPLE = 300
SCHEMA_PACKAGE = "hme-revision-region-annotation-v1"

TOOLS = (
    {"key": "1", "id": "blot", "label": "涂黑/乱涂", "shape": "box", "colour": "#d62828",
     "hint": "把写错的字涂成一团"},
    {"key": "2", "id": "line", "label": "横线划掉", "shape": "line", "colour": "#0353a4",
     "hint": "一道笔迹横穿过去，点两下：起点、终点"},
    {"key": "3", "id": "overwrite", "label": "就地盖写", "shape": "box", "colour": "#2a9d8f",
     "hint": "新字直接写在原字上，没有单独划线"},
)


class RegionPackageError(RuntimeError):
    """Raised when a pinned input or guard refuses the run."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_recursive_sha256s(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{digest_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def atomic_staging(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RegionPackageError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_matched():
    if not MATCHED_SCRIPT.is_file():
        raise RegionPackageError(f"missing pinned experiment 313: {MATCHED_SCRIPT}")
    if digest_file(MATCHED_SCRIPT) != AUTHORITATIVE_MATCHED_SHA256:
        raise RegionPackageError("experiment 313 digest mismatch")
    spec = importlib.util.spec_from_file_location("hme_matched313_for324", MATCHED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_key(line_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_NAMESPACE}\x1f{line_id}".encode("utf-8")).hexdigest()


def build_page(items: list[dict], fingerprint: str) -> str:
    """One self-contained file: images inlined, state kept by the page itself."""
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    tools = json.dumps(list(TOOLS), ensure_ascii=False)
    return _TEMPLATE.replace("__ITEMS__", payload).replace("__TOOLS__", tools).replace(
        "__FINGERPRINT__", fingerprint
    ).replace("__COUNT__", str(len(items)))


_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>涂改区域标注</title>
<style>
:root{--bg:#f7f7f8;--ink:#1a1a1a;--line:#d9d9de;--ok:#2a9d8f;--warn:#c1121f}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
     background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line);
       padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);background:#fff;
       border-radius:7px;cursor:pointer}
button:hover{background:#f0f0f3}
button.primary{background:var(--ok);color:#fff;border-color:transparent}
button.tool{border-width:2px}
button.tool.active{color:#fff}
main{max-width:1180px;margin:16px auto;padding:0 16px 90px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.stage{overflow:auto;background:#fafafa;border:1px solid var(--line);border-radius:8px;padding:10px}
canvas{display:block;cursor:crosshair;image-rendering:pixelated}
.marks{margin-top:10px;display:flex;flex-direction:column;gap:6px}
.mark{display:flex;gap:10px;align-items:center;padding:5px 8px;border:1px solid var(--line);
      border-radius:7px;cursor:pointer}
.mark.sel{border-color:#000;box-shadow:0 0 0 2px rgba(0,0,0,.08)}
.swatch{width:13px;height:13px;border-radius:3px;flex:none}
.grow{flex:1}
.muted{color:#666;font-size:13px}
kbd{background:#eee;border:1px solid #ccc;border-bottom-width:2px;border-radius:4px;
    padding:0 5px;font-family:ui-monospace,monospace;font-size:12px}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid var(--line);
       padding:7px 16px;font-size:13px;display:flex;gap:16px;align-items:center}
.pill{padding:2px 9px;border-radius:999px;background:#eef;font-size:12px}
.bad{background:#fde;color:var(--warn)}
.good{background:#e6f6f2;color:var(--ok)}
table.help{border-collapse:collapse;font-size:13px}
table.help td{padding:2px 10px 2px 0}
</style></head><body>
<header>
  <strong>涂改区域标注</strong>
  <span id="pos" class="pill"></span>
  <span class="grow"></span>
  <button id="pick" class="primary">选择保存文件</button>
  <button id="imp">导入续标</button>
  <button id="exp">导出 JSON</button>
</header>
<main>
  <div class="card">
    <div id="tools" style="display:flex;gap:8px;flex-wrap:wrap"></div>
    <div class="muted" id="toolhint" style="margin-top:7px"></div>
  </div>
  <div class="card">
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
      <button id="zin">＋</button><button id="zout">－</button>
      <span class="muted" id="zlabel"></span>
      <span class="grow"></span>
      <button id="none">本行其实没涂改 <kbd>0</kbd></button>
      <button id="skip">读不出，跳过 <kbd>S</kbd></button>
    </div>
    <div class="stage"><canvas id="cv"></canvas></div>
    <div class="marks" id="marks"></div>
  </div>
  <div class="card">
    <table class="help">
      <tr><td><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd></td><td>切换工具（选一次可连续标）</td>
          <td><kbd>Q</kbd><kbd>E</kbd></td><td>上一张 / 下一张</td></tr>
      <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd></td><td>撤销上一处</td>
          <td><kbd>X</kbd></td><td>删除选中的那一处</td></tr>
      <tr><td><kbd>0</kbd></td><td>本行其实没涂改</td>
          <td><kbd>S</kbd></td><td>读不出，跳过</td></tr>
      <tr><td>滚轮 / <kbd>+</kbd><kbd>-</kbd></td><td>缩放</td>
          <td colspan="2" class="muted">涂黑/盖写＝拖框；划线＝点两下（起点、终点）</td></tr>
    </table>
  </div>
</main>
<footer><span id="save" class="pill">未选择保存文件</span><span id="prog" class="muted"></span></footer>
<script>
const ITEMS=__ITEMS__, TOOLS=__TOOLS__, FINGERPRINT="__FINGERPRINT__", TOTAL=__COUNT__;
const st={i:0,tool:TOOLS[0],zoom:2,rec:{},sel:-1,pending:null,fh:null,mode:"内存"};
const $=s=>document.querySelector(s), cv=$("#cv"), ctx=cv.getContext("2d");
const img=new Image(); let ready=false;

function rec(){ const k=ITEMS[st.i].id; return st.rec[k] ||= {marks:[],flag:null,t:null}; }
function save(){
  const blob=JSON.stringify({schema_version:"hme-revision-region-annotation-v1",
    page_fingerprint:FINGERPRINT,exported_at:new Date().toISOString(),
    storage_mode:st.mode,records:st.rec},null,1);
  if(st.fh){ st.fh.createWritable().then(w=>w.write(blob).then(()=>w.close())).catch(()=>{}); }
  else { try{localStorage.setItem("hme-region-annot",blob); st.mode="localStorage";}catch(e){} }
  paint_status();
}
function paint_status(){
  const done=Object.values(st.rec).filter(r=>r.flag||r.marks.length).length;
  $("#prog").textContent=`已完成 ${done} / ${TOTAL}`;
  const s=$("#save"); s.textContent = st.fh?"✓ 正在实时写入你选的文件":"⚠ "+st.mode+"（建议点「选择保存文件」）";
  s.className="pill "+(st.fh?"good":"bad");
}
function paint_tools(){
  $("#tools").innerHTML="";
  TOOLS.forEach(t=>{ const b=document.createElement("button");
    b.className="tool"+(t.id===st.tool.id?" active":""); b.style.borderColor=t.colour;
    if(t.id===st.tool.id) b.style.background=t.colour;
    b.innerHTML=`<kbd>${t.key}</kbd> ${t.label}`;
    b.onclick=()=>{st.tool=t; st.pending=null; paint_tools(); draw();};
    $("#tools").appendChild(b); });
  $("#toolhint").textContent=st.tool.hint+(st.tool.shape==="line"?"":"（拖动画框）");
}
function draw(){
  if(!ready) return;
  const z=st.zoom; cv.width=img.width*z; cv.height=img.height*z;
  ctx.imageSmoothingEnabled=false; ctx.drawImage(img,0,0,cv.width,cv.height);
  const r=rec();
  r.marks.forEach((m,idx)=>{
    const t=TOOLS.find(x=>x.id===m.type)||TOOLS[0];
    ctx.lineWidth=(idx===st.sel?3:2); ctx.strokeStyle=t.colour;
    if(m.shape==="box"){ ctx.strokeRect(m.x*z,m.y*z,m.w*z,m.h*z);
      ctx.fillStyle=t.colour+"22"; ctx.fillRect(m.x*z,m.y*z,m.w*z,m.h*z); }
    else { ctx.beginPath(); ctx.moveTo(m.x1*z,m.y1*z); ctx.lineTo(m.x2*z,m.y2*z); ctx.stroke(); }
  });
  if(st.pending){ ctx.strokeStyle="#999"; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.arc(st.pending.x*z,st.pending.y*z,4,0,7); ctx.stroke(); ctx.setLineDash([]); }
  $("#zlabel").textContent=`${z}×  ${img.width}×${img.height}px`;
  paint_marks();
}
function paint_marks(){
  const r=rec(), box=$("#marks"); box.innerHTML="";
  if(r.flag){ const d=document.createElement("div"); d.className="mark";
    d.innerHTML=`<span class="grow"><b>${r.flag==="no_revision"?"本行其实没涂改":"读不出，跳过"}</b></span>`;
    const b=document.createElement("button"); b.textContent="取消"; b.onclick=()=>{r.flag=null;touch();};
    d.appendChild(b); box.appendChild(d); return; }
  r.marks.forEach((m,idx)=>{
    const t=TOOLS.find(x=>x.id===m.type)||TOOLS[0], d=document.createElement("div");
    d.className="mark"+(idx===st.sel?" sel":"");
    d.innerHTML=`<span class="swatch" style="background:${t.colour}"></span>
      <span class="grow">${idx+1}. ${t.label} <span class="muted">${
        m.shape==="box"?`${Math.round(m.w)}×${Math.round(m.h)}`:`长 ${Math.round(Math.hypot(m.x2-m.x1,m.y2-m.y1))}`}</span></span>`;
    d.onclick=()=>{st.sel=idx;draw();};
    const b=document.createElement("button"); b.textContent="删除";
    b.onclick=e=>{e.stopPropagation(); r.marks.splice(idx,1); st.sel=-1; touch();};
    d.appendChild(b); box.appendChild(d);
  });
  if(!r.marks.length) box.innerHTML='<span class="muted">这一行还没有标记。</span>';
}
function touch(){ rec().t=new Date().toISOString(); save(); draw(); }
function load(){
  const it=ITEMS[st.i]; ready=false; img.src=it.png;
  $("#pos").textContent=`第 ${st.i+1} / ${TOTAL} 张`; st.sel=-1; st.pending=null;
}
img.onload=()=>{ready=true;draw();};
function go(d){ st.i=Math.max(0,Math.min(TOTAL-1,st.i+d)); load(); }

let drag=null;
cv.onmousedown=e=>{ const z=st.zoom,b=cv.getBoundingClientRect();
  const x=(e.clientX-b.left)/z, y=(e.clientY-b.top)/z; const r=rec(); if(r.flag) return;
  if(st.tool.shape==="line"){
    if(!st.pending){ st.pending={x,y}; draw(); }
    else { r.marks.push({type:st.tool.id,shape:"line",x1:st.pending.x,y1:st.pending.y,x2:x,y2:y});
           st.pending=null; touch(); }
  } else drag={x,y};
};
cv.onmousemove=e=>{ if(!drag) return; const z=st.zoom,b=cv.getBoundingClientRect();
  const x=(e.clientX-b.left)/z,y=(e.clientY-b.top)/z; draw();
  ctx.strokeStyle=st.tool.colour; ctx.setLineDash([4,3]);
  ctx.strokeRect(Math.min(drag.x,x)*z,Math.min(drag.y,y)*z,Math.abs(x-drag.x)*z,Math.abs(y-drag.y)*z);
  ctx.setLineDash([]); };
cv.onmouseup=e=>{ if(!drag) return; const z=st.zoom,b=cv.getBoundingClientRect();
  const x=(e.clientX-b.left)/z,y=(e.clientY-b.top)/z;
  const w=Math.abs(x-drag.x),h=Math.abs(y-drag.y);
  if(w>2&&h>2) rec().marks.push({type:st.tool.id,shape:"box",
      x:Math.min(drag.x,x),y:Math.min(drag.y,y),w,h});
  drag=null; touch(); };
cv.onwheel=e=>{ e.preventDefault(); st.zoom=Math.max(1,Math.min(8,st.zoom+(e.deltaY<0?1:-1))); draw(); };

addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT") return;
  const k=e.key.toLowerCase();
  if(e.ctrlKey||e.metaKey){ if(k==="z"){ const r=rec(); r.marks.pop(); st.sel=-1; touch(); e.preventDefault(); } return; }
  if(k==="q"){go(-1);} else if(k==="e"){go(1);}
  else if(k==="1"||k==="2"||k==="3"){ st.tool=TOOLS[+k-1]; st.pending=null; paint_tools(); draw(); }
  else if(k==="0"){ const r=rec(); r.flag=r.flag==="no_revision"?null:"no_revision"; r.marks=[]; touch(); }
  else if(k==="s"){ const r=rec(); r.flag=r.flag==="illegible"?null:"illegible"; r.marks=[]; touch(); }
  else if(k==="x"){ const r=rec(); if(st.sel>=0){ r.marks.splice(st.sel,1); st.sel=-1; touch(); } }
  else if(k==="+"||k==="="){ st.zoom=Math.min(8,st.zoom+1); draw(); }
  else if(k==="-"){ st.zoom=Math.max(1,st.zoom-1); draw(); }
  else return;
  e.preventDefault();
});
$("#zin").onclick=()=>{st.zoom=Math.min(8,st.zoom+1);draw();};
$("#zout").onclick=()=>{st.zoom=Math.max(1,st.zoom-1);draw();};
$("#none").onclick=()=>{const r=rec(); r.flag="no_revision"; r.marks=[]; touch();};
$("#skip").onclick=()=>{const r=rec(); r.flag="illegible"; r.marks=[]; touch();};
$("#pick").onclick=async()=>{
  if(!window.showSaveFilePicker){ alert("这个浏览器不支持自动写盘，请用 Chrome 或 Edge。仍可用「导出 JSON」。"); return; }
  try{ st.fh=await showSaveFilePicker({suggestedName:"revision_region_annotation.json",
        types:[{description:"JSON",accept:{"application/json":[".json"]}}]});
       st.mode="文件"; save(); }catch(e){}
};
$("#exp").onclick=()=>{
  const blob=new Blob([JSON.stringify({schema_version:"hme-revision-region-annotation-v1",
    page_fingerprint:FINGERPRINT,exported_at:new Date().toISOString(),records:st.rec},null,1)],
    {type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="revision_region_annotation.json"; a.click();
};
$("#imp").onclick=()=>{
  const i=document.createElement("input"); i.type="file"; i.accept=".json";
  i.onchange=()=>{ const f=i.files[0]; if(!f) return; const r=new FileReader();
    r.onload=()=>{ try{ const d=JSON.parse(r.result);
        if(d.page_fingerprint&&d.page_fingerprint!==FINGERPRINT&&
           !confirm("这份文件的指纹和本页不一致，仍要导入吗？")) return;
        st.rec=d.records||{}; load(); paint_status(); }catch(e){alert("读不出这个文件");} };
    r.readAsText(f); };
  i.click();
};
try{ const cached=localStorage.getItem("hme-region-annot");
     if(cached){ const d=JSON.parse(cached); if(d.page_fingerprint===FINGERPRINT) st.rec=d.records||{}; } }catch(e){}
paint_tools(); load(); paint_status();
</script></body></html>
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    output_dir = Path(args.output_dir).resolve()
    matched = load_matched()
    gap = matched.load_gap()
    base = gap.load_base()
    images_dir = Path(args.images_dir)
    try:
        train_rows, eval_rows = base._validate_training_inputs(Path(args.input_root), images_dir)
    except base.StrongVisual250Error as exc:
        raise RegionPackageError(f"pinned fold validation failed: {exc}") from exc

    eval_writers = {str(row["group_uuid"]) for row in eval_rows}
    pool = [row for row in train_rows if row["has_edit"]]
    if any(str(row["group_uuid"]) in eval_writers for row in pool):
        raise RegionPackageError("the pool touches an evaluation writer")
    if not pool:
        raise RegionPackageError("no human-backed revised lines in the pool")
    picked = sorted(pool, key=lambda row: sample_key(str(row["line_id"])))[: args.sample]
    if len(picked) != args.sample:
        raise RegionPackageError(f"pool holds {len(pool)} rows, cannot draw {args.sample}")

    pages: dict[str, Any] = {}
    items, key_rows = [], []
    for index, row in enumerate(picked):
        image_id = str(row["image_id"])
        if image_id not in pages:
            pages.clear()
            pages[image_id] = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
        crop = base.crop_line(pages[image_id], row["box_1000"])
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        item_id = hashlib.sha256(f"{SAMPLE_NAMESPACE}\x1f{row['line_id']}".encode()).hexdigest()[:16]
        items.append({"id": item_id, "png": f"data:image/png;base64,{encoded}",
                      "w": crop.width, "h": crop.height})
        key_rows.append({
            "group_uuid": str(row["group_uuid"]),
            "id": item_id,
            "image_id": image_id,
            "index": index,
            "line_id": str(row["line_id"]),
            "question_global_id": str(row["question_global_id"]),
        })
    for opened in pages.values():
        opened.close()

    fingerprint = hashlib.sha256(
        "\x1f".join(item["id"] for item in items).encode("utf-8")
    ).hexdigest()[:32]
    page = build_page(items, fingerprint)

    staging = atomic_staging(output_dir)
    (staging / "标注页.html").write_text(page, encoding="utf-8")
    write_jsonl(staging / "key.jsonl", key_rows)
    manifest = {
        "create_only": True,
        "escapes": ["no_revision", "illegible"],
        "evaluation_writers_touched": False,
        "external_blind_set_touched": False,
        "labels_shown_to_annotator": False,
        "network_used": False,
        "page_fingerprint": fingerprint,
        "page_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
        "paid_api_calls": 0,
        "pinned_experiment_313_sha256": AUTHORITATIVE_MATCHED_SHA256,
        "pool_size": len(pool),
        "sample_namespace": SAMPLE_NAMESPACE,
        "sampled": len(items),
        "schema_version": SCHEMA_PACKAGE,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "source": "human-backed revised lines of the pinned fold-0 train split",
        "tools": list(TOOLS),
    }
    write_json(staging / "package_manifest.json", manifest)
    write_json(staging / "complete.json", {"schema_version": SCHEMA_PACKAGE, "status": "complete"})
    write_recursive_sha256s(staging)
    staging.rename(output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = run(args)
    except RegionPackageError as exc:
        raise SystemExit(f"experiment 324 refused: {exc}") from exc
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "page_fingerprint": manifest["page_fingerprint"],
        "pool_size": manifest["pool_size"],
        "sampled": manifest["sampled"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
