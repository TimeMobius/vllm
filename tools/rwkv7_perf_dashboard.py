#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a self-contained RWKV7 performance history dashboard.

The input directory stores immutable JSON measurement artifacts. This command
extracts verified before/after comparison records into JSONL and embeds them in
a standalone HTML report, so a benchmark result can be inspected without a
separate web server.
"""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
from pathlib import Path
from typing import Any


def load_comparisons(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = set(records_dir.rglob("*_comparison.json"))
    paths.update(records_dir.rglob("comparison.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if {
            "algorithm_id",
            "baseline_avg_tps",
            "heuristic_avg_tps",
        } <= payload.keys():
            status = "verified"
            baseline_tps = payload["baseline_avg_tps"]
            tps = payload["heuristic_avg_tps"]
            speedup_percent = payload.get("speedup_percent")
            algorithm_id = payload["algorithm_id"]
            parent_id = payload.get("parent_id")
            accuracy_record = payload.get("accuracy_record")
            notes = payload.get("notes", "")
            prompt_tokens = payload.get("prompt_tokens")
            output_tokens = payload.get("output_tokens_per_round")
        elif payload.get("status") == "rejected":
            # Rejected candidates remain visible for auditability, but must not
            # be included in the verified performance trend line.
            baseline = payload.get("baseline", {})
            candidate = payload.get("candidate", {})
            experiments = payload.get("experiments", {})
            if experiments:
                baseline_tps = payload.get("reference_tps")
                candidate = experiments.get("piecewise", {})
                tps = candidate.get("avg_tps")
            else:
                baseline_tps = baseline.get("avg_tps")
                tps = candidate.get("avg_tps")
            if baseline_tps is None or tps is None:
                continue
            status = "rejected"
            speedup_percent = payload.get("speedup_percent")
            if speedup_percent is None:
                speedup_percent = (tps / baseline_tps - 1) * 100
            algorithm_id = payload.get("record_id", path.parent.name)
            parent_id = "eager-t1-reference"
            accuracy_record = payload.get("accuracy")
            notes = payload.get("conclusion", "")
            prompt_tokens = payload.get("workload", "see record")
            output_tokens = "see record"
        else:
            continue
        records.append(
            {
                "record_path": str(path.relative_to(records_dir)),
                "algorithm_id": algorithm_id,
                "parent_id": parent_id,
                "status": status,
                "baseline_tps": baseline_tps,
                "tps": tps,
                "speedup_percent": speedup_percent,
                "prompt_tokens": prompt_tokens,
                "output_tokens_per_round": output_tokens,
                "gpu": payload.get("gpu"),
                "accuracy_record": accuracy_record,
                "notes": notes,
            }
        )
    return records


def build_html(records: list[dict[str, Any]]) -> str:
    data = json.dumps(records, ensure_ascii=False)
    title = "RWKV7 vLLM-SP 性能演进"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #10131a; color: #ecf2ff; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
p {{ color: #aebbd0; }}
.card {{ background: #171c27; border: 1px solid #2b3548; border-radius: 12px; padding: 20px; margin-top: 18px; }}
#chart {{ width: 100%; min-height: 360px; display: block; background: #101722; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 10px; border-bottom: 1px solid #2b3548; text-align: left; vertical-align: top; }}
th {{ color: #93a8ca; font-weight: 600; }}
.up {{ color: #55e6a5; font-weight: 700; }} .bad {{ color: #ff8694; font-weight: 700; }}
code {{ color: #97d8ff; }} .muted {{ color: #93a8ca; font-size: 13px; }}
.empty {{ color: #ffcf70; }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<p>每一个数据点来自保留的 JSON 测试记录。折线图只包含通过精度门禁的保留优化；被拒绝候选以红色表格行保留，方便回溯。不同 GPU、模型或负载不得直接横向比较。</p>
<div class="card"><svg id="chart" viewBox="0 0 1000 380" role="img" aria-label="算法 TPS 折线图"></svg></div>
<div class="card"><h2>已记录的算法</h2><div id="table"></div></div>
<div class="card muted"><strong>数据位置：</strong><code>benchmark_records/performance_history.jsonl</code>。重新生成：<code>/mnt/data/anaconda3/envs/vllm-sp/bin/python tools/rwkv7_perf_dashboard.py</code></div>
</main>
<script>
const records = {data};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const chart = document.getElementById('chart');
const table = document.getElementById('table');
if (!records.length) {{
  chart.outerHTML = '<p class="empty">尚无 *_comparison.json 记录。</p>';
  table.innerHTML = '<p class="empty">请先完成基线与优化后的 A/B 测试。</p>';
}} else {{
  const verified = records.filter(r => r.status === 'verified');
  const points = verified.length ? [{{algorithm_id: 'baseline-current-flags', tps: verified[0].baseline_tps, uplift: 0, notes: 'comparison baseline'}}, ...verified.map(r => ({{algorithm_id:r.algorithm_id,tps:r.tps,uplift:r.speedup_percent,notes:r.notes}}))] : [];
  if (!points.length) {{
    chart.outerHTML = '<p class="empty">尚无通过精度门禁的 *_comparison.json 记录。</p>';
  }} else {{
  const W=1000,H=380,L=72,R=30,T=30,B=68;
  const min=Math.min(...points.map(p=>p.tps))*0.94, max=Math.max(...points.map(p=>p.tps))*1.06;
  const sx=i=>L+(W-L-R)*(points.length===1?.5:i/(points.length-1));
  const sy=v=>H-B-(v-min)/(max-min||1)*(H-T-B);
  let svg='';
  for(let i=0;i<5;i++){{ const v=min+(max-min)*i/4, y=sy(v); svg+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{y}}" y2="${{y}}" stroke="#29364c"/><text x="${{L-10}}" y="${{y+5}}" fill="#93a8ca" text-anchor="end" font-size="13">${{v.toFixed(1)}}</text>`; }}
  svg += `<path d="${{points.map((p,i)=>`${{i?'L':'M'}} ${{sx(i)}} ${{sy(p.tps)}}`).join(' ')}}" fill="none" stroke="#6ca8ff" stroke-width="3"/>`;
  points.forEach((p,i)=>{{const x=sx(i),y=sy(p.tps);svg+=`<circle cx="${{x}}" cy="${{y}}" r="6" fill="#55e6a5"/><text x="${{x}}" y="${{H-B+22}}" fill="#dce7fa" text-anchor="middle" font-size="12">${{esc(p.algorithm_id)}}</text><text x="${{x}}" y="${{y-12}}" fill="#55e6a5" text-anchor="middle" font-size="13">${{p.tps.toFixed(2)}} TPS</text>`;}});
  svg += `<text x="${{L}}" y="20" fill="#93a8ca" font-size="13">Aggregate TPS（固定负载）</text>`;
  chart.innerHTML=svg;
  }}
  table.innerHTML=`<table><thead><tr><th>状态 / 算法</th><th>TPS</th><th>相对基线</th><th>精度</th><th>负载</th><th>说明 / 原始记录</th></tr></thead><tbody>${{records.map(r=>`<tr><td class="${{r.status === 'verified' ? 'up':'bad'}}">${{r.status === 'verified' ? '已保留':'已回退'}}<br><code>${{esc(r.algorithm_id)}}</code><div class="muted">parent: ${{esc(r.parent_id)}}</div></td><td>${{r.tps.toFixed(3)}}</td><td class="${{r.speedup_percent >= 0 && r.status === 'verified' ? 'up':'bad'}}">${{r.speedup_percent >= 0 ? '+':''}}${{r.speedup_percent.toFixed(2)}}%</td><td><code>${{esc(typeof r.accuracy_record === 'string' ? r.accuracy_record : JSON.stringify(r.accuracy_record))}}</code></td><td>${{esc(r.prompt_tokens)}} × ${{esc(r.output_tokens_per_round)}}<br><span class="muted">${{esc(r.gpu)}}</span></td><td>${{esc(r.notes)}}<div class="muted"><code>${{esc(r.record_path)}}</code></div></td></tr>`).join('')}}</tbody></table>`;
}}
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=Path("benchmark_records"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records_dir = args.records_dir.resolve()
    output = args.output or records_dir / "dashboard.html"
    records = load_comparisons(records_dir)
    history_path = records_dir / "performance_history.jsonl"
    history_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    output.write_text(build_html(records), encoding="utf-8")
    print(f"wrote {history_path}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
