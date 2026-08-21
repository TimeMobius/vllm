import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F

lib = next(Path('/tmp/rwkv7_dense_gemv_mapping_probe/build').glob('*.so'))
torch.ops.load_library(str(lib))
torch.backends.cuda.matmul.allow_tf32 = False

def run_case(name, out_features, in_features, make_x, seeds=range(4)):
    rows=[]
    for seed in seeds:
        g=torch.Generator(device='cuda').manual_seed(20260821 + seed*100003 + in_features)
        x=make_x(g, in_features)
        w=torch.randn(out_features,in_features,device='cuda',dtype=torch.bfloat16,generator=g)
        ref=F.linear(x,w)
        for variant in range(16):
            out=torch.empty_like(ref)
            torch.ops.rwkv7gemvprobe.bf16_gemv_variant_out(out,x,w,variant)
            torch.cuda.synchronize()
            different=int((out != ref).sum().item())
            rows.append({'case':name,'seed':seed,'variant':variant,'equal':different==0,'different':different,'max_abs':float((out.float()-ref.float()).abs().max())})
    return rows

normal=lambda g,k:torch.randn(1,k,device='cuda',dtype=torch.bfloat16,generator=g)
def sparse(g,k):
    x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16,generator=g)
    # Exactly emulate the relevant BF16 relu² materialization style.
    return torch.square(torch.relu(x))
rows=[]
rows += run_case('ffn_down_normal',4096,16384,normal)
rows += run_case('ffn_down_sparse_relu2',4096,16384,sparse)
rows += run_case('ffn_up_normal',16384,4096,normal,range(2))
summary={}
for case in sorted({r['case'] for r in rows}):
    summary[case]={}
    for v in range(16):
        rs=[r for r in rows if r['case']==case and r['variant']==v]
        summary[case][str(v)]={'all_equal':all(r['equal'] for r in rs),'differing':sum(r['different'] for r in rs),'max_abs':max(r['max_abs'] for r in rs)}
print(json.dumps(summary,indent=2))
Path('/tmp/rwkv7_dense_gemv_mapping_probe/results.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
