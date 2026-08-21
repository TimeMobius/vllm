import json, statistics, torch
import torch.nn.functional as F
LIB='/tmp/rwkv7_sparse_exact_fragment/build/rwkv7_sparse_exact_fragment.so'
torch.ops.load_library(LIB)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.manual_seed(20260821)
w=torch.randn(4096,16384,device='cuda',dtype=torch.bfloat16)

def bench(fn,reps=50,warm=10):
 for _ in range(warm):fn()
 torch.cuda.synchronize();ts=[]
 for _ in range(reps):
  a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();fn();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000)
 return statistics.mean(ts)
rows=[]
for zero_frac in [0.0,0.5,0.9,0.94,0.98]:
 x=torch.randn(1,16384,device='cuda',dtype=torch.bfloat16)
 if zero_frac:
  keep=torch.rand(1,16384,device='cuda')>=zero_frac
  x=x*keep.to(x.dtype)
 out=torch.empty(1,4096,device='cuda',dtype=torch.bfloat16)
 ref=F.linear(x,w)
 torch.ops.rwkv7sparseexp.bf16_gemv_d16384_out(out,x,w)
 torch.cuda.synchronize()
 d=(out.float()-ref.float()).abs()
 rows.append({'zero_frac':zero_frac,'actual_zero_frac':float((x==0).float().mean()),'equal':torch.equal(out,ref),'different':int((out!=ref).sum()),'max_abs':float(d.max()),'baseline_us':bench(lambda:F.linear(x,w)),'candidate_us':bench(lambda:torch.ops.rwkv7sparseexp.bf16_gemv_d16384_out(out,x,w))})
for r in rows:r['speedup']=r['baseline_us']/r['candidate_us']
print(json.dumps(rows,indent=2))
