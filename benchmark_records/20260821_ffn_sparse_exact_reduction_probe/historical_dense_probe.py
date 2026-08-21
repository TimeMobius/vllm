import json
import statistics
import torch
import torch.nn.functional as F

LIB='/tmp/vllm-rwkv7-exact-build/_C.abi3.so'
torch.ops.load_library(LIB)
print('op',torch.ops._C.rwkv7_bf16_gemv_out)
print('schema',torch._C._dispatch_find_schema_or_throw('_C::rwkv7_bf16_gemv_out','').schema())
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.manual_seed(20260821)
shapes=[('attn',4096,4096),('ffn_up',16384,4096),('ffn_down',4096,16384),('lora_in_192',192,4096),('lora_out_192',4096,192)]
rows=[]
for name,n,k in shapes:
    x=torch.randn(1,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    out=torch.empty(1,n,device='cuda',dtype=torch.bfloat16)
    ref=F.linear(x,w)
    torch.ops._C.rwkv7_bf16_gemv_out(out,x,w)
    torch.cuda.synchronize()
    equal=torch.equal(out,ref)
    diff=(out.float()-ref.float()).abs()
    def b(fn, reps=50, warm=10):
       for _ in range(warm):fn()
       torch.cuda.synchronize(); e=[]
       for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); z.record(); z.synchronize();e.append(a.elapsed_time(z)*1000)
       return statistics.mean(e)
    baseline=b(lambda:F.linear(x,w))
    cand=b(lambda:torch.ops._C.rwkv7_bf16_gemv_out(out,x,w))
    rows.append({'name':name,'n':n,'k':k,'equal':equal,'different':int((out!=ref).sum()),'max_abs':float(diff.max()),'baseline_us':baseline,'candidate_us':cand,'speedup':baseline/cand})
print(json.dumps(rows,indent=2))
