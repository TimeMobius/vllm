import json
import os
import statistics
import time

import torch
import vllm._C  # Register relu2 CUDA custom op.
import torch.nn.functional as F

assert torch.cuda.is_available()
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device='cuda'
dtype=torch.bfloat16
x=torch.randn((1,4096),device=device,dtype=dtype)
w1=torch.randn((16384,4096),device=device,dtype=dtype)
w2=torch.randn((4096,16384),device=device,dtype=dtype)

# Current RWKV7 path: CUDA custom relu2 kernel between two eager F.linear calls.
def eager(x,w1,w2):
    h=F.linear(x,w1)
    out=torch.empty_like(h)
    torch.ops._C.relu2(out,h)
    return F.linear(out,w2)

# This is the only source-level compile form that could reduce FFN graph launch
# count without changing either GEMV. It must match the current custom relu2 path.
def candidate(x,w1,w2):
    return F.linear(torch.relu(F.linear(x,w1)).square(),w2)

compiled=torch.compile(candidate,fullgraph=True,dynamic=False)
ref=eager(x,w1,w2)
act=candidate(x,w1,w2)
# Compile separately, then make its first replay visible before timing.
got=compiled(x,w1,w2)
torch.cuda.synchronize()
print(json.dumps({
  'native_expression_equal_current_relu2': bool(torch.equal(act,ref)),
  'compiled_equal_current_relu2': bool(torch.equal(got,ref)),
  'max_abs_native': float((act.float()-ref.float()).abs().max()),
  'max_abs_compiled': float((got.float()-ref.float()).abs().max()),
},indent=2))

# no-grad, fixed operands; steady time.
def bench(fn,iters=50,warmup=10):
    for _ in range(warmup): fn(x,w1,w2)
    torch.cuda.synchronize()
    events=[]
    for _ in range(iters):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record(); fn(x,w1,w2); end.record(); end.synchronize()
        events.append(start.elapsed_time(end))
    return {'mean_ms':statistics.mean(events),'median_ms':statistics.median(events),'min_ms':min(events),'max_ms':max(events)}
print(json.dumps({'eager_current':bench(eager),'eager_native_expr':bench(candidate),'compiled_native_expr':bench(compiled)},indent=2))
