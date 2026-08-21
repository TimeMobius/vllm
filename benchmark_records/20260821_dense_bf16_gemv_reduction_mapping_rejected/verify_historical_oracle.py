import json
import torch
import torch.nn.functional as F

torch.ops.load_library('/tmp/rwkv7_gemv_fragment/build/rwkv7_gemv_fragment.so')
rows=[]
for name, out_features, in_features, sparse in [
    ('ffn_down_normal',4096,16384,False),
    ('ffn_down_sparse_relu2',4096,16384,True),
    ('ffn_up_normal',16384,4096,False),
]:
    for seed in range(4):
        g=torch.Generator(device='cuda').manual_seed(20260821 + seed*100003 + in_features)
        x=torch.randn(1,in_features,device='cuda',dtype=torch.bfloat16,generator=g)
        if sparse: x=torch.square(torch.relu(x))
        w=torch.randn(out_features,in_features,device='cuda',dtype=torch.bfloat16,generator=g)
        ref=F.linear(x,w)
        out=torch.empty_like(ref)
        torch.ops.rwkv7exp.bf16_gemv_out(out,x,w)
        torch.cuda.synchronize()
        diff=int((out != ref).sum().item())
        rows.append({'case':name,'seed':seed,'equal':diff==0,'different':diff,'max_abs':float((out.float()-ref.float()).abs().max())})
print(json.dumps(rows,indent=2))
if not all(x['equal'] for x in rows): raise SystemExit(1)
