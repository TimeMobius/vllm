import statistics
import torch
import torch.nn.functional as F
import vllm._C

torch.ops.load_library('/tmp/rwkv7_gemv_fragment/build/rwkv7_gemv_fragment.so')
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.is_bf16_supported()

def bench(fn, *, warmup=100, iters=400, samples=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out=[]
    for _ in range(samples):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): fn()
        end.record(); end.synchronize()
        out.append(start.elapsed_time(end)*1000/iters)
    return statistics.median(out),out

for seed in range(4):
    torch.manual_seed(2026082100+seed)
    x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(16384,4096,device='cuda',dtype=torch.bfloat16)
    custom=torch.empty(1,16384,device='cuda',dtype=torch.bfloat16)
    ref=F.linear(x,w)
    torch.ops.rwkv7exp.bf16_gemv_out(custom,x,w)
    torch.cuda.synchronize()
    act_ref=torch.empty_like(ref)
    act_custom=torch.empty_like(custom)
    torch.ops._C.relu2(act_ref, ref)
    torch.ops._C.relu2(act_custom, custom)
    print({'seed':seed,'gemv_equal':torch.equal(custom,ref),'gemv_different':int((custom!=ref).sum()),'relu2_equal':torch.equal(act_custom,act_ref),'relu2_different':int((act_custom!=act_ref).sum())},flush=True)
    activation=torch.empty_like(ref)
    base_buffer=torch.empty_like(ref)
    def baseline():
        torch.ops._C.relu2(activation, F.linear(x,w))
    def native_path():
        torch.ops.rwkv7exp.bf16_gemv_out(custom,x,w)
        torch.ops._C.relu2(activation, custom)
    def relu_path():
        torch.ops._C.relu2(activation, ref)
    base,base_s=bench(baseline)
    native,native_s=bench(native_path)
    gemv,g_s=bench(lambda:torch.ops.rwkv7exp.bf16_gemv_out(custom,x,w))
    relu,r_s=bench(relu_path)
    print({'seed':seed,'base_us':base,'native_plus_relu_us':native,'native_gemv_us':gemv,'relu_us':relu,'speedup':base/native,'base_samples':base_s,'native_samples':native_s},flush=True)
