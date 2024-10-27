import ebcandle

print(f"mkl:         {ebcandle.utils.has_mkl()}")
print(f"accelerate:  {ebcandle.utils.has_accelerate()}")
print(f"num-threads: {ebcandle.utils.get_num_threads()}")
print(f"cuda:        {ebcandle.utils.cuda_is_available()}")

t = ebcandle.Tensor(42.0)
print(t)
print(t.shape, t.rank, t.device)
print(t + t)

t = ebcandle.Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
print(t)
print(t + t)

t = t.reshape([2, 4])
print(t.matmul(t.t()))

print(t.to_dtype(ebcandle.u8))
print(t.to_dtype("u8"))

t = ebcandle.randn((5, 3))
print(t)
print(t.dtype)

t = ebcandle.randn((16, 256))
quant_t = t.quantize("q6k")
dequant_t = quant_t.dequantize()
diff2 = (t - dequant_t).sqr()
print(diff2.mean_all())
