# Raw eval logs

Logs live under the Brev instance (not committed — too large, may contain HF
paths). Copy or archive before tearing down the instance.

| Run | Model | Clips | Log path (absolute) |
|-----|-------|-------|---------------------|
| Smoke | bf16 | 10 | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/baseline_eval_v2.log` |
| Smoke | NVFP4 | 10 | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/nvfp4_eval_v6.log` |
| Primary | bf16 | 100 | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/baseline_eval_100.log` |
| Primary | NVFP4 | 100 | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/nvfp4_eval_100.log` |
| Quantization | NVFP4 PTQ | 100 calib | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/quantize_nvfp4.log` |
| Combined runner | both | 100 | `/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/eval_100_combined.log` |

Quantized checkpoint (7.6 GB, not in git):

`/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/outputs/alpamayo1.5_nvfp4_calib100/`
