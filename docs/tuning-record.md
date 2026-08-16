# Qwen3.8-27B 部署与调优全记录

> 环境：云上 QEMU VM（4×GPU 直通）· 4×RTX 4090 24GB VM · SGLang 0.5.17 · 2026-08-15/16
> 最终性能（**2026-08-16 EAGLE 修复后**）：**短上下文 ~142 tps / 50k 长上下文 decode ~107-148 tps / 100k+ 长上下文正常 / prefill ~4-4.6k tokens/s / 256K 上下文**

---

## 1. 环境与架构

| 项 | 值 |
|---|---|
| 主机 | 云上 QEMU VM（GPU 直通，无 P2P） |
| 虚拟化 | QEMU VM（GPU 直通，**无 P2P**） |
| GPU | 4× RTX 4090 24GB（SM89 / Ada） |
| 驱动 / CUDA | 580.173.02 / CUDA 13.2 |
| 内存 / 磁盘 | 32GB / 95GB（可用 ~35GB） |
| 模型 | `Qwen3.8-27B-FP8`（官方 FP8 版，29GB，`/root/models/Qwen3.8-27B-FP8`，safetensors crc32 全部通过） |
| 引擎 | SGLang 0.5.17（`/opt/sgenv`，Python 3.12，阿里云镜像安装） |
| 模型要点 | 27B 稠密、多模态、混合注意力（**48 层 Gated DeltaNet 线性注意力 + 16 层全注意力**，每 4 层一个）、原生 256K（KV cache 仅 64KB/token，FP8 后 32KB）、内置 MTP 草稿头 |

**架构影响**：混合注意力使 KV cache 极小（256K 只需 ~16GB/4 卡），但线性注意力状态需按 32-token 检查点缓存；GDN 层的加速后端（flashinfer/cutedsl）均要求 SM90+（TMA 指令），SM89 只能用 Triton。

---

## 2. 部署步骤与踩坑记录

### 2.1 权重
- 用户已在 ModelScope 下载 FP8 版到 `/root/models/Qwen3.8-27B-FP8`；
- 全量 crc32 校验：**所有 safetensors 通过**（仅 3 个小 JSON 与清单不一致，为用户编辑所致，无害）。

### 2.2 环境
```bash
python3 -m venv /opt/sgenv          # 或由用户预置
pip install sglang[all] -i https://mirrors.aliyun.com/pypi/simple/
```

### 2.3 启动失败修复链（关键！）
| # | 报错 | 根因 | 修复 |
|---|---|---|---|
| 1 | `flashinfer JIT: CUDA compiler and CUDA toolkit headers are incompatible` | pip `cuda-toolkit` 混装：nvcc 13.3.73 + 头文件 13.0，cccl 版本检查失败 | 给 `flashinfer/data/cccl/.../cuda_toolkit.h` 加 `#define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 1`（同大版本内安全） |
| 2 | `FileNotFoundError: 'ninja'` | flashinfer JIT 需要 ninja | `pip install ninja` + `export PATH=/opt/sgenv/bin:$PATH` |
| 3 | `/usr/bin/ld: cannot find -lcudart` | pip 工具包无链接库 | `ln -s <pip>/nvidia/cu13/lib/libcudart.so.13 /usr/local/cuda/lib64/libcudart.so` + libcuda 符号链接 |
| 4 | `fastsafetensors: no GPU was found` | fastsafetensors 0.3.3 与 CUDA 13 探测失败 | 改用 `--load-format safetensors`（31GB 内存逐 shard 加载可行） |
| 5 | `peer access is not supported`（VM 无 P2P） | sglang 多模态 CUDA IPC 池需要 P2P | `--mm-feature-transport cpu` |
| 6 | `unrecognized arguments: --max-model-len` | sglang 0.5.17 参数改名 | 用 `--context-length 262144` |

### 2.4 服务化
- 启动脚本 `/root/run_sglang.sh` + systemd 单元 `sglang-qwen38.service`（`Restart=on-failure`）；
- 日志走 journald（`journalctl -u sglang-qwen38`）。

---

## 3. 性能调优历程（decode tps 演进）

| 阶段 | 短上下文 | 50k 长上下文 | 手段 |
|---|---|---|---|
| 基线 | 57 | — | 默认配置 |
| W8A8 GEMM 调优 | **73** | — | 为 4090 生成 5 个 `(N,K)` 的 tile 配置（借同架构 L40 参数，`sglang/kernels/ops/quantization/configs/`） |
| +EAGLE（MTP） | **110** | 12.8 ❌（后查明为统计假象，见 §6） | 投机解码 |
| 关 EAGLE | 74 | 53.9 | 长上下文回归 |
| +flashinfer full-attn | 74 | **70.3** ✅ | 16 层全注意力从 Triton 换 CUDA kernel（EAGLE 关闭后可行），长上下文 +30% |
| +bf16-ssm 状态 | 75.2 | 75.3 | `--mamba-ssm-dtype bfloat16`：mamba 状态 fp32→bf16（**非权重量化**），并发上限 33→65，质量门禁与基线等价 |
| **EAGLE 修复（最终）** | **~142** | **~107-148** ✅ | EAGLE(3/1/4) + `mem-fraction 0.80` + `--disable-prefill-cuda-graph`（修 OOM，见 §6）；50k 实测 accept len 2.7-3.05、needle ✓ |

### 无效/不可行的尝试
- `--enable-gdn-replayssm-spec`：无效（长上下文仍 12.9）
- `--num-continuous-decode-steps 4`：无效
- NCCL 参数（LL128/BUFFSIZE/通道数/线程数）：无效（已是 LL 协议）
- `NCCL_PROTO=LL NCCL_ALGO=TREE`：**启动崩溃**（`NCCL error: invalid usage`，graph capture 期），NCCL 层调优放弃
- `--enable-quant-communications`：源码确认仅 prefill 生效
- `--enable-flashinfer-allreduce-fusion`：崩溃（flashinfer 0.6.15.post1 版本限制）
- `--enable-torch-compile`：**启动崩溃**（inductor 包装 `_causal_conv1d_update_kernel` 时 `launcher() missing _grid_2` 参数，0.5.17 + 当前 torch 版本的兼容性 bug）
- prefill CUDA graph：显存余量不足（mamba 状态缓存占 5.3GB/卡），且收益仅 ~2-3%；EAGLE 下必须关闭（见 §6 OOM 修复）
- prefill chunk 8192→16384：**无增益**（50k TTFT 14.3s↔14.3s，100k 32s↔32s，prefill 是 GEMM 算力 bound ~4-4.6k t/s ≈ FP8 理论值的 70%）；→32768 **OOM**
- GDN 层 flashinfer / cutedsl 后端：**SM90+ TMA 硬限制**（CuTe-DSL 实现），SM89 不可用
- GDN Triton kernel `num_warps=2`：更慢（68 vs 74），`num_warps=1` 已最优
- **自研 GDN 融合 kernel：不值得写**——0.5.17 已内置 `fused_recurrent_gated_delta_rule_packed_decode`（L2norm+softplus gating+delta rule 单 kernel），decode profile 显示 GDN 全链路（packed_decode 3µs + conv1d 1.8µs + gated norm）**仅占 decode ~2%**，瓶颈在 NCCL all-reduce（21%）
- SHM 自研 all-reduce 替代 NCCL：VM 无 P2P，需经主机内存中转（D2H+H2D ≈ 15-20µs vs NCCL LL 23µs），收益甚微风险高，未做
- 长上下文 EAGLE/MTP：vLLM 不行；sglang 侧"崩溃"为假象（见 §6）

---

## 4. CUDA 底层分析（torch profiler）

### 工具链折腾
- nsys 2024.2：CUPTI 与 CUDA 13 不兼容（只有 API 无 kernel）；
- nsys 2026.1.3：kernel 可采，但 **TP rank 是 spawn 子进程，注入失败**（`--trace-fork-before-exec=true` 无效）；
- **最终方案：sglang 内建 torch.profiler**——`POST /start_profile`（`{"output_dir": ..., "with_stack": true}`）→ 跑负载 → `POST /stop_profile`，产出 4 个 TP rank 的 chrome trace（需带 API key）。

### kernel 时间分布（decode 段）
| Kernel | 占比 | 说明 |
|---|---|---|
| `_w8a8_block_fp8_matmul` | 62% | FP8 GEMM（Triton），带宽瓶颈主体 |
| `ncclDevKernel_AllReduce` | 17-19% | 跨卡通信，VM 无 P2P 结构性开销 |
| GDN Triton（gemvx/stage1/recurrent） | 10-25% | SM89 无 CUDA 替代实现 |
| quant/RMSNorm/采样等 | ~8% | 小 kernel |

**关键结论**：kernel 执行期带宽利用率 ~80%（不错）；decode 段 kernel 背靠背（间隙仅 3%）；综合有效带宽 ~55%——瓶颈是通信（19%）而非调度间隙。gemvx（7.9%）实为 cuBLASLt 的通用 GEMV（`libcublasLt.so`）。

---

## 5. 推理格式与 API 问题

### 5.1 thinking 格式
- **现象**：`content` 混入思考文本、`</think>` 悬空、`reasoning_content: null`；
- **根因**：① Qwen3.8 模板把 `<think>` 注入 prompt（模型原始输出无开标签）；② sglang auto 检测的 parser 不传导到响应层（`server_args.reasoning_parser=None`）；
- **修复**：显式 `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`；
- 修复后：`content`=答案、`reasoning_content`=思考、`reasoning_tokens` 正确统计（OpenAI 兼容）。

### 5.2 缓存命中统计
- radix cache 命中正常（重复 50k 前缀命中 99.9%，日志 `#cached-token: 50560`）；
- **短请求 0 命中是设计行为**：混合注意力模型的 mamba 状态检查点按 **32-token 对齐**（`mamba_cache_chunk_size = max(FLA_CHUNK_SIZE=32, page_size)`），<32 tokens 的前缀无法形成检查点；
- API 默认不报告缓存：`--enable-cache-report` 开启后返回 `usage.prompt_tokens_details.cached_tokens`。

### 5.3 鉴权
- `--api-key sk-***  # 生产环境请替换为自己的密钥`：所有端点要求 `Authorization: Bearer sk-***`，无 key 返回 401。

---

## 6. MTP / EAGLE 投机解码专项（2026-08-16 翻案）

| 引擎/配置 | 短上下文 | 50k 长上下文 |
|---|---|---|
| **sglang + EAGLE(3/1/4)（最终配置）** | **~142 tps** | **~107-148 tps** ✅ |
| sglang + EAGLE(4/1/5) | ~142 tps（accept 3.1-3.3） | ~107 tps（accept 2.7-3.3） |
| sglang + EAGLE(3/topk2/6) | 119.5 tps ❌ | — （树草稿 overhead 大于收益，弃用） |
| vLLM 0.27.2rc1.dev116 + MTP | 21 tps | 22.2 tps |
| vLLM 0.27.2rc1.dev126 + MTP（含 PR #49793 all-reduce 融合） | 22 tps | 24.7 tps |
| vLLM 无投机 | 48 tps | 47.6 tps |
| sglang 无投机（旧最终） | 74-75 tps | 70-75 tps |

### 旧结论"长上下文 EAGLE 崩溃（12.8 tps）"的翻案

旧测量是**统计假象 + 真实 OOM 叠加**：

1. **日志统计假象（主因）**：`Decode batch` 日志行的 `gen throughput` = 距上一行日志的 token/墙钟时间。长 prefill（50k ≈ 15s）结束后的**第一行** decode 日志把整个 prefill 时间计入分母 → 显示 6-13 tps。该行之后 decode 实际 **92-148 tps**（accept len 2.1-3.05 不掉）。压测脚本若只采到这一行（输出太短、日志间隔 40 迭代）就会报出"崩溃"。修正方法：丢弃每个请求 prefill 后的首行 decode 日志，或用客户端 (wall − TTFT) 计算 decode tps。
2. **真实 OOM（已修）**：EAGLE 在 `mem-fraction 0.85/0.88` 下有两个 OOM 点——
   - 启动期：draft-extend CUDA graph 捕获 logits all-gather buffer（词表 248k 放大显存）失败；
   - 运行期：长 prefill 每个 chunk 后跟 draft-extend（已随 target 分块，PR #26329），瞬时激活撞上余量不足。
   - **修复：`--mem-fraction-static 0.80` + `--disable-prefill-cuda-graph`**（prefill 图捕获是显存大头；prefill 本身 GEMM bound，禁图无损 TTFT）。
3. 顺带核实：verify 后 mamba 状态提交走 `scatter_mamba_states_after_mtp_verify` 融合 gather-scatter（O(1) 定长），追踪/提交路径无 O(序列长度) 开销；50k spec decode profile 显示全注意力 verify 仅 ~1.8ms/迭代，无病态热点。**GDN "状态重放 O(n) 灾难"在 0.5.17 上不存在**。

### 质量与正确性验证（EAGLE 配置）

- 8 题质量集：无异常项（Q1 乘法题与基线同样答错，属模型行为；EAGLE 与基线关键答案等价）；
- 50k needle（884213）✓、100k（实测 133k tokens）needle ✓（含生产 :8000 冒烟）；
- 4 并发冒烟 233-256 tps 聚合 ✓；多模态（红色方块识图）✓；
- 注意：temp=0 下同配置多次运行输出也不逐字一致（原子/批处理数值抖动），属正常现象。

### 其余结论

- vLLM MTP 的 draft 前向在 SM89 上执行效率仅 ~25%（等效 5 次完整前向/4 token，带宽需求 38.6GB/token），长上下文也无优势；
- 两个 MTP 相关 PR（#51812 gates 对齐、#51674 fused kernel）已进 nightly，新 PR #49793（trailing all-reduce 融合 + local-argmax）小幅提升（+11%），仍远不及 sglang EAGLE；
- EAGLE 参数结论：**3 steps / topk1 / 4 draft 为生产最优**（4 steps accept 略高但 tps 噪声内打平、每请求多占 mamba 槽位降低并发；topk2 树草稿在带宽 bound 下净亏）。spec 模式 mamba 5 槽/请求，`max_running_requests` 被 mamba cache 封顶 ~27（提并发可加大 `--max-mamba-cache-size`）。

---


## 7. 最终性能基准

### 7.1 EAGLE 配置（2026-08-16 起生产，单发，:8001 干净环境实测）

| 输入上下文 | TTFT（cache miss） | decode tps | accept len | 备注 |
|---|---|---|---|---|
| ~500 t | ~0.9s | **~142** | 2.9-3.1 | 无投机基线 74.85 |
| 50k（65k tokens） | 14.3-16.4s | **107-148** | 2.7-3.05 | needle ✓；无投机基线 74.84 |
| 100k（131-133k tokens） | 21-33s | 正常无衰减崩溃 | — | needle ✓（133k）；600 tokens 正常输出 |
| 4 并发短请求 | — | 聚合 **233-256** tps | — | 无投机基线 229 tps |

prefill 稳态 ~4-4.6k t/s（GEMM bound，约为 4×4090 FP8 理论值的 70%；chunk 8k/16k 无差别，32k OOM）。

### 7.2 旧 nospec flashinfer 配置（2026-08-15，冷启动，输出 200 tokens）

| 输入上下文 | 单发 TTFT | 单发 decode | 4并发平均 TTFT | 4并发聚合 decode | 4并发单请求均摊 | 单发 prefill | 4并发聚合 prefill |
|---|---|---|---|---|---|---|---|
| 100 t | 0.17s | 73.5 tps | 0.32s | 229.2 tps | 62.4 tps | 601 t/s* | 1,251 t/s |
| 5k | 1.34s | 73.5 tps | 3.28s | 166.2 tps | 54.2 tps | 3,736 t/s | 5,448 t/s |
| 20k | 3.81s | 72.1 tps | 5.48s | 126.8 tps | 47.6 tps | 5,250 t/s | 13,985 t/s |
| 50k | 10.12s | 70.1 tps | 11.12s | 136.8 tps | 60.8 tps | 4,940 t/s | 17,986 t/s |
| 100k | 22.41s | 68.0 tps | 23.67s | 182.3 tps | 56.6 tps | 4,462 t/s | 16,900 t/s |

*100 tokens 输入的 prefill 被启动开销稀释。图表见 `images/nospec_benchmark.png`。*

**要点**：EAGLE 单发 decode 较无投机 **~1.9×**（142 vs 75），且长上下文不再降速（旧"12.8 tps"为日志统计假象，见 §6）；压测口径修正——服务端 `gen throughput` 首行日志跨 prefill 窗口需丢弃，或改用客户端 (wall − TTFT)。

---

## 8. 最终配置与入口

### 8.1 `/root/run_sglang.sh`（最优参数，2026-08-16 起为 EAGLE 版）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m sglang.launch_server \
  --model-path /root/models/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 --context-length 262144 \
  --chunked-prefill-size 8192 --max-prefill-tokens 32768 \
  --host 0.0.0.0 --port 8000 --mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --load-format safetensors --kv-cache-dtype fp8_e4m3 \
  --mm-feature-transport cpu \
  --attention-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --api-key sk-***  # 生产环境请替换为自己的密钥 --enable-metrics --enable-cache-report
```
**EAGLE 显存红线**：`mem-fraction` 必须 ≤0.80 且关闭 prefill CUDA graph，否则 draft-extend graph 捕获（启动期）或长 prefill 的 draft-extend（运行期）OOM，见 §6。
备份：`run_sglang.sh.best-nospec`（无投机回退版，74-75 tps）、`run_sglang.sh.eagle-final`（同当前生产）、`run_sglang.sh.bak-pre-fix-20260816`（EAGLE@0.88 有 OOM 缺陷的旧版）、`run_sglang.sh.triton-baseline`、`run_sglang.sh.pre-opt`。
实验工具链：`/root/gdn-opt/`（`sgl_launch.sh` 多配置启动器、`sgl_bench3.py` 统一压测、`sgl_ttft.py` prefill 压测、`sgl_final_check.py` 验收、`results/` 全部 JSON 原始数据）；本地镜像 `/Users/wushanzheng/Documents/dsh/gdn-opt/`。

### 8.2 调用入口
服务监听标准 OpenAI 兼容端点（`/v1/chat/completions`，含流式、`reasoning_content`、`prompt_tokens_details.cached_tokens`），统一 `Authorization: Bearer <api-key>` 鉴权；按自身网络环境配置内网/公网入口即可。

---

## 9. 关键文件清单

| 位置 | 说明 |
|---|---|
| `/root/run_sglang.sh` | sglang 启动脚本（**EAGLE 最优配置，2026-08-16 起**） |
| `/root/run_sglang.sh.best-nospec` | 无投机回退版（74-75 tps） |
| `/root/gdn-opt/` | 本轮调优实验工具链与 `results/` 原始数据 |
| `/etc/systemd/system/sglang-qwen38.service` | systemd 服务 |
| `/opt/sgenv/` | sglang 环境（含 W8A8 调优配置、cccl 补丁） |
| `/opt/vllm-env/` + `/root/vllm_run.sh` | vLLM nightly 环境（dev126，备用） |
| `/root/models/Qwen3.8-27B-FP8/` | 模型权重 |
| `images/`（本仓库） | 性能图表（含无投机完整 benchmark） |

---

## 10. 遗留问题与未来方向

1. ~~**MTP 投机解码不可用**~~ **已解决（2026-08-16）**：长上下文"崩溃"为日志统计假象 + EAGLE 显存红线（0.80 + 禁 prefill 图）修复后，EAGLE 全面上线（见 §6）；vLLM 侧仍无优势；
2. **通信瓶颈 15-21%**：VM 无 P2P 的结构性开销（NCCL 已是 RING_LL；`NCCL_ALGO=TREE` 直接崩溃；flashinfer/mscclpp/torch-symm-mem 融合 all-reduce 均需 P2P/NVLS）。**驱动层面已逐项排查（2026-08-16）**：① PCIe 链路负载下实测 Gen4 x16 满速（空闲显示 2.5GT/s 是链路省电降速，假象）；② 锁频 `-lgc 3105 -lmc 10501` 无收益——负载下被功耗墙压回 ~2745/10251 MHz，已还原默认；③ P2P 是 GeForce 驱动的产品级封锁，无用户态开关。换物理机（NVLink）或 H20/A100 级硬件可消除大半；PP2×TP2 可将 all-reduce 环从 4 卡缩到 2 卡，但引入流水线气泡且 EAGLE×PP 兼容性未验证，风险大于收益；
3. **GDN 层加速**：0.5.17 的 Triton 融合 kernel 已仅占 decode ~2%，无需自研；flashinfer/cutedsl 后端需 SM90+（TMA），换 Hopper/Blackwell 后可弃用 Triton；
4. **`--enable-torch-compile` 不可用**：inductor 包装 `_causal_conv1d_update_kernel` 崩溃（grid 参数 bug），待 sglang/torch 升级后重试；
6. **磁盘**：35GB 可用（vLLM 环境占 ~10GB），如需 BF16 版模型需清理。
