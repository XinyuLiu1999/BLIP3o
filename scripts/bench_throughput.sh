#!/bin/bash
# ============================================================
# 吞吐扫参: 在 128-shard 子集上把几组配置各跑 STEPS 步, 量 samples/s 与显存峰值。
# 用法: bash scripts/bench_throughput.sh            # 默认 5 组配置
#       STEPS=100 bash scripts/bench_throughput.sh
#       CONFIGS="nogc:16:False:1 bs32:32:False:4" bash scripts/bench_throughput.sh
#
# 每组配置的格式: 名字:PER_DEVICE_BS:GRAD_CKPT:DL_WORKERS
# 显存峰值用 nvidia-smi 采样(绝对值), 比 HF 的 mem_delta 更可信。
# ============================================================
set -u

REPO_DIR=/cephfs/liuxinyu/BLIP3o
RESULTS=/cephfs/liuxinyu/BLIP3o-Pretrain-results
SUBSET=${SUBSET:-$REPO_DIR/experiments/smoke_2gpu}
STEPS=${STEPS:-60}
BENCH_DIR=${BENCH_DIR:-$RESULTS/_bench}

# base 先跑, 作为对照; 后面依次隔离 gradient checkpointing / dataloader / batch size
CONFIGS=${CONFIGS:-"base:16:True:1 nogc:16:False:1 w4:16:True:4 bs32:32:False:4 bs64:64:False:4"}

if [ ! -d "$SUBSET" ]; then
    echo "ERROR: 子集实验目录不存在: $SUBSET"
    echo "       建法: head -128 experiments/rebalanced_E_guarded/shardlist.txt > .../shardlist.txt"
    exit 1
fi

# nvidia-smi 能列出卡不代表 CUDA 能用: NVSwitch 机器上 fabricmanager 没起来时,
# nvidia-smi -L 正常, 但 cudaGetDeviceCount 返回 802 (cudaErrorSystemNotReady),
# 每组配置都会在 NCCL init 处炸。先花一秒确认, 别浪费 40 分钟跑 5 组必失败的配置。
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)" 2>/dev/null; then
    echo "ERROR: nvidia-smi 看得到卡, 但 torch 拿不到 CUDA 设备。"
    python -c "import torch; torch.cuda.init()" 2>&1 | tail -3
    echo "       常见原因: NVSwitch fabricmanager 未就绪 (CUDA error 802)。确认:"
    echo "         nvidia-smi -q | grep -iA3 fabric"
    echo "       Fabric State 不是 Completed 就是集群侧问题, 容器内无法自行修复。"
    exit 1
fi

mkdir -p "$BENCH_DIR"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "============================================"
echo "吞吐扫参  GPUs=$NGPU  STEPS=$STEPS  子集=$SUBSET"
echo "配置: $CONFIGS"
echo "============================================"

for cfg in $CONFIGS; do
    IFS=: read -r name bs gc w <<<"$cfg"
    run="bench-$name"
    echo
    echo ">>> [$name] per_device_bs=$bs grad_ckpt=$gc workers=$w  (global=$((NGPU*bs)))"
    rm -rf "${RESULTS:?}/$run"

    # 后台采样显存, 与本组配置同生命周期
    memfile="$BENCH_DIR/$name.mem"; : >"$memfile"
    ( while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$memfile" 2>/dev/null
        sleep 5
      done ) &
    sampler=$!

    EXPERIMENT_DIR="$SUBSET" RUN_NAME="$run" MAX_STEPS="$STEPS" SAVE_STEPS=100000000 \
    PER_DEVICE_BS="$bs" GRAD_CKPT="$gc" DL_WORKERS="$w" SKIP_MEM_METRICS=False \
    WANDB_MODE=offline \
        bash "$REPO_DIR/scripts/run_rebalanced_single_node.sh" >/dev/null 2>&1
    rc=$?

    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

    log="$RESULTS/$run/logs/train.log"
    cp "$log" "$BENCH_DIR/$name.log" 2>/dev/null
    peak=$(sort -n "$memfile" 2>/dev/null | tail -1)
    sps=$(grep -oE "'train_samples_per_second': [0-9.]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
    oom=$(grep -c "CUDA out of memory" "$log" 2>/dev/null)

    if [ "$rc" -ne 0 ] || [ -z "$sps" ]; then
        status="FAIL$([ "${oom:-0}" -gt 0 ] && echo '(OOM)')"
        echo "    -> $status  峰值显存 ${peak:-?} MiB"
        # 把真正的报错抬到终端, 而不是让用户去 grep 6 万行日志
        echo "       首个报错:"
        grep -m3 -E "Error [0-9]+:|ValueError|RuntimeError|CUDA out of memory|fatal error" \
             "$log" 2>/dev/null | cut -c1-160 | sed 's/^/         /'
        # OOM 是这组配置的正常结论(继续测下一组); 其余错误说明环境有问题, 早停。
        if [ "${oom:-0}" -eq 0 ]; then
            echo "       -> 非 OOM 故障, 后续配置大概率同样失败, 停止扫参。"
            rm -rf "${RESULTS:?}/$run"
            break
        fi
    else
        echo "    -> ${sps} samples/s   峰值显存 ${peak:-?} MiB / 81920"
        status="$sps"
    fi
    echo "$name|$bs|$gc|$w|$status|${peak:-?}" >>"$BENCH_DIR/summary.psv"

    # 每组跑完都会存一份最终模型(数 GB), 及时清掉
    rm -rf "${RESULTS:?}/$run"
done

echo
echo "============================================"
printf "%-8s %6s %8s %8s %14s %12s\n" 配置 bs grad_ckpt workers samples/s 峰值MiB
awk -F'|' '{printf "%-8s %6s %8s %8s %14s %12s\n", $1,$2,$3,$4,$5,$6}' "$BENCH_DIR/summary.psv"
echo "============================================"
echo "明细日志: $BENCH_DIR/*.log"
