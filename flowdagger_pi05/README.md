# flowdagger_pi05

FlowDAgger with a **pi0.5** base policy (openpi / JAX), running on the
**MetaWorld assembly** task.

A small steering actor predicts the per-chunk Gaussian noise that pi0.5 denoises
into an action chunk. When the policy stalls, a scripted MetaWorld expert takes
over; its executed actions are inverted through pi0.5's flow-matching sampler to
recover the noise that would have produced them, and the steering actor is
trained to predict that noise with a behavior-cloning (MSE) loss. The base
policy weights are never updated.


## Install

Python 3.11 and a CUDA 12 GPU are recommended.

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate

# 1. openpi submodule (pi0.5 model, sampler, transforms)
git submodule update --init flowdagger_pi05/openpi
pip install -e flowdagger_pi05/openpi

# 2. Reproduction JAX/CUDA stack + envs. Installed AFTER openpi.
pip install -r flowdagger_pi05/requirements.txt
```

The pi0.5 MetaWorld checkpoint is fetched automatically from the Hub
([pi05-metaworld](https://huggingface.co/mmurray-ms/pi05-metaworld))
on the first run and cached under `~/.cache/huggingface`. To use a local
checkpoint instead, set `METAWORLD_CHECKPOINT` to a dir containing `params/`
and `assets/`.

## Run

```bash
cd flowdagger_pi05
python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

On a headless machine (no display), set `MUJOCO_GL=egl` so MuJoCo can render the
camera observations off-screen:

```bash
MUJOCO_GL=egl python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

wandb is opt-in: pass `--prefix <name>` to log a run (project `flowdagger` by
default). Without `--prefix` no wandb run is created (a short id is still
generated for the local output dir). Set `--wandb_project` / `--wandb_entity` to
redirect, or `WANDB_MODE=offline` to log locally. Outputs (eval videos,
`eval_results.jsonl`, steering checkpoints) go to `$EXP/<run-name>` if `EXP` is
set, else `~/flowdagger_runs/<run-name>`.
# ARX 双臂部署

ARX 服务端入口：

```bash
cd /home/ubuntu/yzy/FlowDAgger/flowdagger_pi05
../run_arx_flowdagger_server.sh --record-only  # 可选：示教 / baseline
../run_arx_offline_train.sh                    # 可选：用示教先训一版 steering
../run_arx_flowdagger_server.sh                # bootstrap / closed-loop 在线 BC
```

机械臂断开时可用独立端口做纯协议测试：

```bash
../run_arx_flowdagger_server.sh --protocol-only --addr 'tcp://*:5558'
```

该模式固定返回零动作，episode 会标记 `runtime_mode=protocol_only`，不会被
离线训练或正式验收计入。严禁让实机控制客户端连接这个端口。

在连接机械臂之前，先运行真实模型端到端离线验证：

```bash
cd /home/ubuntu/yzy/FlowDAgger
./run_arx_end_to_end_smoke.sh
```

它会在统一 `flowdagger_runs/.../offline_validation/` 下跑通专家回合、反演、
100 步 BC、checkpoint 重载、shadow 和 steering 推理；验证 checkpoint 不会
写入正式运行目录的 `steering_checkpoints/ACTIVE`。

默认读取 `arx_campaign.yaml` 里的 `openpi_root` 和只读 `base_checkpoint`。
本 campaign 写入 yaml 中的 `output_root`（默认 `{runs_root}/{campaign_id}`）。
旧 campaign 不会混入。反演窗口按 `(obs, actions, 冻结基座)` 缓存，后续 BC 只反演新窗口。

部署按原版 FlowDAgger：完整服务跑 bootstrap（无 steering 时走基座），接管成功后把专家窗口反演进 replay，回合结束做 100 步 BC 并保存 ACTIVE。Steering 回归完整 50×32 noise。不需要审核 manifest、验证降幅门禁或 eligibility 文件。示教/baseline/shadow 只归档；bootstrap 与 closed-loop 的接管成功会触发在线更新。反演 batch 默认为 1，上限 4。
