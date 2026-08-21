# MeZO full-parameter training (Qwen3-0.6B on LaMP-3)

<!-- @author Niket Agarwal <niket.a@samsung.com> -->

MeZO is a zeroth-order optimizer: it estimates the gradient from two forward
passes with opposite random perturbations, and never calls `backwarding()`.
That makes it the first backprop-free optimizer in nntrainer and lets it
fine-tune every parameter without storing gradients.

This document covers running it on **x86 Linux**. For how the optimizer itself
is wired into the framework, see the `MeZO` class in `nntrainer/optimizers/`.

---

## 1. Prerequisites

```bash
sudo apt install -y meson ninja-build cmake git curl g++ pkg-config
```

Submodules and the CausalLM `json.hpp` dependency are fetched below.

## 2. Build

```bash
git submodule update --init --depth 1 \
  subprojects/OpenBLAS subprojects/googletest subprojects/iniparser

# CausalLM needs json.hpp. jni/prepare_encoder.sh does this but requires wget
# and swallows download failures, so fetch it directly:
curl -sSL -o /tmp/encoder.tar.gz \
  https://github.com/nnstreamer/nnstreamer-android-resource/raw/main/external/encoder-0.2.tar.gz
tar -xzf /tmp/encoder.tar.gz -C /tmp json.hpp
cp /tmp/json.hpp Applications/CausalLM/json.hpp

meson setup build \
  -Denable-transformer=true \
  -Denable-tflite-backbone=false \
  -Denable-tflite-interpreter=false

ninja -C build Applications/CausalLM/nntr_mezo_train
```

`enable-tflite-*` are off because this path does not use TFLite and enabling
them requires `flatc` plus a tensorflow-lite install.

Verify the optimizer before spending compute on it:

```bash
ninja -C build test/unittest/unittest_nntrainer_mezo_interface
./build/test/unittest/unittest_nntrainer_mezo_interface   # expect 8/8
```

## 3. Prepare the model

```bash
cd Applications/CausalLM/res/qwen3/qwen3-0.6b
pip install torch transformers
HF_HUB_DISABLE_XET=1 python3 weight_converter.py \
    --model_path Qwen/Qwen3-0.6B \
    --output_name model/nntr_qwen3_0.6b_fp32.bin
```

`HF_HUB_DISABLE_XET=1` avoids an intermittent failure in the Hub's newer
transfer backend.

Then place alongside the `.bin` (all four are required):

| File | Where from |
|---|---|
| `nntr_qwen3_0.6b_fp32.bin` | the converter above (~2.4 GB) |
| `config.json` | the HF snapshot |
| `generation_config.json` | the HF snapshot |
| `tokenizer.json` | `huggingface_hub.hf_hub_download("Qwen/Qwen3-0.6B", "tokenizer.json")` |

Finally add `model/nntr_config.json`:

```json
{
    "model_type": "CausalLM",
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_target": [],
    "bad_word_ids": [],
    "fsu": false,
    "fsu_lookahead": 2,
    "num_to_generate": 512,
    "init_seq_len": 512,
    "max_seq_len": 512,
    "batch_size": 1,
    "tokenizer_file": "tokenizer.json",
    "sample_input": ""
}
```

`"lora_rank": 0` is required — it is what leaves every layer trainable. The
driver forces it to 0 regardless, since this is full-parameter fine-tuning.

## 4. Train

```bash
./build/Applications/CausalLM/nntr_mezo_train \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/model \
  Applications/CausalLM/res/train_data/lamp3_user_train.txt \
  Applications/CausalLM/res/train_data/lamp3_user_test.txt \
  --MeZO_lr 0.000001 --MeZO_epsilon 0.0001 \
  --MeZO_lr_decay 0.995 \
  --batch_size 8 \
  --label_tokens 16,17,18,19,20 \
  --seq_len 512 --epochs 200 \
  --output mezo_qwen3_lamp3.bin
```

Three files are written next to `--output`:

| File | Contents |
|---|---|
| `mezo_qwen3_lamp3.bin` | best checkpoint so far (by validation loss) |
| `mezo_qwen3_lamp3_latest.bin` | end of the most recent epoch — resume from this |
| `mezo_qwen3_lamp3_history.csv` | `epoch,train_loss,valid_loss,accuracy` |

Both checkpoints are written to a temporary file and renamed into place, so an
interrupted save never leaves a truncated file.

### Resuming

```bash
./build/Applications/CausalLM/nntr_mezo_train ... \
  --resume mezo_qwen3_lamp3_latest.bin --start_epoch 37
```

MeZO carries no optimizer state — no momentum, no moment estimates — so the
weights are the entire training state and a resume is exact, not approximate.
`--start_epoch` only affects log numbering.

## 5. Choosing hyperparameters

### `MeZO_epsilon` first -- it decides whether anything converges

`epsilon` is not a step size. It is how far the two probes are displaced to
*measure* the gradient:

```
g = ( L(θ+εz) - L(θ-εz) ) / 2ε
```

That estimate is only meaningful while the probes stay in the region where the
loss is locally quadratic. Too large and the difference is dominated by
higher-order terms -- the estimate is mostly bias, and **no learning rate will
converge**, because every step points somewhere slightly wrong.

There is a cheap way to check a candidate ε: run one epoch with
`--MeZO_lr 0`, so no weight ever changes, and compare the reported
`train_loss` (the mean of the two probes) against `valid_loss` (the true,
unperturbed loss). A good ε makes them nearly equal.

Measured on Qwen3-0.6B, true `L(θ) = 3.80787`:

| ε | probe mean | gap | verdict |
|---|---|---|---|
| `1e-2` | 4.63661 | 0.83 | far outside the quadratic regime |
| `1e-3` | 4.39336 | 0.59 | still outside |
| `1e-4` | 3.80676 | **0.001** | usable |

If the gap scaled as ε² the 1e-2 -> 1e-3 step would have cut it ~100x. It
barely moved, which is how you can tell both are outside the usable band.
**Use `1e-4` for this model.** Going much smaller risks the opposite failure:
`L₊ - L₋` shrinking into float32 rounding noise.

### Then `MeZO_lr`, remembering it is coupled to ε

The update scales as **`lr / ε`**, so changing both together changes nothing.
Tune `lr` with ε fixed. At ε=1e-4, `lr=1e-6` was stable and converging.

### `MeZO_lr_decay` -- needed for the last stretch

A rate large enough to make early progress is too coarse to converge finely
later: loss descends, then random-walks on a floor. `MeZO_lr_decay` multiplies
the rate by that factor on every update (default `1.0`, i.e. off, which
reproduces the previous fixed-rate behaviour exactly).

With `0.995` the rate halves about every 139 updates. On an 8-sample overfit
this took a run that had stalled around 0.8 for ~60 epochs down to 0.72 with
accuracy rising 50% -> 87.5%.

### Batch size: use the whole dataset if you can

MeZO's update is a *single scalar* multiplying one random direction, applied to
every parameter at once. If that scalar is measured on a subset, a direction
that happens to help those samples and hurt the rest flips its sign -- and the
whole model moves the wrong way. Ordinary SGD averages a full gradient vector
and partly self-corrects; MeZO has nothing to average against.

So MeZO carries two noise sources, and one of them is free to remove:

| noise source | minibatch | full batch |
|---|---|---|
| random direction `z` | unavoidable | unavoidable |
| which samples were measured | **present** | **none** |

Measured on a small overfit set:

| batch | share of data per step | outcome |
|---|---|---|
| 1 of 8 | 12.5% | loss 0.83 -> 2.72 in 8 updates |
| 4 of 16 | 25% | stable ~60 epochs, then 1.4 -> 24.8 |
| 8 of 8 | 100% | 3.81 -> 0.72, accuracy 87.5% |

Smaller batches do not even buy speed here: at 16 samples, batch 4 does 4
updates per epoch and batch 16 does 1, but both cost 16 sample-forwards. The
cheap extra updates are actively harmful.

**Budget thousands of steps.** The original paper trains for ~20K at batch 64.
Anything under a few hundred is dominated by noise in either direction.

## 6. Memory

Peak resident for Qwen3-0.6B FP32 is ~2.7 GB, close to the 2.4 GB of weights,
because the gradient-free path skips work a backprop optimizer needs:

- no per-weight gradient tensors (`NetworkGraph::setSkipGradients`) — they
  would be allocated and zero-filled but never written
- perturbations applied in place, with no model-sized noise tensor
- the plain weight loader instead of `load_weight_lora()`, which builds a
  throwaway second copy of the model

If the machine is short on RAM and the weight file is being paged in and out,
configure with `-Denable-mmap=false -Dmmap-read=false`.

## 7. Reading the loss

`train_loss` is the mean of the two probe losses, `(L(θ+εz) + L(θ−εz)) / 2`,
which equals `L(θ) + O(ε²)`. This matters: the last forward pass inside a step
is a *perturbed* model, so reporting its loss directly — as a naive
implementation does — describes weights that are not the ones being trained.

`valid_loss` is an ordinary unperturbed forward pass and is the number to
trust when comparing runs.

## 8. Sanity check on a small model

`Applications/MeZO/` trains a 3-layer fully-connected net on MNIST through the
same `MeZO::trainStep()`, converging in minutes rather than days:

```bash
ninja -C build Applications/MeZO/jni/mezo_example
./build/Applications/MeZO/jni/mezo_example /path/to/mnist   # raw idx files
```

Expect roughly 0.686 → 0.089 training loss and ~70% accuracy over 100 epochs.
Useful for confirming the optimizer works before committing days of compute.
