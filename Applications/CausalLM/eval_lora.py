#!/usr/bin/env python3
"""
Evaluate trained LoRA adapter on lamp3_user_test.txt.

Parses the chat-format test file, extracts the prompt (user message) and
expected answer (assistant response), runs nntr_causallm inference on each
prompt, and compares the model's output to the expected answer.

Usage:
    python3 eval_lora.py --model_dir <path> --test_data <path> [options]

Options:
    --model_dir <path>      Model directory (default: res/qwen3/qwen3-0.6b)
    --test_data <path>      Test data file (default: res/train_data/lamp3_user_test.txt)
    --num_generate <int>    Number of tokens to generate per sample (default: 5)
    --output <path>         Output results file (default: /tmp/eval_results.txt)
    --nntrainer_root <path> NNTrainer root directory (to find build output)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def parse_test_data(path):
    """Parse the chat-format test file into (prompt, expected_answer) pairs.

    Each sample looks like:
        <|im_start|>user
        <question text><|im_end|>
        <|im_start|>assistant
        <answer>

    We extract the full prompt up to (but not including) the answer, and the
    expected answer (the text after the assistant marker).
    """
    with open(path, "r") as f:
        content = f.read()

    samples = []
    # Split on <|im_start|>user to get individual samples
    parts = content.split("<|im_start|>user")
    for part in parts[1:]:  # skip the first empty part
        # Find the assistant response
        assistant_match = re.search(
            r"<\|im_start\|>assistant\n(.+?)(?=<\|im_start\|>user|$)",
            part,
            re.DOTALL,
        )
        if not assistant_match:
            continue

        answer = assistant_match.group(1).strip()
        # The prompt is everything up to and including "<|im_start|>assistant\n"
        # but NOT the answer itself
        prompt = "<|im_start|>user" + part[: assistant_match.start() + len("<|im_start|>assistant\n")]

        # Insert think-disable marker (same as ensureThinkDisableMarker in
        # lora_train.cpp) so the model answers directly instead of thinking
        think_disable = "🤖\n\n Sure! \n\n"
        prompt = prompt + think_disable

        samples.append({"prompt": prompt, "expected": answer})

    return samples


def run_inference(nntr_causallm, model_dir, config_path, config, prompt, env):
    """Run nntr_causallm with the given prompt and capture the output.

    Updates sample_input in nntr_config.json before each call, since
    nntr_causallm reads the prompt from there when no argv[2] is given.
    """
    config["sample_input"] = prompt
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    try:
        cmd = [nntr_causallm, model_dir]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout per sample
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"


def extract_prediction(output):
    """Extract the model's prediction from nntr_causallm output.

    The output prints the prompt first, then generates tokens after it.
    The generated text follows the assistant marker and the think-disable
    marker (🤖\n\n Sure! \n\n). We look for a digit 1-5 in the generated
    portion (after the last assistant marker).
    """
    # The generated text is everything after the last assistant marker
    marker = "<|im_start|>assistant\n"
    if marker in output:
        after = output.rsplit(marker, 1)[-1]
    else:
        after = output

    # Look for a digit 1-5 anywhere after the think-disable markers
    # The think-disable text is "🤖\n\n Sure! \n\n" — skip past it
    # and search for the first digit 1-5 in the remaining text
    think_end = " Sure! \n\n"
    search_text = after
    if think_end in search_text:
        search_text = search_text.split(think_end, 1)[-1]

    # Find the first digit 1-5
    match = re.search(r"[1-5]", search_text)
    if match:
        return match.group(0)

    # Fallback: search in the entire generated portion
    match = re.search(r"[1-5]", after)
    if match:
        return match.group(0)

    # Last resort: return the raw text (trimmed)
    lines = [l.strip() for l in after.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on test data")
    parser.add_argument("--model_dir", default="res/qwen3/qwen3-0.6b")
    parser.add_argument("--test_data", default="res/train_data/lamp3_user_test.txt")
    parser.add_argument("--num_generate", default=20, type=int)
    parser.add_argument("--output", default="/tmp/eval_results.txt")
    parser.add_argument("--nntrainer_root", default=".")
    args = parser.parse_args()

    # Build paths
    nntr_causallm = os.path.join(args.nntrainer_root, "build/Applications/CausalLM/nntr_causallm")
    config_path = os.path.join(args.model_dir, "nntr_config.json")

    if not os.path.isfile(nntr_causallm):
        print(f"ERROR: nntr_causallm not found at {nntr_causallm}")
        print("Please build the project first: ninja -C build")
        sys.exit(1)

    if not os.path.isfile(config_path):
        print(f"ERROR: nntr_config.json not found at {config_path}")
        sys.exit(1)

    # Check that LoRA adapter exists in model dir
    lora_path = os.path.join(args.model_dir, "lora_adapter.bin")
    if not os.path.isfile(lora_path):
        print(f"ERROR: lora_adapter.bin not found at {lora_path}")
        print("Please copy your trained adapter to the model directory.")
        sys.exit(1)

    # Load config once; we'll update sample_input per-sample
    with open(config_path, "r") as f:
        original_config = json.load(f)

    config = original_config.copy()
    config["lora_file_name"] = "lora_adapter.bin"
    config["num_to_generate"] = args.num_generate
    config["batch_size"] = 1

    print(f"Using LoRA: lora_file_name=lora_adapter.bin, num_to_generate={args.num_generate}")
    print()

    # Set up environment
    env = os.environ.copy()
    openblas_lib = os.path.join(args.nntrainer_root, "subprojects/OpenBLAS/build/lib")
    if os.path.isdir(openblas_lib):
        env["LD_LIBRARY_PATH"] = openblas_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    # Parse test data
    samples = parse_test_data(args.test_data)
    print(f"Loaded {len(samples)} test samples from {args.test_data}")
    print()

    # Run evaluation
    results = []
    correct = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        expected = sample["expected"]

        print(f"[{i+1}/{total}] Expected: {expected}")

        stdout, stderr = run_inference(nntr_causallm, args.model_dir, config_path, config, prompt, env)

        if stderr == "TIMEOUT":
            prediction = "TIMEOUT"
        else:
            prediction = extract_prediction(stdout)

        # For LAMP-3, the answer is a single digit (1-5)
        # Extract just the first digit from the prediction
        pred_digit = re.search(r"[1-5]", prediction)
        pred_clean = pred_digit.group(0) if pred_digit else prediction.strip()

        is_correct = pred_clean == expected.strip()
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"         Predicted: {pred_clean}  {status}")
        print()

        results.append({
            "index": i + 1,
            "expected": expected.strip(),
            "predicted": pred_clean,
            "correct": is_correct,
            "raw_output": stdout[-200:] if stdout else stderr,
        })

    # Restore original config
    with open(config_path, "w") as f:
        json.dump(original_config, f, indent=2)

    # Write results
    accuracy = (correct / total * 100) if total > 0 else 0
    print("=" * 50)
    print(f"  Evaluation Complete")
    print(f"  Total samples:  {total}")
    print(f"  Correct:         {correct}")
    print(f"  Accuracy:        {accuracy:.1f}%")
    print("=" * 50)

    with open(args.output, "w") as f:
        f.write(f"LoRA Evaluation Results\n")
        f.write(f"Model: {args.model_dir}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(f"LoRA: {lora_path}\n")
        f.write(f"Tokens generated: {args.num_generate}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.1f}%\n")
        f.write(f"\n{'='*80}\nPer-sample results:\n{'='*80}\n")
        for r in results:
            status = "CORRECT" if r["correct"] else "WRONG"
            f.write(f"[{r['index']:3d}] Expected: {r['expected']:5s}  Predicted: {r['predicted']:10s}  {status}\n")

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
