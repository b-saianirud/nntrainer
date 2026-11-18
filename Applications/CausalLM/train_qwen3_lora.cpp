// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   train_qwen3_lora.cpp
 * @date   31 July 2026
 * @brief  CLI driver for LoRA fine-tuning a Qwen3 CausalLM model.
 * @bug    No known bugs except for NYI items
 */

#include <causal_lm_loss.h>
#include <lora_train.h>
#include <qwen3_causallm.h>
#include <transformer.h>

#include <dataset.h>
#include <model.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

/**
 * @brief Read a thermal zone temperature from /sys/class/thermal.
 * @param zone_index thermal zone index (default 0 = usually CPU/GPU)
 * @return temperature in millidegrees Celsius, or -1 on failure
 */
int readThermalZone(int zone_index = 0) {
  std::string path =
    "/sys/class/thermal/thermal_zone" + std::to_string(zone_index) + "/temp";
  std::ifstream f(path);
  if (!f.is_open())
    return -1;
  int temp;
  f >> temp;
  return temp;
}

/**
 * @brief Find the thermal zone whose type contains "cpu" or "gpu".
 * Scans /sys/class/thermal/thermal_zoneN/type for a match and returns
 * the first matching zone index. Falls back to zone 0.
 */
int findCpuThermalZone() {
  for (int i = 0; i < 16; ++i) {
    std::string type_path =
      "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
    std::ifstream f(type_path);
    if (!f.is_open())
      continue;
    std::string type;
    std::getline(f, type);
    // Match common type names: "cpu", "cpu-0-0", "soc", "gpu", etc.
    if (type.find("cpu") != std::string::npos ||
        type.find("CPU") != std::string::npos ||
        type.find("gpu") != std::string::npos ||
        type.find("GPU") != std::string::npos)
      return i;
  }
  return 0;
}

/**
 * @brief Read current process RSS memory from /proc/self/status
 * (Linux/Android).
 * @return RSS in kilobytes, or -1 on failure.
 */
long readMemoryUsageKB() {
  std::ifstream f("/proc/self/status");
  if (!f.is_open())
    return -1;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      // Format: "VmRSS:    12345 kB"
      long kb = -1;
      std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
      return kb;
    }
  }
  return -1;
}

/**
 * @brief Read total system memory from /proc/meminfo (Linux/Android).
 * @return total system memory in kilobytes, or -1 on failure.
 */
long readTotalMemoryKB() {
  std::ifstream f("/proc/meminfo");
  if (!f.is_open())
    return -1;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 9, "MemTotal:") == 0) {
      long kb = -1;
      std::sscanf(line.c_str(), "MemTotal: %ld kB", &kb);
      return kb;
    }
  }
  return -1;
}

void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_dir> <train_data.txt> [options]\n"
    << "\nOptions:\n"
    << "  --lr <float>          learning rate (default 1e-4)\n"
    << "  --epochs <int>        number of epochs (default 1)\n"
    << "  --output <path>       LoRA adapter output path\n"
    << "                        (default <model_dir>/lora_adapter.bin)\n"
    << "  --lora_path <path>    resume from an existing LoRA adapter\n"
    << "  --lora_rank <int>     LoRA rank; overrides nntr_config.json\n"
    << "  --lora_alpha <int>    LoRA alpha; overrides nntr_config.json\n"
    << "  --max_samples <int>   cap the number of training samples\n"
    << "  --val_data <path>     separate validation data file (default:\n"
    << "                        reuse the training data for validation)\n"
    << "  --max_val_samples <int> cap the number of validation samples\n"
    << "  --seq_len <int>       training sequence length; overrides\n"
    << "                        nntr_config.json's init_seq_len. Attention\n"
    << "                        memory is quadratic in this, so prefer the\n"
    << "                        smallest length that fits your samples.\n"
    << "  --clip_grad <float>   clip LoRA gradients to this global norm\n"
    << "                        (0 = off, the default)\n"
    << "  --train_norms         also train the RMSNorm gammas alongside the\n"
    << "                        LoRA adapters (default: norms frozen)\n"
    << "  --seed <int>          RNG seed for epoch shuffling (default 42)\n"
    << "  --batch_size <int>    training batch size; overrides\n"
    << "                        nntr_config.json's batch_size\n"
    << "  --lr_decay_steps <int> decay lr linearly to 0 over this many steps\n"
    << "                        (default 0 = constant lr)\n"
    << "  --adam_eps <float>    Adam epsilon (default: nntrainer's, 1e-7)\n"
    << "  --full_seq_loss       next-token loss at EVERY position of each "
       "line\n"
    << "                        (language-model style, masked on padding)\n"
    << "                        instead of the last token only. Each line is\n"
    << "                        one sample; <eos> is appended. valid_acc is\n"
    << "                        not meaningful in this mode; the saved "
       "adapter\n"
    << "                        is the one with the lowest valid_loss.\n"
    << "  --eval_only           with --full_seq_loss: no training, just "
       "report\n"
    << "                        token-level validation loss / perplexity /\n"
    << "                        top-1 / top-5 accuracy of --lora_path on\n"
    << "                        --val_data (omit --lora_path for the base "
       "model)\n"
    << "  --eos_id <int>        end-of-text id appended with --full_seq_loss\n"
    << "                        (default 151643, Qwen3 <|endoftext|>)\n";
}

/** @brief Per-epoch bookkeeping shared with the training callback. */
struct EpochState {
  causallm::Qwen3CausalLM *model;
  std::string output_path;
  unsigned int epoch = 0;
  unsigned int best_epoch = 0;
  float best_loss = std::numeric_limits<float>::max();
  float best_acc = -1.0f;
  bool select_by_loss = false; // --full_seq_loss: lowest valid_loss wins
  int thermal_zone = 0;
  long total_mem_kb = -1;
  std::chrono::steady_clock::time_point train_start;
  std::chrono::steady_clock::time_point epoch_start;
};

void onEpochComplete(void *user_data) {
  auto *st = static_cast<EpochState *>(user_data);
  ++st->epoch;

  auto now = std::chrono::steady_clock::now();
  auto epoch_duration =
    std::chrono::duration_cast<std::chrono::seconds>(now - st->epoch_start)
      .count();
  auto total_duration =
    std::chrono::duration_cast<std::chrono::seconds>(now - st->train_start)
      .count();

  auto train_stats = st->model->getTrainingStats();
  auto valid_stats = st->model->getValidStats();

  // Read temperature once for this epoch
  int temp_mc = readThermalZone(st->thermal_zone);
  float temp_c = (temp_mc >= 0) ? (temp_mc / 1000.0f) : -1.0f;

  // Read memory usage
  long rss_kb = readMemoryUsageKB();
  float rss_mb = (rss_kb >= 0) ? (rss_kb / 1024.0f) : -1.0f;
  float mem_pct = -1.0f;
  if (rss_kb >= 0 && st->total_mem_kb > 0)
    mem_pct = (static_cast<float>(rss_kb) / st->total_mem_kb) * 100.0f;

  // Print neatly formatted epoch summary
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Epoch " << st->epoch << " Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  train_loss:    " << train_stats.loss << std::endl;
  std::cout << "  train_acc:     " << train_stats.accuracy << std::endl;
  std::cout << "  valid_loss:    " << valid_stats.loss << std::endl;
  if (!st->select_by_loss)
    std::cout << "  valid_acc:     " << valid_stats.accuracy << std::endl;
  else {
    auto &es = causallm::g_causal_lm_eval_stats;
    if (es.tokens) {
      const double mean_nll = es.nll / es.tokens;
      std::cout << "  valid_tok_loss:" << mean_nll << "  (ppl "
                << std::exp(mean_nll) << ", " << es.tokens << " tokens)\n"
                << "  valid_top1:    " << 100.0 * es.top1 / es.tokens << " %\n"
                << "  valid_top5:    " << 100.0 * es.top5 / es.tokens << " %"
                << std::endl;
    }
    es.reset();
  }
  std::cout << "  time (epoch):  " << epoch_duration << " s" << std::endl;
  std::cout << "  time (total):  " << total_duration << " s" << std::endl;
  if (temp_c >= 0)
    std::cout << "  temperature:   " << temp_c << " C" << std::endl;
  if (rss_mb >= 0) {
    std::cout << "  memory (RSS):  " << rss_mb << " MB";
    if (mem_pct >= 0)
      std::cout << " (" << mem_pct << "% of system)";
    std::cout << std::endl;
  }
  std::cout << "========================================" << std::endl;

  // Save whenever validation improves (accuracy, or loss for --full_seq_loss
  // where accuracy is meaningless), so an interrupted run still leaves the
  // best adapter on disk.
  const bool improved = st->select_by_loss
                          ? valid_stats.loss < st->best_loss
                          : valid_stats.accuracy > st->best_acc;
  if (improved) {
    st->best_epoch = st->epoch;
    st->best_acc = valid_stats.accuracy;
    st->best_loss = valid_stats.loss;
    try {
      st->model->save_weight_lora(st->output_path);
      std::cout << "  saved LoRA adapter -> " << st->output_path << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "  failed to save LoRA adapter: " << e.what() << std::endl;
    }
  }

  // Reset epoch start time for next epoch
  st->epoch_start = std::chrono::steady_clock::now();
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  const std::string data_path = argv[2];

  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path;
  std::string resume_lora_path;
  unsigned int lora_rank_override = 0;
  unsigned int lora_alpha_override = 0;
  unsigned int max_samples = 0;
  std::string val_data_path;
  unsigned int max_val_samples = 0;
  unsigned int seq_len_override = 0;
  float clip_grad = -1.0f;
  bool train_norms = false;
  unsigned int seed = 42;
  unsigned int batch_size_override = 0;
  bool full_seq_loss = false;
  bool eval_only = false;
  int eos_id = 151643;
  unsigned int lr_decay_steps = 0;
  float adam_eps = 0.0f;

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    try {
      if (arg == "--lr")
        lr = std::stof(next("--lr"));
      else if (arg == "--epochs")
        epochs = static_cast<unsigned int>(std::stoul(next("--epochs")));
      else if (arg == "--output")
        output_path = next("--output");
      else if (arg == "--lora_path")
        resume_lora_path = next("--lora_path");
      else if (arg == "--lora_rank")
        lora_rank_override =
          static_cast<unsigned int>(std::stoul(next("--lora_rank")));
      else if (arg == "--lora_alpha")
        lora_alpha_override =
          static_cast<unsigned int>(std::stoul(next("--lora_alpha")));
      else if (arg == "--max_samples")
        max_samples =
          static_cast<unsigned int>(std::stoul(next("--max_samples")));
      else if (arg == "--val_data")
        val_data_path = next("--val_data");
      else if (arg == "--max_val_samples")
        max_val_samples =
          static_cast<unsigned int>(std::stoul(next("--max_val_samples")));
      else if (arg == "--seq_len")
        seq_len_override =
          static_cast<unsigned int>(std::stoul(next("--seq_len")));
      else if (arg == "--clip_grad")
        clip_grad = std::stof(next("--clip_grad"));
      else if (arg == "--train_norms")
        train_norms = true;
      else if (arg == "--batch_size")
        batch_size_override =
          static_cast<unsigned int>(std::stoul(next("--batch_size")));
      else if (arg == "--lr_decay_steps")
        lr_decay_steps =
          static_cast<unsigned int>(std::stoul(next("--lr_decay_steps")));
      else if (arg == "--adam_eps")
        adam_eps = std::stof(next("--adam_eps"));
      else if (arg == "--full_seq_loss")
        full_seq_loss = true;
      else if (arg == "--eval_only")
        eval_only = true;
      else if (arg == "--eos_id")
        eos_id = std::stoi(next("--eos_id"));
      else if (arg == "--seed")
        seed = static_cast<unsigned int>(std::stoul(next("--seed")));
      else {
        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage(argv[0]);
        return EXIT_FAILURE;
      }
    } catch (const std::exception &e) {
      std::cerr << "Bad arguments: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (eval_only && (!full_seq_loss || val_data_path.empty())) {
    std::cerr << "--eval_only needs --full_seq_loss and --val_data"
              << std::endl;
    return EXIT_FAILURE;
  }
  if (eval_only) {
    // The framework only validates inside train(): run one step at a ~zero
    // learning rate on a single batch, then read the validation statistics.
    epochs = 1;
    lr = 1e-30f;
  }

  if (output_path.empty())
    output_path = model_path + "/lora_adapter.bin";

  try {
    causallm::json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    causallm::json generation_cfg = causallm::json::object();
    const std::string gen_cfg_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(gen_cfg_path))
      generation_cfg = causallm::LoadJsonFile(gen_cfg_path);
    causallm::json nntr_cfg =
      causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // Resolve the tokenizer path against the model directory. These configs
    // are shipped with on-device (e.g. /data/local/tmp/...) absolute paths
    // that do not exist on a build host, so fall back to the same filename
    // inside model_dir whenever the configured path is missing.
    {
      std::filesystem::path tok =
        nntr_cfg.value("tokenizer_file", std::string("tokenizer.json"));
      if (tok.is_relative())
        tok = std::filesystem::path(model_path) / tok;
      if (!std::filesystem::exists(tok))
        tok = std::filesystem::path(model_path) / tok.filename();
      if (!std::filesystem::exists(tok))
        throw std::runtime_error("tokenizer not found; looked for " +
                                 tok.string());
      nntr_cfg["tokenizer_file"] = tok.string();
    }

    // Default the LoRA config if the model's nntr_config.json leaves it off
    // (or has it at 0, the "LoRA disabled" default). CLI flags win.
    if (lora_rank_override)
      nntr_cfg["lora_rank"] = lora_rank_override;
    else if (!nntr_cfg.contains("lora_rank") ||
             nntr_cfg["lora_rank"].get<unsigned int>() == 0)
      nntr_cfg["lora_rank"] = 8;

    if (lora_alpha_override)
      nntr_cfg["lora_alpha"] = lora_alpha_override;
    else if (!nntr_cfg.contains("lora_alpha") ||
             nntr_cfg["lora_alpha"].get<unsigned int>() == 0)
      nntr_cfg["lora_alpha"] = nntr_cfg["lora_rank"].get<unsigned int>() * 2;

    if (!nntr_cfg.contains("lora_target") || nntr_cfg["lora_target"].empty())
      nntr_cfg["lora_target"] = std::vector<std::string>{
        "wq", "wk", "wv", "wo", "ffn_up", "ffn_gate", "ffn_down"};

    if (train_norms)
      nntr_cfg["lora_train_norms"] = true;

    if (clip_grad >= 0.0f)
      nntr_cfg["lora_clip_grad_by_norm"] = clip_grad;

    if (batch_size_override)
      nntr_cfg["batch_size"] = batch_size_override;

    if (seq_len_override) {
      nntr_cfg["init_seq_len"] = seq_len_override;
      // max_seq_len must not sit below the training length; mha_core derives
      // its max_timestep from it.
      if (nntr_cfg.value("max_seq_len", 0u) < seq_len_override)
        nntr_cfg["max_seq_len"] = seq_len_override;
    }

    const unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    const unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    std::cout << "model:        " << model_path << "\n"
              << "data:         " << data_path << "\n"
              << "lora_rank:    " << nntr_cfg["lora_rank"].get<unsigned int>()
              << "\n"
              << "lora_alpha:   " << nntr_cfg["lora_alpha"].get<unsigned int>()
              << "\n"
              << "seq_len:      "
              << nntr_cfg["init_seq_len"].get<unsigned int>() << "\n"
              << "clip_grad:    "
              << nntr_cfg.value("lora_clip_grad_by_norm", 0.0f) << "\n"
              << "batch_size:   " << nntr_cfg.value("batch_size", 1u) << "\n"
              << "loss:         "
              << (full_seq_loss ? "full-sequence next-token" : "last-token")
              << "\n"
              << "lr:           " << lr << "\n"
              << "epochs:       " << epochs << "\n"
              << "output:       " << output_path << std::endl;

    causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
    model.initializeForTraining(lr, epochs, full_seq_loss, lr_decay_steps,
                                adam_eps);

    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    model.load_weight_lora(weight_file, resume_lora_path);

    auto *tokenizer = model.getTokenizer();
    if (tokenizer == nullptr)
      throw std::runtime_error(
        "model has no tokenizer; a tokenizer_file is required for training");

    if (eval_only)
      max_samples = nntr_cfg.value("batch_size", 1u);
    const int gen_eos_id = full_seq_loss ? eos_id : -1;
    causallm::TrainingDataGenerator data_gen(
      data_path, tokenizer, seq_len, vocab_size, max_samples, seed, gen_eos_id);
    std::cout << "samples:      " << data_gen.size() << std::endl;

    std::shared_ptr<ml::train::Dataset> dataset =
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::trainingDataGenCb, &data_gen);
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);

    // Use a separate validation file when provided; otherwise fall back to
    // reusing the training data (original behavior).
    std::unique_ptr<causallm::TrainingDataGenerator> val_gen;
    if (!val_data_path.empty()) {
      val_gen = std::make_unique<causallm::TrainingDataGenerator>(
        val_data_path, tokenizer, seq_len, vocab_size, max_val_samples, seed,
        gen_eos_id);
      std::cout << "val_samples:  " << val_gen->size() << std::endl;
      std::shared_ptr<ml::train::Dataset> val_dataset =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 causallm::trainingDataGenCb, val_gen.get());
      model.setDataset(ml::train::DatasetModeType::MODE_VALID, val_dataset);
    } else {
      model.setDataset(ml::train::DatasetModeType::MODE_VALID, dataset);
    }

    // Set up per-epoch monitoring state
    int thermal_zone = findCpuThermalZone();
    long total_mem = readTotalMemoryKB();

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "  System Info" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    if (total_mem > 0)
      std::cout << "  total memory:  " << (total_mem / 1024.0f) << " MB"
                << std::endl;
    int pre_temp = readThermalZone(thermal_zone);
    if (pre_temp >= 0)
      std::cout << "  pre-train temp:" << (pre_temp / 1000.0f) << " C"
                << std::endl;
    long pre_rss = readMemoryUsageKB();
    if (pre_rss > 0)
      std::cout << "  pre-train RSS: " << (pre_rss / 1024.0f) << " MB"
                << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    auto train_start = std::chrono::steady_clock::now();

    EpochState state;
    state.model = &model;
    state.output_path = output_path;
    state.select_by_loss = full_seq_loss;
    state.thermal_zone = thermal_zone;
    state.total_mem_kb = total_mem;
    state.train_start = train_start;
    state.epoch_start = train_start;

    model.train(onEpochComplete, &state);

    auto train_end = std::chrono::steady_clock::now();
    auto total_train_time =
      std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start)
        .count();

    // Print final summary
    int final_temp = readThermalZone(thermal_zone);
    long final_rss = readMemoryUsageKB();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Training Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  best valid_loss: " << state.best_loss << std::endl;
    std::cout << "  best epoch:      " << state.best_epoch << std::endl;
    std::cout << "  total time:      " << total_train_time << " s" << std::endl;
    if (final_temp >= 0)
      std::cout << "  final temp:      " << (final_temp / 1000.0f) << " C"
                << std::endl;
    if (final_rss > 0) {
      std::cout << "  final RSS:       " << (final_rss / 1024.0f) << " MB";
      if (total_mem > 0)
        std::cout << " ("
                  << (static_cast<float>(final_rss) / total_mem * 100.0f)
                  << "% of system)";
      std::cout << std::endl;
    }
    std::cout << "  adapter:         " << output_path << std::endl;
    std::cout << "========================================" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
