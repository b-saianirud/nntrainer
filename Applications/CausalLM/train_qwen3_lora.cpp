// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   train_qwen3_lora.cpp
 * @date   31 July 2026
 * @brief  CLI driver for LoRA fine-tuning a Qwen3 CausalLM model.
 * @bug    No known bugs except for NYI items
 */

#include <lora_train.h>
#include <qwen3_causallm.h>
#include <transformer.h>

#include <dataset.h>
#include <model.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


namespace {

/**
 * @brief Read a thermal zone temperature from /sys/class/thermal.
 * @param zone_index thermal zone index (default 0 = usually CPU/GPU)
 * @return temperature in millidegrees Celsius, or -1 on failure
 */
int readThermalZone(int zone_index = 0) {
  std::string path = "/sys/class/thermal/thermal_zone" +
                     std::to_string(zone_index) + "/temp";
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
    std::string type_path = "/sys/class/thermal/thermal_zone" +
                            std::to_string(i) + "/type";
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
 * @brief Read current process RSS memory from /proc/self/status (Linux/Android).
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
    << "  --seq_len <int>       training sequence length; overrides\n"
    << "                        nntr_config.json's init_seq_len. Attention\n"
    << "                        memory is quadratic in this, so prefer the\n"
    << "                        smallest length that fits your samples.\n"
    << "  --clip_grad <float>   clip LoRA gradients to this global norm\n"
    << "                        (0 = off, the default)\n"
    << "  --train_norms         also train the RMSNorm gammas alongside the\n"
    << "                        LoRA adapters (default: norms frozen)\n"
    << "  --lora_qat            fake-quantize loraA/loraB to the Q4_0 grid\n"
    << "                        during training (per-block EMA scales,\n"
    << "                        straight-through backward). Implies\n"
    << "                        --lora_weight_q4.\n"
    << "  --lora_weight_q4      also save a Q4_0 adapter (output path with\n"
    << "                        a _q4.bin suffix) alongside the FP32 one.\n"
    << "                        With --lora_qat, force-feeds the calibrated\n"
    << "                        EMA scales; without it, quantizes the\n"
    << "                        trained FP32 adapter post hoc (PTQ).\n"
    << "  --seed <int>          RNG seed for epoch shuffling (default 42)\n";
}

/** @brief Per-epoch bookkeeping shared with the training callback. */
struct EpochState {
  causallm::Qwen3CausalLM *model;
  std::string output_path;
  std::string q4_output_path; // empty unless --lora_weight_q4 was passed
  unsigned int epoch = 0;
  float best_loss = std::numeric_limits<float>::max();
  int thermal_zone = 0;
  long total_mem_kb = -1;
  std::chrono::steady_clock::time_point train_start;
  std::chrono::steady_clock::time_point epoch_start;
};

void onEpochComplete(void *user_data) {
  auto *st = static_cast<EpochState *>(user_data);
  ++st->epoch;

  auto now = std::chrono::steady_clock::now();
  auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
                          now - st->epoch_start)
                          .count();
  auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                          now - st->train_start)
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
  std::cout << "  valid_loss:    " << valid_stats.loss << std::endl;
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

  // Save whenever validation loss improves, so an interrupted run still
  // leaves the best adapter on disk.
  if (valid_stats.loss < st->best_loss) {
    st->best_loss = valid_stats.loss;
    try {
      st->model->save_weight_lora(st->output_path);
      std::cout << "  saved LoRA adapter -> " << st->output_path << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "  failed to save LoRA adapter: " << e.what() << std::endl;
    }
    if (!st->q4_output_path.empty()) {
      try {
        st->model->save_weight_lora_q4(st->q4_output_path);
        std::cout << "  saved Q4_0 LoRA adapter -> " << st->q4_output_path
                  << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "  failed to save Q4_0 LoRA adapter: " << e.what()
                  << std::endl;
      }
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
  unsigned int seq_len_override = 0;
  float clip_grad = -1.0f;
  bool train_norms = false;
  bool lora_qat = false;
  bool lora_weight_q4 = false;
  unsigned int seed = 42;

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
      else if (arg == "--seq_len")
        seq_len_override =
          static_cast<unsigned int>(std::stoul(next("--seq_len")));
      else if (arg == "--clip_grad")
        clip_grad = std::stof(next("--clip_grad"));
      else if (arg == "--train_norms")
        train_norms = true;
      else if (arg == "--lora_qat")
        lora_qat = true;
      else if (arg == "--lora_weight_q4")
        lora_weight_q4 = true;
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

  if (output_path.empty())
    output_path = model_path + "/lora_adapter.bin";

  try {
    causallm::json cfg =
      causallm::LoadJsonFile(model_path + "/config.json");
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

    if (!nntr_cfg.contains("lora_target") ||
        nntr_cfg["lora_target"].empty())
      nntr_cfg["lora_target"] =
        std::vector<std::string>{"wq",     "wk",       "wv",      "wo",
                                 "ffn_up", "ffn_gate", "ffn_down"};

    if (train_norms)
      nntr_cfg["lora_train_norms"] = true;

    if (clip_grad >= 0.0f)
      nntr_cfg["lora_clip_grad_by_norm"] = clip_grad;

    // --lora_qat implies --lora_weight_q4: there is no point calibrating
    // fake-quant EMA scales without also saving the adapter they describe.
    if (lora_qat)
      lora_weight_q4 = true;
    if (lora_qat)
      nntr_cfg["lora_qat"] = true;
    if (lora_weight_q4)
      nntr_cfg["lora_weight_q4"] = true;

    if (seq_len_override) {
      nntr_cfg["init_seq_len"] = seq_len_override;
      // max_seq_len must not sit below the training length; mha_core derives
      // its max_timestep from it.
      if (nntr_cfg.value("max_seq_len", 0u) < seq_len_override)
        nntr_cfg["max_seq_len"] = seq_len_override;
    }

    const unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    const unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    std::string q4_output_path;
    if (lora_weight_q4) {
      std::filesystem::path p(output_path);
      q4_output_path = (p.parent_path() / (p.stem().string() + "_q4.bin")).string();
    }

    std::cout << "model:        " << model_path << "\n"
              << "data:         " << data_path << "\n"
              << "lora_rank:    " << nntr_cfg["lora_rank"].get<unsigned int>()
              << "\n"
              << "lora_alpha:   " << nntr_cfg["lora_alpha"].get<unsigned int>()
              << "\n"
              << "seq_len:      " << nntr_cfg["init_seq_len"].get<unsigned int>()
              << "\n"
              << "clip_grad:    "
              << nntr_cfg.value("lora_clip_grad_by_norm", 0.0f) << "\n"
              << "lora_qat:     " << (lora_qat ? "true" : "false") << "\n"
              << "lr:           " << lr << "\n"
              << "epochs:       " << epochs << "\n"
              << "output:       " << output_path << "\n";
    if (!q4_output_path.empty())
      std::cout << "output_q4:    " << q4_output_path << std::endl;
    else
      std::cout << std::flush;

    causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
    model.initializeForTraining(lr, epochs);

    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    model.load_weight_lora(weight_file, resume_lora_path);

    auto *tokenizer = model.getTokenizer();
    if (tokenizer == nullptr)
      throw std::runtime_error(
        "model has no tokenizer; a tokenizer_file is required for training");

    causallm::TrainingDataGenerator data_gen(data_path, tokenizer, seq_len,
                                             vocab_size, max_samples, seed);
    std::cout << "samples:      " << data_gen.size() << std::endl;

    std::shared_ptr<ml::train::Dataset> dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, causallm::trainingDataGenCb,
      &data_gen);
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);
    model.setDataset(ml::train::DatasetModeType::MODE_VALID, dataset);

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
    state.q4_output_path = q4_output_path;
    state.thermal_zone = thermal_zone;
    state.total_mem_kb = total_mem;
    state.train_start = train_start;
    state.epoch_start = train_start;

    model.train(onEpochComplete, &state);

    auto train_end = std::chrono::steady_clock::now();
    auto total_train_time = std::chrono::duration_cast<std::chrono::seconds>(
                              train_end - train_start)
                              .count();

    // Print final summary
    int final_temp = readThermalZone(thermal_zone);
    long final_rss = readMemoryUsageKB();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Training Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  best valid_loss: " << state.best_loss << std::endl;
    std::cout << "  total time:      " << total_train_time << " s" << std::endl;
    if (final_temp >= 0)
      std::cout << "  final temp:      " << (final_temp / 1000.0f) << " C"
                << std::endl;
    if (final_rss > 0) {
      std::cout << "  final RSS:       " << (final_rss / 1024.0f) << " MB";
      if (total_mem > 0)
        std::cout << " (" << (static_cast<float>(final_rss) / total_mem * 100.0f)
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
