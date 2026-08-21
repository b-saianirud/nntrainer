// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   train_qwen3_mezo.cpp
 * @date   07 August 2026
 * @brief  CLI driver for full-parameter MeZO (zeroth-order, backprop-free)
 *         fine-tuning of a Qwen3 CausalLM model.
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Unlike train_qwen3_lora.cpp, this driver never enables LoRA
 *          (lora_rank is forced to 0 regardless of nntr_config.json), so
 *          every layer stays at nntrainer's default trainable=true and the
 *          MeZO optimizer perturbs/updates the full ~0.6B parameter set —
 *          see NeuralNetwork::getParameterPointers() and the
 *          !opt->requiresBackprop() branch in NeuralNetwork::train_run().
 */

#include <lora_train.h>
#include <qwen3_causallm.h>
#include <transformer.h>

#include <dataset.h>
#include <model.h>
#include <util_func.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

namespace {

void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog
    << " <model_dir> <train_data.txt> <valid_data.txt> [options]\n"
    << "\nOptions:\n"
    << "  --MeZO_lr <float>      MeZO learning rate (default 1e-4)\n"
    << "  --MeZO_epsilon <float> MeZO perturbation magnitude (default 1e-2).\n"
    << "                         Must be small enough that the two probes sit\n"
    << "                         in the local-quadratic regime; if the probe\n"
    << "                         mean is far above the unperturbed loss, the\n"
    << "                         gradient estimate is mostly bias and no\n"
    << "                         learning rate will converge. 1e-4 measured\n"
    << "                         well for Qwen3-0.6B; 1e-2 did not.\n"
    << "  --MeZO_lr_decay <f>    per-update multiplicative decay on the\n"
    << "                         learning rate (default 1.0 = off). A rate\n"
    << "                         big enough to make early progress is too\n"
    << "                         coarse to converge finely later, so a fixed\n"
    << "                         rate descends and then random-walks on a\n"
    << "                         floor. e.g. 0.995 halves it every ~139 steps.\n"
    << "  --MeZO_min_lr <float>  floor for the decayed rate (default 0)\n"
    << "  --epochs <int>         number of epochs (default 1)\n"
    << "  --output <path>        full checkpoint output path\n"
    << "                         (default <model_dir>/mezo_checkpoint.bin)\n"
    << "  --max_samples <int>    cap the number of training samples\n"
    << "  --seq_len <int>        training sequence length; overrides\n"
    << "                         nntr_config.json's init_seq_len. Attention\n"
    << "                         memory is quadratic in this, so prefer the\n"
    << "                         smallest length that fits your samples.\n"
    << "  --seed <int>           RNG seed for epoch shuffling (default 42)\n"
    << "  --resume <path>        continue from a checkpoint saved by an\n"
    << "                         earlier run instead of the base weights.\n"
    << "                         MeZO keeps no optimizer state (no momentum,\n"
    << "                         no moments), so the weights are the entire\n"
    << "                         training state and this resumes exactly.\n"
    << "  --start_epoch <int>    epoch number to count from when resuming;\n"
    << "                         affects logging only (default 0)\n"
    << "  --label_tokens <ids>   comma-separated token ids the answer must be\n"
    << "                         one of, e.g. 16,17,18,19,20 for \"1\"..\"5\".\n"
    << "                         Restricts training to a choice among these\n"
    << "                         instead of a softmax over the whole ~152k\n"
    << "                         vocabulary, which is what the MeZO paper does\n"
    << "                         for classification. Ids must be contiguous.\n"
    << "                         Overrides nntr_config.json's label_token_ids.\n"
    << "  --no_save_best         do not write the best-so-far checkpoint.\n"
    << "                         That checkpoint is written every time valid\n"
    << "                         loss improves, which on a well-converging\n"
    << "                         run is nearly every epoch -- ~2.4GB each for\n"
    << "                         Qwen3-0.6B. On a memory-constrained machine\n"
    << "                         that I/O can outweigh the training itself.\n"
    << "                         Use for hyperparameter sweeps, where only\n"
    << "                         the loss curve matters.\n"
    << "  --batch_size <int>     samples averaged into each MeZO step\n"
    << "                         (default: nntr_config.json's batch_size).\n"
    << "                         MeZO's gradient estimate has variance that\n"
    << "                         grows with parameter count, and averaging\n"
    << "                         over a batch is the main lever against it --\n"
    << "                         the MeZO paper uses 64, and the MNIST demo\n"
    << "                         in Applications/MeZO uses 32. At batch 1 the\n"
    << "                         estimate is too noisy to converge on a model\n"
    << "                         this size.\n"
    << "  --save_every <int>     epochs between rolling checkpoint writes\n"
    << "                         (default 1). A full checkpoint is ~2.4GB for\n"
    << "                         Qwen3-0.6B, so raise this if epochs are\n"
    << "                         short enough that saving dominates. The\n"
    << "                         best-so-far checkpoint is written too,\n"
    << "                         unless --no_save_best.\n"
    << "\n"
    << "This driver always fine-tunes every parameter (lora_rank is forced\n"
    << "to 0 regardless of nntr_config.json) using the MeZO optimizer, which\n"
    << "never computes a gradient via backpropagation.\n";
}

/** @brief Per-epoch bookkeeping shared with the training callback. */
struct EpochState {
  causallm::Qwen3CausalLM *model;
  std::string output_path;  /**< best-so-far checkpoint */
  std::string latest_path;  /**< most recent epoch, for resuming after a crash */
  std::string csv_path;     /**< machine-readable loss history */
  unsigned int epoch = 0;   /**< counts from the resumed-at epoch, not from 0 */
  unsigned int save_every = 1; /**< epochs between "latest" checkpoint writes */
  float best_loss = std::numeric_limits<float>::max();
  bool no_save_best = false;  /**< skip best-checkpoint writes entirely */
};

/**
 * @brief Save via a temporary file, then rename into place.
 * @note A full Qwen3-0.6B checkpoint is ~2.4GB and takes a noticeable moment
 *       to write. Writing straight to the destination means a crash (or an
 *       OOM kill, which is how several runs ended) during that window leaves
 *       a truncated file that cannot be resumed from - losing the whole run.
 *       rename() within a filesystem is atomic, so the destination is always
 *       either the previous good checkpoint or the complete new one.
 */
static void saveCheckpointAtomically(causallm::Qwen3CausalLM *model,
                                     const std::string &path) {
  const std::string tmp = path + ".tmp";
  model->save_weight(tmp);
  std::filesystem::rename(tmp, path);
}

void onEpochComplete(void *user_data) {
  auto *st = static_cast<EpochState *>(user_data);
  ++st->epoch;

  auto train_stats = st->model->getTrainingStats();
  auto valid_stats = st->model->getValidStats();

  std::cout << "[epoch " << st->epoch << "] train_loss=" << train_stats.loss
            << " valid_loss=" << valid_stats.loss
            << " accuracy=" << valid_stats.accuracy << "%" << std::endl;

  // Append to the CSV before touching the (much slower) checkpoints, so the
  // loss history survives even if a save is what kills the process.
  if (!st->csv_path.empty()) {
    std::ofstream csv(st->csv_path, std::ios::app);
    if (csv)
      csv << st->epoch << "," << train_stats.loss << "," << valid_stats.loss
          << "," << valid_stats.accuracy << "\n";
  }

  // Refresh "latest" so a crash costs at most --save_every epochs. Saving
  // only on improvement is not enough: MeZO plateaus and backslides for long
  // stretches, and an interrupted run would rewind to whenever it last
  // improved.
  if (st->save_every > 0 && st->epoch % st->save_every == 0) {
    try {
      saveCheckpointAtomically(st->model, st->latest_path);
    } catch (const std::exception &e) {
      std::cerr << "  failed to save latest checkpoint: " << e.what()
                << std::endl;
    }
  }

  if (valid_stats.loss < st->best_loss) {
    st->best_loss = valid_stats.loss;
    if (st->no_save_best)
      return;
    try {
      saveCheckpointAtomically(st->model, st->output_path);
      std::cout << "  new best -> " << st->output_path << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "  failed to save best checkpoint: " << e.what()
                << std::endl;
    }
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  const std::string train_data_path = argv[2];
  const std::string valid_data_path = argv[3];

  float mezo_lr = 1e-4f;
  float mezo_epsilon = 1e-2f;
  unsigned int epochs = 1;
  std::string output_path;
  unsigned int max_samples = 0;
  unsigned int seq_len_override = 0;
  unsigned int seed = 42;
  std::string resume_path;
  unsigned int start_epoch = 0;
  unsigned int save_every = 1;
  unsigned int batch_size_override = 0;
  bool no_save_best = false;
  float mezo_lr_decay = 1.0f;
  float mezo_min_lr = 0.0f;
  std::vector<unsigned int> label_tokens;

  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    try {
      if (arg == "--MeZO_lr")
        mezo_lr = std::stof(next("--MeZO_lr"));
      else if (arg == "--MeZO_epsilon")
        mezo_epsilon = std::stof(next("--MeZO_epsilon"));
      else if (arg == "--MeZO_lr_decay")
        mezo_lr_decay = std::stof(next("--MeZO_lr_decay"));
      else if (arg == "--MeZO_min_lr")
        mezo_min_lr = std::stof(next("--MeZO_min_lr"));
      else if (arg == "--epochs")
        epochs = static_cast<unsigned int>(std::stoul(next("--epochs")));
      else if (arg == "--output")
        output_path = next("--output");
      else if (arg == "--no_save_best")
        no_save_best = true;
      else if (arg == "--batch_size")
        batch_size_override =
          static_cast<unsigned int>(std::stoul(next("--batch_size")));
      else if (arg == "--max_samples")
        max_samples =
          static_cast<unsigned int>(std::stoul(next("--max_samples")));
      else if (arg == "--seq_len")
        seq_len_override =
          static_cast<unsigned int>(std::stoul(next("--seq_len")));
      else if (arg == "--seed")
        seed = static_cast<unsigned int>(std::stoul(next("--seed")));
      else if (arg == "--resume")
        resume_path = next("--resume");
      else if (arg == "--start_epoch")
        start_epoch =
          static_cast<unsigned int>(std::stoul(next("--start_epoch")));
      else if (arg == "--label_tokens") {
        std::stringstream ss(next("--label_tokens"));
        std::string tok;
        while (std::getline(ss, tok, ','))
          if (!tok.empty())
            label_tokens.push_back(
              static_cast<unsigned int>(std::stoul(tok)));
      } else if (arg == "--save_every")
        save_every =
          static_cast<unsigned int>(std::stoul(next("--save_every")));
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
    output_path = model_path + "/mezo_checkpoint.bin";

  try {
    causallm::json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    causallm::json generation_cfg = causallm::json::object();
    const std::string gen_cfg_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(gen_cfg_path))
      generation_cfg = causallm::LoadJsonFile(gen_cfg_path);
    causallm::json nntr_cfg =
      causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // Resolve the tokenizer path against the model directory, same fallback
    // train_qwen3_lora.cpp uses for on-device configs shipped with absolute
    // paths that don't exist on a build host.
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

    // This driver is full-parameter only: force LoRA off regardless of what
    // nntr_config.json says, so every layer stays trainable (see
    // Transformer::hasLoRA()) and MeZO perturbs the whole model.
    nntr_cfg["lora_rank"] = 0;

    /**
     * Batch size is the primary control on MeZO's estimator variance, so it is
     * worth overriding from the command line rather than only from the model's
     * config file.
     */
    if (batch_size_override)
      nntr_cfg["batch_size"] = batch_size_override;

    if (!label_tokens.empty())
      nntr_cfg["label_token_ids"] = label_tokens;

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
              << "train_data:   " << train_data_path << "\n"
              << "valid_data:   " << valid_data_path << "\n"
              << "seq_len:      " << seq_len << "\n"
              << "MeZO_lr:      " << mezo_lr << "\n"
              << "MeZO_epsilon: " << mezo_epsilon << "\n"
              << "MeZO_lr_decay: " << mezo_lr_decay << "\n"
              << "epochs:       " << epochs << "\n"
              << "output:       " << output_path << std::endl;

    causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
    model.initializeForTraining(
      mezo_lr, epochs, "MeZO",
      {nntrainer::withKey("MeZO_learning_rate", mezo_lr),
       nntrainer::withKey("MeZO_epsilon", mezo_epsilon),
       nntrainer::withKey("MeZO_lr_decay", mezo_lr_decay),
       nntrainer::withKey("MeZO_min_learning_rate", mezo_min_lr)});

    // A MeZO checkpoint is a plain full-weight file in the same format as the
    // base checkpoint, so resuming is just loading that instead. Nothing else
    // carries over because MeZO holds no optimizer state.
    const std::string weight_file =
      resume_path.empty()
        ? model_path + "/" + nntr_cfg["model_file_name"].get<std::string>()
        : resume_path;
    if (!resume_path.empty()) {
      if (!std::filesystem::exists(resume_path))
        throw std::runtime_error("resume checkpoint not found: " + resume_path);
      std::cout << "resuming from:  " << resume_path << std::endl;
    }
    // lora_rank is forced to 0 above, so the compiled graph contains no
    // loraA/loraB tensors and the ordinary positional loader lines up with
    // the checkpoint exactly. Deliberately NOT load_weight_lora(): that path
    // builds a throwaway LoRA-free copy of the whole model to load into and
    // then copies across by name, which transiently doubles resident weight
    // memory (~2.4GB extra for Qwen3-0.6B FP32) for no benefit here.
    model.load_weight(weight_file);

    auto *tokenizer = model.getTokenizer();
    if (tokenizer == nullptr)
      throw std::runtime_error(
        "model has no tokenizer; a tokenizer_file is required for training");

    // The graph's head is sliced to these, so labels must be the same width.
    const std::vector<int32_t> label_ids(model.getLabelTokenIds().begin(),
                                         model.getLabelTokenIds().end());
    if (!label_ids.empty())
      std::cout << "label tokens: " << label_ids.size()
                << " (objective is a " << label_ids.size()
                << "-way choice, not " << vocab_size << "-way)" << std::endl;

    causallm::TrainingDataGenerator train_gen(train_data_path, tokenizer,
                                              seq_len, vocab_size, max_samples,
                                              seed, label_ids);
    causallm::TrainingDataGenerator valid_gen(valid_data_path, tokenizer,
                                              seq_len, vocab_size,
                                              /*max_samples=*/0, seed,
                                              label_ids);
    std::cout << "train_samples: " << train_gen.size() << "\n"
              << "valid_samples: " << valid_gen.size() << std::endl;

    std::shared_ptr<ml::train::Dataset> train_dataset =
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::trainingDataGenCb, &train_gen);
    std::shared_ptr<ml::train::Dataset> valid_dataset =
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::trainingDataGenCb, &valid_gen);
    model.setDataset(ml::train::DatasetModeType::MODE_TRAIN, train_dataset);
    model.setDataset(ml::train::DatasetModeType::MODE_VALID, valid_dataset);

    std::filesystem::path op(output_path);
    const std::string latest_path =
      (op.parent_path() / (op.stem().string() + "_latest.bin")).string();
    const std::string csv_path =
      (op.parent_path() / (op.stem().string() + "_history.csv")).string();
    if (!std::filesystem::exists(csv_path)) {
      std::ofstream csv(csv_path);
      csv << "epoch,train_loss,valid_loss,accuracy\n";
    }
    std::cout << "latest ckpt:  " << latest_path << "\n"
              << "history:      " << csv_path << std::endl;

    EpochState state{&model,   output_path, latest_path,
                     csv_path, start_epoch,  save_every,
                     std::numeric_limits<float>::max(), no_save_best};
    model.train(onEpochComplete, &state);

    std::cout << "training complete; best valid_loss=" << state.best_loss
              << ", checkpoint at " << output_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
