// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   main.cpp
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MeZO optimizer application for MNIST training
 */

#include <algorithm>
#include <arpa/inet.h>
#include <climits>
#include <cmath>
#include <dataset.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <model.h>
#include <nntrainer_log.h>
#include <numeric>
#include <optimizer.h>
#include <random>
#include <util_func.h>
#include <vector>

using namespace nntrainer;

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @struct UserData
 * @brief User data structure for dataset callback
 */
struct UserData {
  std::vector<float> data;
  std::vector<float> labels;
  unsigned int num_samples;
  unsigned int current_index;
  unsigned int feature_size;
  unsigned int label_size;
  std::vector<unsigned int> indices;
  std::mt19937 rng;

  UserData(const std::vector<float> &d, const std::vector<float> &l,
           unsigned int fs, unsigned int ls) :
    data(d),
    labels(l),
    num_samples(d.size() / fs),
    current_index(0),
    feature_size(fs),
    label_size(ls),
    indices(num_samples),
    rng(0) {
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
  }
};

// Dataset callback function
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  UserData *ud = static_cast<UserData *>(user_data);

  if (ud->current_index >= ud->num_samples) {
    // Reset and shuffle indices for next epoch
    ud->current_index = 0;
    std::shuffle(ud->indices.begin(), ud->indices.end(), ud->rng);
  }

  unsigned int idx = ud->indices[ud->current_index];

  // Copy input data
  std::copy(ud->data.begin() + static_cast<size_t>(idx) * ud->feature_size,
            ud->data.begin() + static_cast<size_t>(idx + 1) * ud->feature_size,
            *outVec);

  // Copy label data
  std::copy(ud->labels.begin() + static_cast<size_t>(idx) * ud->label_size,
            ud->labels.begin() + static_cast<size_t>(idx + 1) * ud->label_size,
            *outLabel);

  ud->current_index++;

  *last = (ud->current_index >= ud->num_samples);

  return 0;
}

static uint32_t read_u32_be(std::ifstream &ifs) {
  uint32_t val = 0;
  ifs.read(reinterpret_cast<char *>(&val), sizeof(val));
  return ntohl(val);
}

// Function to load raw MNIST data
std::pair<std::vector<float>, std::vector<float>>
loadMNISTData(const std::string &data_file, const std::string &label_file) {

  std::vector<float> data;
  std::vector<float> labels;
  static constexpr float MNIST_MEAN = 0.1307f;
  static constexpr float MNIST_STD = 0.3081f;

  std::ifstream ifs_images(data_file, std::ios::binary);
  if (!ifs_images.is_open()) {
    ml_loge("Failed to open MNIST images file: %s", data_file.c_str());
    return {data, labels};
  }

  uint32_t magic_images = read_u32_be(ifs_images);
  uint32_t num_images = read_u32_be(ifs_images);
  uint32_t rows = read_u32_be(ifs_images);
  uint32_t cols = read_u32_be(ifs_images);

  if (magic_images != 2051) {
    ml_loge("Invalid MNIST images magic: %u", magic_images);
    return {data, labels};
  }

  std::ifstream ifs_labels(label_file, std::ios::binary);
  if (!ifs_labels.is_open()) {
    ml_loge("Failed to open MNIST labels file: %s", label_file.c_str());
    return {data, labels};
  }

  uint32_t magic_labels = read_u32_be(ifs_labels);
  uint32_t num_labels = read_u32_be(ifs_labels);

  if (magic_labels != 2049) {
    ml_loge("Invalid MNIST labels magic: %u", magic_labels);
    return {data, labels};
  }

  if (num_images != num_labels) {
    ml_loge("Images/labels count mismatch: %u vs %u", num_images, num_labels);
    return {data, labels};
  }

  size_t image_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  data.reserve((size_t)num_images * image_size);
  labels.reserve((size_t)num_images * 10); // 10 classes for one-hot encoding

  // Read images with normalization
  for (uint32_t i = 0; i < num_images; ++i) {
    std::vector<unsigned char> buf(image_size);
    ifs_images.read(reinterpret_cast<char *>(buf.data()), image_size);
    if (!ifs_images) {
      ml_loge("Error reading image %u", i);
      return {data, labels};
    }
    for (size_t p = 0; p < image_size; ++p) {
      // Normalize: (pixel / 255.0 - mean) / std
      float pixel = static_cast<float>(buf[p]) / 255.0f;
      float normalized = (pixel - MNIST_MEAN) / MNIST_STD;
      data.push_back(normalized);
    }
  }

  // Read labels and convert to one-hot encoding
  for (uint32_t i = 0; i < num_labels; ++i) {
    unsigned char lab = 0;
    ifs_labels.read(reinterpret_cast<char *>(&lab), 1);
    if (!ifs_labels) {
      ml_loge("Error reading label %u", i);
      return {data, labels};
    }

    // Convert to one-hot encoding
    for (int j = 0; j < 10; ++j) {
      labels.push_back(j == lab ? 1.0f : 0.0f);
    }
  }

  ml_logi("Loaded MNIST: %u samples, image=%ux%u, classes=%u", num_images, rows,
          cols, 10);

  return {data, labels};
}

std::vector<LayerHandle> createSimpleGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "1:1:784")}));

  // Hidden layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 256),
                 nntrainer::withKey("weight_initializer", "xavier_uniform"),
                 nntrainer::withKey("activation", "relu")}));

  // Output layer
  layers.push_back(
    createLayer("fully_connected",
                {
                  nntrainer::withKey("unit", 10),
                  nntrainer::withKey("weight_initializer", "xavier_uniform"),
                }));

  return layers;
}

int main(int argc, char *argv[]) {
  // Check command line arguments
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <data_folder_path>" << std::endl;
    std::cout << "Example: " << argv[0] << " /path/to/mnist/data/" << std::endl;
    return 0;
  }

  try {
    std::string data_folder = argv[1];
    // Ensure folder path ends with '/'
    if (data_folder.back() != '/') {
      data_folder += '/';
    }

    std::string train_data_file = data_folder + "train-images-idx3-ubyte";
    std::string train_label_file = data_folder + "train-labels-idx1-ubyte";
    std::string val_data_file = data_folder + "t10k-images-idx3-ubyte";
    std::string val_label_file = data_folder + "t10k-labels-idx1-ubyte";

    std::cout << "Loading MNIST training and validation data..." << std::endl;

    // Load MNIST data
    auto [train_data, train_labels] =
      loadMNISTData(train_data_file, train_label_file);
    auto [val_data, val_labels] = loadMNISTData(val_data_file, val_label_file);

    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "Training data size: " << train_data.size() << " ("
              << train_data.size() / 784 << " samples)" << std::endl;
    std::cout << "Training labels size: " << train_labels.size() << " ("
              << train_labels.size() / 10 << " samples)" << std::endl;
    std::cout << "Validation data size: " << val_data.size() << " ("
              << val_data.size() / 784 << " samples)" << std::endl;
    std::cout << "Validation labels size: " << val_labels.size() << " ("
              << val_labels.size() / 10 << " samples)" << std::endl;

    // Create user data for training and validation
    const unsigned int feature_size = 784;
    const unsigned int label_size = 10;

    auto train_user_data = std::make_unique<UserData>(train_data, train_labels,
                                                      feature_size, label_size);
    auto val_user_data = std::make_unique<UserData>(val_data, val_labels,
                                                    feature_size, label_size);

    // Create datasets
    std::shared_ptr<ml::train::Dataset> dataset_train =
      ml::train::createDataset(ml::train::DatasetType::GENERATOR, getSample,
                               train_user_data.get());
    std::shared_ptr<ml::train::Dataset> dataset_val = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, getSample, val_user_data.get());

    // Create model
    auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                        {nntrainer::withKey("loss", "mse")});

    // Add layers to the model
    auto layers = createSimpleGraph();
    for (auto &layer : layers) {
      model->addLayer(layer);
    }

    // Set the optimizer
    auto optimizer = ml::train::createOptimizer(
      "MeZO", {"MeZO_learning_rate=0.0001", "MeZO_epsilon=0.001"});
    model->setOptimizer(std::move(optimizer));

    // Set model properties
    model->setProperty({"epochs=100", "batch_size=32"});

    // Compile and initialize model

    model->compile();
    model->initialize();
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    // Set datasets
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);

    // Train the model
    model->train();

    std::cout << "Training completed!" << std::endl;
    std::cout << "Final training loss: " << model->getTrainingLoss()
              << std::endl;
    std::cout << "Final validation loss: " << model->getValidationLoss()
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
