#pragma once
/// AlphaZero Trainer — orchestrates self-play, inference, and training.

#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>
#include "AlphaZeroBitNet.h"
#include "InferenceServer.h"
#include "ReplayBuffer.h"
#include "SelfPlayWorker.h"
#include "TrainConfig.h"

namespace azb {

class AlphaZeroTrainer {
public:
    explicit AlphaZeroTrainer(const TrainConfig& cfg);

    /// Run the full training loop for cfg.num_iters iterations.
    void train();

private:
    /// Run one training epoch. Returns loss value.
    float train_epoch(torch::optim::Adam& optimizer);

    /// Get the model save path.
    std::string model_path() const;

    TrainConfig cfg_;
    torch::Device device_;
    AlphaZeroBitNet model_{nullptr};
    ReplayBuffer replay_buffer_;
};

}  // namespace azb
