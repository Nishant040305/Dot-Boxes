#pragma once
/// AlphaZero Trainer — orchestrates self-play, inference, and training.
/// Supports both standard AlphaZeroBitNet and hierarchical PatchNet via ModelHandle.

#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>
#include "AlphaZeroBitNet.h"
#include "ModelHandle.h"
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
    struct TrainLossStats {
        float loss = 0.0f;
        float value_loss = 0.0f;
        float policy_loss = 0.0f;
        int batches = 0;
    };

    /// Run one training epoch. Returns average losses over the epoch.
    /// If batch_debug_csv is non-null, writes policy/legality diagnostics for this epoch.
    TrainLossStats train_epoch(torch::optim::Adam& optimizer,
                               std::ostream* batch_debug_csv,
                               size_t phase_idx,
                               const std::string& phase_name,
                               int iter_1_based,
                               int epoch_1_based,
                               int epochs_total);

    /// Get the model save path.
    std::string model_path() const;

    TrainConfig cfg_;
    torch::Device device_;
    ModelHandle model_;
    ReplayBuffer replay_buffer_;
};

}  // namespace azb
