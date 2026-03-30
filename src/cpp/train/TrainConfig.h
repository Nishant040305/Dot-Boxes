#pragma once
/// Training configuration for AlphaZero.
/// All hyperparameters in one place for easy tuning.

#include <string>
#include <vector>

namespace azb {

struct Phase {
    std::string name;
    int iterations = 10;
    int mcts_sims = 400;
    int episodes_per_iter = 100;
    int epochs = 10;
    float lr = 0.001f;
    int temp_threshold = 8;
    float temp_explore = 1.0f;
    float temp_exploit = 0.3f;
};

struct TrainConfig {
    // Board dimensions (rows x cols boxes)
    int rows = 3;
    int cols = 3;

    // Neural network
    int hidden_size       = 256;
    int num_res_blocks    = 6;
    double dropout        = 0.1;

    // Phase management
    std::vector<Phase> phases;

    // Self-play (defaults if not in phase)
    int num_workers       = 64;
    int mcts_sims         = 400;
    int episodes_per_iter = 100;

    // Training
    int batch_size        = 128;
    int epochs            = 10;
    double learning_rate  = 0.001;
    int buffer_capacity   = 50000;
    int buffer_grow       = 0;      // Grow buffer by this much per iteration
    int num_iters         = 10;

    // Model history
    int keep_checkpoints  = 5;      // Keep last N checkpoints
    std::vector<std::string> model_history;

    // MCTS
    float c_puct            = 1.5f;
    float dirichlet_alpha   = 0.3f;
    float dirichlet_epsilon = 0.25f;
    float fpu_reduction     = 0.25f;

    // Temperature schedule
    int temp_threshold    = 10;   // moves < this use temp=1, else temp=0
    float temp_explore    = 1.0f;
    float temp_exploit    = 0.3f;

    // Paths
    std::string model_name = "alphazero";
    std::string model_dir  = "../models";

    // Inference batching
    int max_inference_batch = 64;

    /// Helper to get default phases
    static std::vector<Phase> get_default_phases() {
        return {
            {"Bootstrap", 15, 200, 100, 10, 0.002f, 8, 1.0f, 0.3f},
            {"Refinement", 20, 400, 100, 15, 0.001f, 6, 0.8f, 0.1f},
            {"Mastery", 15, 600, 100, 20, 0.0003f, 4, 0.8f, 0.0f}
        };
    }
};

}  // namespace azb
