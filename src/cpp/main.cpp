/// AlphaZero Dots-and-Boxes Training — C++ Entry Point
///
/// Usage:
///   ./alphazero_train [options]
///
/// Options:
///   --rows N          Board rows (default: 3)
///   --cols N          Board cols (default: 3, same as rows if only --rows given)
///   --workers N       Self-play workers (default: 16)
///   --sims N          MCTS simulations per move (default: 400)
///   --iters N         Training iterations (default: 10)
///   --episodes N      Games per iteration (default: 100)
///   --batch N         Training batch size (default: 128)
///   --epochs N        Training epochs per iteration (default: 10)
///   --lr F            Learning rate (default: 0.001)
///   --hidden N        Hidden layer size (default: 256)
///   --blocks N        Residual blocks (default: 6)
///   --model-dir PATH  Model save directory (default: ../models)

#include <iostream>
#include <string>
#include <vector>

#include "AlphaZeroTrain.h"
#include "TrainConfig.h"

static void print_help() {
    std::cout << "AlphaZero Dots-and-Boxes Trainer (C++/LibTorch)\n"
              << "\nUsage: ./alphazero_train [options]\n"
              << "\nOptions:\n"
              << "  --rows N          Board rows (default: 3)\n"
              << "  --cols N          Board cols (default: same as rows)\n"
              << "  --workers N       Self-play workers (default: 16)\n"
              << "  --sims N          MCTS simulations (default: 400)\n"
              << "  --iters N         Training iterations (default: 10)\n"
              << "  --episodes N      Games per iteration (default: 100)\n"
              << "  --batch N         Training batch size (default: 128)\n"
              << "  --epochs N        Epochs per iteration (default: 10)\n"
              << "  --lr F            Learning rate (default: 0.001)\n"
              << "  --hidden N        Hidden size (default: 256)\n"
              << "  --blocks N        Residual blocks (default: 6)\n"
              << "  --model-dir PATH  Model directory (default: ../models)\n"
              << "  --help            Show this help\n"
              << "  --phased          Use phased training (default: false)\n";
}

int main(int argc, char* argv[]) {
    // LibTorch internal threads for matrix ops in inference.
    // Too many = starves workers; too few = slow inference.
    // Sweet spot: ~2 for a model with 6 res blocks.
    torch::set_num_threads(2);
    torch::set_num_interop_threads(1);

    azb::TrainConfig cfg;
    bool cols_set = false;
    bool phased = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
        if (arg == "--phased") {
            phased = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return 1;
        }
        std::string val = argv[++i];

        if      (arg == "--rows")      cfg.rows = std::stoi(val);
        else if (arg == "--cols")      { cfg.cols = std::stoi(val); cols_set = true; }
        else if (arg == "--workers")   cfg.num_workers = std::stoi(val);
        else if (arg == "--sims")      cfg.mcts_sims = std::stoi(val);
        else if (arg == "--iters")     cfg.num_iters = std::stoi(val);
        else if (arg == "--episodes")  cfg.episodes_per_iter = std::stoi(val);
        else if (arg == "--batch")     cfg.batch_size = std::stoi(val);
        else if (arg == "--epochs")    cfg.epochs = std::stoi(val);
        else if (arg == "--lr")        cfg.learning_rate = std::stod(val);
        else if (arg == "--hidden")    cfg.hidden_size = std::stoi(val);
        else if (arg == "--blocks")    cfg.num_res_blocks = std::stoi(val);
        else if (arg == "--model-dir") cfg.model_dir = val;
        else if (arg == "--grow")      cfg.buffer_grow = std::stoi(val);
        else if (arg == "--keep")      cfg.keep_checkpoints = std::stoi(val);
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    if (phased) {
        cfg.phases = azb::TrainConfig::get_default_phases();
    }

    if (!cols_set) cfg.cols = cfg.rows;

    std::cout << "=== AlphaZero Dots-and-Boxes Trainer ===" << std::endl;
    std::cout << "Board: " << cfg.rows << "x" << cfg.cols << " boxes" << std::endl;
    if (phased) {
        std::cout << "Mode: Phased Training (Bootstrap -> Refinement -> Mastery)" << std::endl;
    }
    std::cout << "Model: hidden=" << cfg.hidden_size << ", blocks=" << cfg.num_res_blocks << std::endl;

    azb::AlphaZeroTrainer trainer(cfg);
    trainer.train();

    return 0;
}
