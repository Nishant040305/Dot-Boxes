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
///   --config NAME     Load predefined config (2x2, 3x3, 4x3, 5x5, 5x5_patch, 7x8)

#include <iostream>
#include <string>
#include <vector>

#include "AlphaZeroTrain.h"
#include "TrainConfig.h"

// Include predefined config builder functions
#include "training_config/_4x3.cpp"
#include "training_config/_5x5.cpp"
#include "training_config/_5x5_patch.cpp"
#include "training_config/_7x8.cpp"
#include "training_config/_7x8_patch.cpp"
#include "training_config/_3x3.cpp"
#include "training_config/_2x2.cpp"

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
               << "  --config NAME     Load predefined config (2x2, 3x3, 4x3, 5x5, 5x5_patch, 7x8, 7x8_patch)\n"
               << "  --value-eval V    Value target eval (winloss, score, score_sqrt, score_tanh)\n"
               << "  --dag            Enable DAG transpositions (default: on)\n"
               << "  --no-dag         Disable DAG (use tree only)\n"
               << "  --help            Show this help\n"
              << "  --phased          Use phased training (default: false)\n"
              << "  --resume          Resume from last saved iteration state\n"
              << "\nPatchNet (hierarchical) options:\n"
              << "  --patch           Enable PatchNet mode\n"
              << "  --patch-rows N    Local patch rows (default: 3)\n"
              << "  --patch-cols N    Local patch cols (default: 3)\n"
              << "  --local-model P   Path to pre-trained local model weights\n"
              << "  --local-hidden N  Local model hidden size (default: 128)\n"
              << "  --local-blocks N  Local model residual blocks (default: 6)\n"
              << "  --global-hidden N Global aggregator hidden size (default: 192)\n"
              << "  --global-blocks N Global aggregator residual blocks (default: 4)\n";
}

int main(int argc, char* argv[]) {
    // LibTorch internal threads for matrix ops in inference.
    // Tuned for i7-13700H: 4 intra-op threads for DNNL/OpenMP matrix ops,
    // leaving remaining cores for the 16 MCTS self-play workers.
    torch::set_num_threads(4);
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
        if (arg == "--dag") {
            cfg.use_dag = true;
            continue;
        }
        if (arg == "--no-dag") {
            cfg.use_dag = false;
            continue;
        }
        if (arg == "--resume") {
            cfg.resume = true;
            continue;
        }
        if (arg == "--patch") {
            cfg.use_patch_net = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return 1;
        }
        std::string val = argv[++i];

        if      (arg == "--config") {
            const bool resume_flag = cfg.resume;
            const bool use_dag_flag = cfg.use_dag;
            azb::TrainConfig loaded;
            if (val == "4x3") loaded = make_4x3_config();
            else if (val == "5x5") loaded = make_5x5_config();
            else if (val == "5x5_patch") loaded = make_5x5_patch_config();
            else if (val == "7x8") loaded = make_7x8_config();
            else if (val == "7x8_patch") loaded = make_7x8_patch_config();
            else if (val == "3x3") loaded = make_3x3_config();
            else if (val == "2x2") loaded = make_2x2_config();
            else {
                std::cerr << "Unknown config: " << val << std::endl;
                return 1;
            }
            loaded.resume = resume_flag;
            loaded.use_dag = use_dag_flag;
            cfg = std::move(loaded);
            cols_set = true;
        }
        else if (arg == "--rows")      cfg.rows = std::stoi(val);
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
        else if (arg == "--value-eval") {
            azb::ValueEval eval;
            if (!azb::parse_value_eval(val, eval)) {
                std::cerr << "Unknown value eval: " << val << std::endl;
                return 1;
            }
            cfg.value_eval = eval;
        }
        else if (arg == "--grow")      cfg.buffer_grow = std::stoi(val);
        else if (arg == "--keep")      cfg.keep_checkpoints = std::stoi(val);
        // PatchNet-specific flags
        else if (arg == "--patch-rows")    cfg.patch_rows = std::stoi(val);
        else if (arg == "--patch-cols")    cfg.patch_cols = std::stoi(val);
        else if (arg == "--local-model")   cfg.local_model_path = val;
        else if (arg == "--local-hidden")  cfg.local_hidden_size = std::stoi(val);
        else if (arg == "--local-blocks")  cfg.local_num_res_blocks = std::stoi(val);
        else if (arg == "--global-hidden") cfg.global_hidden_size = std::stoi(val);
        else if (arg == "--global-blocks") cfg.global_num_res_blocks = std::stoi(val);
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
    if (cfg.use_patch_net) {
        std::cout << "Mode: PatchNet (local " << cfg.patch_rows << "x" << cfg.patch_cols
                  << " → global " << cfg.rows << "x" << cfg.cols << ")" << std::endl;
        std::cout << "Local model: " << cfg.local_model_path
                  << " (" << cfg.local_hidden_size << "h, "
                  << cfg.local_num_res_blocks << " blocks, FROZEN)" << std::endl;
        std::cout << "Global aggregator: " << cfg.global_hidden_size << "h, "
                  << cfg.global_num_res_blocks << " blocks" << std::endl;
    } else {
        if (phased) {
            std::cout << "Mode: Phased Training (Bootstrap -> Refinement -> Mastery)" << std::endl;
        }
        std::cout << "Model: hidden=" << cfg.hidden_size << ", blocks=" << cfg.num_res_blocks << std::endl;
    }
    std::cout << "Value eval: " << azb::value_eval_name(cfg.value_eval) << std::endl;

    azb::AlphaZeroTrainer trainer(cfg);
    trainer.train();

    return 0;
}
