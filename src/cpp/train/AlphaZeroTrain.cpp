#include "AlphaZeroTrain.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace azb {

AlphaZeroTrainer::AlphaZeroTrainer(const TrainConfig& cfg)
    : cfg_(cfg),
      device_(torch::kCPU),
      model_(cfg.rows, cfg.cols, cfg.hidden_size, cfg.num_res_blocks, cfg.dropout),
      replay_buffer_(cfg.buffer_capacity) {

    if (torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        std::cout << "[Trainer] Using CUDA" << std::endl;
    } else {
        std::cout << "[Trainer] Using CPU" << std::endl;
    }
    model_->to(device_);

    if (cfg_.resume) {
        const auto path = model_path();
        if (std::filesystem::exists(path)) {
            try {
                torch::load(model_, path);
                model_->to(device_);
                std::cout << "[Trainer] Loaded model from " << path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Trainer] Failed to load model: " << e.what()
                          << ", starting fresh." << std::endl;
            }
        } else {
            std::cerr << "[Trainer] Resume requested but no model found at "
                      << path << ". Starting fresh." << std::endl;
        }
    }
}

std::string AlphaZeroTrainer::model_path() const {
    return cfg_.model_dir + "/alphazero_" +
           std::to_string(cfg_.rows) + "x" + std::to_string(cfg_.cols) + ".pt";
}

void AlphaZeroTrainer::train() {
    std::filesystem::create_directories(cfg_.model_dir);

    // Write model architecture sidecar so Python can auto-detect hidden/blocks
    {
        std::ofstream info(cfg_.model_dir + "/model_info.json");
        info << "{\"rows\":" << cfg_.rows
             << ",\"cols\":" << cfg_.cols
             << ",\"hidden_size\":" << cfg_.hidden_size
             << ",\"num_res_blocks\":" << cfg_.num_res_blocks
             << ",\"value_eval\":\"" << azb::value_eval_name(cfg_.value_eval) << "\""
             << "}\n";
    }

    // If no phases provided, create a default one from existing config
    if (cfg_.phases.empty()) {
        Phase p;
        p.name = "Default";
        p.iterations = cfg_.num_iters;
        p.mcts_sims = cfg_.mcts_sims;
        p.episodes_per_iter = cfg_.episodes_per_iter;
        p.epochs = cfg_.epochs;
        p.lr = static_cast<float>(cfg_.learning_rate);
        p.temp_threshold = cfg_.temp_threshold;
        p.temp_explore = cfg_.temp_explore;
        p.temp_exploit = cfg_.temp_exploit;
        cfg_.phases.push_back(p);
    }

    int start_phase_idx = 0;
    int start_iter = 0;
    std::string state_path = cfg_.model_dir + "/train_state_" + std::to_string(cfg_.rows) + "x" + std::to_string(cfg_.cols) + ".txt";
    std::string optim_path = cfg_.model_dir + "/optimizer_state_" + std::to_string(cfg_.rows) + "x" + std::to_string(cfg_.cols) + ".pt";
    std::string replay_path = cfg_.model_dir + "/replay_buffer_" + std::to_string(cfg_.rows) + "x" + std::to_string(cfg_.cols) + ".pt";

    if (cfg_.resume) {
        if (std::filesystem::exists(state_path)) {
            std::ifstream ifs(state_path);
            if (ifs >> start_phase_idx >> start_iter) {
                std::cout << "[Trainer] Resuming from phase " << start_phase_idx << ", iter " << start_iter << std::endl;
            } else {
                std::cerr << "[Trainer] Failed to parse train state. Starting fresh." << std::endl;
                start_phase_idx = 0;
                start_iter = 0;
            }
        } else {
            std::cout << "[Trainer] No train state found to resume. Starting fresh." << std::endl;
        }
    }

    // Persistent optimizer
    // Note: We'll update the LR for each phase
    torch::optim::Adam optimizer(model_->parameters(),
                                 torch::optim::AdamOptions(cfg_.phases.empty() ? cfg_.learning_rate : cfg_.phases[0].lr));
    if (cfg_.resume && std::filesystem::exists(optim_path)) {
        try {
            torch::load(optimizer, optim_path);
            std::cout << "[Trainer] Loaded optimizer state from " << optim_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Trainer] Failed to load optimizer state: " << e.what()
                      << ", continuing with fresh optimizer." << std::endl;
        }
    } else if (cfg_.resume) {
        std::cout << "[Trainer] No optimizer state found to resume." << std::endl;
    }

    if (cfg_.resume && std::filesystem::exists(replay_path)) {
        if (replay_buffer_.load(replay_path, cfg_.buffer_capacity)) {
            std::cout << "[Trainer] Loaded replay buffer (" << replay_buffer_.size()
                      << " samples, capacity " << replay_buffer_.capacity() << ") from "
                      << replay_path << std::endl;
        } else {
            std::cerr << "[Trainer] Failed to load replay buffer. Starting fresh." << std::endl;
        }
    } else if (cfg_.resume) {
        std::cout << "[Trainer] No replay buffer found to resume." << std::endl;
    }

    for (size_t p_idx = start_phase_idx; p_idx < cfg_.phases.size(); p_idx++) {
        const auto& phase = cfg_.phases[p_idx];
        std::cout << "\n>>> STARTING PHASE: " << phase.name << " <<<" << std::endl;
        std::cout << "Config: sims=" << phase.mcts_sims 
                  << ", episodes=" << phase.episodes_per_iter
                  << ", lr=" << phase.lr << std::endl;

        // Update learning rate
        for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(phase.lr);
        }

        int iter_start_loop = (p_idx == static_cast<size_t>(start_phase_idx)) ? start_iter : 0;

        for (int iter = iter_start_loop; iter < phase.iterations; iter++) {
            std::cout << "\n=== [" << phase.name << "] Iteration " 
                      << (iter + 1) << "/" << phase.iterations << " ===" << std::endl;
            auto iter_start = std::chrono::steady_clock::now();

            // ── Self-Play Phase ──────────────────────────────────────────

            // Create inference model (copy of training model)
            AlphaZeroBitNet infer_model(cfg_.rows, cfg_.cols, cfg_.hidden_size,
                                        cfg_.num_res_blocks, cfg_.dropout);
            {
                torch::NoGradGuard no_grad;
                auto src = model_->named_parameters();
                for (auto& p : infer_model->named_parameters()) {
                    auto* found = src.find(p.key());
                    if (found) p.value().copy_(*found);
                }
                auto src_buf = model_->named_buffers();
                for (auto& b : infer_model->named_buffers()) {
                    auto* found = src_buf.find(b.key());
                    if (found) b.value().copy_(*found);
                }
            }
            infer_model->to(device_);
            infer_model->eval();

            InferenceServer server(infer_model, device_, cfg_.num_workers,
                                   cfg_.max_inference_batch);

            std::atomic<int> games_collected{0};
            std::atomic<bool> stop_workers{false};

            std::thread server_thread([&server]() { server.run(); });

            std::vector<std::unique_ptr<SelfPlayWorker>> workers;
            std::vector<std::thread> worker_threads;
            for (int w = 0; w < cfg_.num_workers; w++) {
                workers.push_back(std::make_unique<SelfPlayWorker>(
                    w, cfg_, phase, server, replay_buffer_, games_collected, stop_workers));
            }
            for (int w = 0; w < cfg_.num_workers; w++) {
                auto* worker_ptr = workers[w].get();
                worker_threads.emplace_back([worker_ptr]() { worker_ptr->run(); });
            }

            // Wait for enough games
            int last_reported = 0;
            while (games_collected.load() < phase.episodes_per_iter) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                int gc = games_collected.load();
                if (gc >= last_reported + 10) {
                    std::cout << "    Collected " << gc << "/" << phase.episodes_per_iter
                              << " games" << std::endl;
                    last_reported = gc;
                }
            }

            stop_workers.store(true);
            server.stop();
            for (auto& t : worker_threads) if (t.joinable()) t.join();
            if (server_thread.joinable()) server_thread.join();

            auto collect_end = std::chrono::steady_clock::now();
            double collect_secs = std::chrono::duration<double>(collect_end - iter_start).count();
            std::cout << "  Collection: " << collect_secs << "s, Buffer: " << replay_buffer_.size() << std::endl;

            // ── Training Phase ───────────────────────────────────────────

            if (replay_buffer_.size() < cfg_.batch_size) {
                std::cout << "  Wait for more samples..." << std::endl;
                continue;
            }

            std::cout << "  Training..." << std::endl;
            model_->train();
            float total_loss = 0.0f;
            for (int epoch = 0; epoch < phase.epochs; epoch++) {
                total_loss += train_epoch(optimizer);
            }
            std::cout << "  Avg Loss: " << total_loss / phase.epochs << std::endl;

            // ── Checkpointing & Adaptation ───────────────────────────────
            
            // Save iteration-specific model
            std::string iter_path = cfg_.model_dir + "/checkpoint_iter" + std::to_string(iter + 1) + ".pt";
            torch::save(model_, iter_path);
            
            // Overwrite latest main model
            torch::save(model_, model_path());
            
            // Register in history
            cfg_.model_history.push_back(iter_path);
            if (static_cast<int>(cfg_.model_history.size()) > cfg_.keep_checkpoints) {
                std::string to_remove = cfg_.model_history.front();
                cfg_.model_history.erase(cfg_.model_history.begin());
                if (std::filesystem::exists(to_remove)) {
                    std::filesystem::remove(to_remove);
                }
            }

            // Adaptive Buffer: GROW capacity
            if (cfg_.buffer_grow > 0) {
                int new_capacity = replay_buffer_.capacity() + cfg_.buffer_grow;
                std::cout << "  Buffer grown to: " << new_capacity << " samples" << std::endl;
                replay_buffer_.set_capacity(new_capacity);
            }

            int saved_iter = iter + 1;
            int saved_phase = p_idx;
            if (saved_iter >= phase.iterations) {
                saved_phase++;
                saved_iter = 0;
            }
            std::ofstream ofs(state_path);
            ofs << saved_phase << " " << saved_iter << "\n";
            ofs.close();
            try {
                torch::save(optimizer, optim_path);
            } catch (const std::exception& e) {
                std::cerr << "[Trainer] Failed to save optimizer state: " << e.what() << std::endl;
            }
            if (!replay_buffer_.save(replay_path)) {
                std::cerr << "[Trainer] Failed to save replay buffer." << std::endl;
            }

            std::cout << "  Models saved to " << iter_path << " and " << model_path() << "\n  State saved." << std::endl;
        }
    }
    std::cout << "\nAll training phases complete." << std::endl;
}

float AlphaZeroTrainer::train_epoch(torch::optim::Adam& optimizer) {
    auto batch = replay_buffer_.sample(cfg_.batch_size);
    if (batch.empty()) return 0.0f;

    std::vector<torch::Tensor> states, policies, masks;
    std::vector<float> values;
    states.reserve(batch.size());
    policies.reserve(batch.size());
    masks.reserve(batch.size());
    values.reserve(batch.size());

    for (auto& sample : batch) {
        states.push_back(sample.state);
        policies.push_back(sample.policy);
        masks.push_back(sample.legal_mask);
        values.push_back(sample.value);
    }

    auto states_t   = torch::stack(states).to(device_);
    auto policies_t = torch::stack(policies).to(device_);
    auto masks_t    = torch::stack(masks).to(device_);   // (batch, action_size), 1.0=legal
    auto values_t   = torch::tensor(values, torch::kFloat32).unsqueeze(1).to(device_);

    // Forward
    auto [logits, out_value] = model_->forward(states_t);

    // ── Value loss ───────────────────────────────────────────────────────
    // Value targets are exactly ±1 or 0. MSE is appropriate here.
    auto v_loss = torch::mse_loss(out_value, values_t);

    // ── Policy loss (masked cross-entropy) ───────────────────────────────
    // Set illegal-move logits to -1e9 BEFORE log_softmax.
    // This ensures the network's gradient on illegal moves is ~zero,
    // and the log_softmax denominator is not inflated by illegal actions.
    // Without this, every gradient step wastes capacity on impossible moves
    // and the CE loss underestimates legal-move probabilities.
    auto masked_logits = logits + (masks_t - 1.0f) * 1e9f;  // illegal → -1e9
    auto log_probs = torch::log_softmax(masked_logits, /*dim=*/1);
    // Standard AlphaZero cross-entropy: -sum(pi * log p) / batch
    auto p_loss = -(policies_t * log_probs).sum() / static_cast<float>(cfg_.batch_size);

    // Combine: weight value loss x1.5 to give value head sufficient signal
    // relative to policy head (policy has more outputs = larger raw gradient)
    auto loss = 1.5f * v_loss + p_loss;

    // Backward
    optimizer.zero_grad();
    loss.backward();
    // Gradient clipping for stability
    torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0f);
    optimizer.step();

    return loss.item<float>();
}


}  // namespace azb
