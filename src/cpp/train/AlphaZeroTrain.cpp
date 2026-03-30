#include "AlphaZeroTrain.h"

#include <chrono>
#include <filesystem>
#include <iostream>

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
    }
}

std::string AlphaZeroTrainer::model_path() const {
    return cfg_.model_dir + "/alphazero_" +
           std::to_string(cfg_.rows) + "x" + std::to_string(cfg_.cols) + ".pt";
}

void AlphaZeroTrainer::train() {
    std::filesystem::create_directories(cfg_.model_dir);

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

    // Persistent optimizer
    // Note: We'll update the LR for each phase
    torch::optim::Adam optimizer(model_->parameters(),
                                 torch::optim::AdamOptions(cfg_.phases[0].lr));

    for (const auto& phase : cfg_.phases) {
        std::cout << "\n>>> STARTING PHASE: " << phase.name << " <<<" << std::endl;
        std::cout << "Config: sims=" << phase.mcts_sims 
                  << ", episodes=" << phase.episodes_per_iter
                  << ", lr=" << phase.lr << std::endl;

        // Update learning rate
        for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(phase.lr);
        }

        for (int iter = 0; iter < phase.iterations; iter++) {
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

            std::cout << "  Models saved to " << iter_path << " and " << model_path() << std::endl;
        }
    }
    std::cout << "\nAll training phases complete." << std::endl;
}

float AlphaZeroTrainer::train_epoch(torch::optim::Adam& optimizer) {
    auto batch = replay_buffer_.sample(cfg_.batch_size);
    if (batch.empty()) return 0.0f;

    std::vector<torch::Tensor> states, policies;
    std::vector<float> values;
    states.reserve(batch.size());
    policies.reserve(batch.size());
    values.reserve(batch.size());

    for (auto& sample : batch) {
        states.push_back(sample.state);
        policies.push_back(sample.policy);
        values.push_back(sample.value);
    }

    auto states_t = torch::stack(states).to(device_);
    auto policies_t = torch::stack(policies).to(device_);
    auto values_t = torch::tensor(values, torch::kFloat32).unsqueeze(1).to(device_);

    // Forward
    auto [logits, out_value] = model_->forward(states_t);

    // Loss
    auto v_loss = torch::mse_loss(out_value, values_t);
    auto log_probs = torch::log_softmax(logits, /*dim=*/1);
    auto p_loss = -(policies_t * log_probs).sum() / static_cast<float>(cfg_.batch_size);
    auto loss = v_loss + p_loss;

    // Backward
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    return loss.item<float>();
}

}  // namespace azb
