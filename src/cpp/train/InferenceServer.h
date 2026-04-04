#pragma once
/// Batched inference server that runs in a dedicated thread.
/// Worker threads send StateSnapshots, this thread batches them and runs the model.

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>
#include "BitBoardEnv.h"
#include "AlphaZeroBitNet.h"

namespace azb {

/// Thrown when the inference server is shutting down.
struct ShutdownException : public std::exception {
    const char* what() const noexcept override { return "Inference server stopped"; }
};

/// Request from a worker to the inference server.
struct InferenceRequest {
    uint64_t request_id = 0;
    int worker_id = -1;
    StateSnapshot state;
};

/// Response from the inference server to a worker.
struct InferenceResponse {
    std::vector<float> policy;
    float value = 0.0f;
};

/// Batched inference server — runs model on a dedicated thread.
class InferenceServer {
public:
    InferenceServer(AlphaZeroBitNet& model, torch::Device device, int num_workers,
                    int max_batch_size = 16)
        : model_(model), device_(device), max_batch_(max_batch_size) {
        (void)num_workers;
    }

    /// Submit a request (called by worker threads). Throws ShutdownException on stop.
    uint64_t submit(int worker_id, const StateSnapshot& state) {
        if (stopped_.load()) throw ShutdownException();
        const uint64_t request_id = next_id_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(req_mu_);
            requests_.push({request_id, worker_id, state});
        }
        req_cv_.notify_one();
        return request_id;
    }

    /// Try to fetch a response without blocking. Returns true if ready.
    bool try_get(uint64_t request_id, InferenceResponse& out) {
        std::lock_guard<std::mutex> lock(resp_mu_);
        auto it = responses_.find(request_id);
        if (it == responses_.end()) return false;
        out = std::move(it->second);
        responses_.erase(it);
        return true;
    }

    /// Run the inference loop (call from dedicated thread). Blocks until stop() is called.
    void run() {
        while (!stopped_.load()) {
            std::vector<InferenceRequest> batch;

            {
                std::unique_lock<std::mutex> lock(req_mu_);
                // Wait until at least one request arrives (or shutdown)
                req_cv_.wait(lock, [&] { return !requests_.empty() || stopped_.load(); });
                if (stopped_.load() && requests_.empty()) break;

                // Grab the first request
                batch.push_back(std::move(requests_.front()));
                requests_.pop();
            }

            // Brief spin-wait to accumulate more requests into the batch.
            // This is critical: without this, we process batches of size 1,
            // wasting parallelism. With this, workers have time to submit
            // their requests before we run inference.
            auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            while (static_cast<int>(batch.size()) < max_batch_) {
                std::unique_lock<std::mutex> lock(req_mu_);
                if (requests_.empty()) {
                    // Wait briefly for more
                    auto now = std::chrono::steady_clock::now();
                    if (now >= deadline) break;
                    req_cv_.wait_until(lock, deadline,
                        [&] { return !requests_.empty() || stopped_.load(); });
                }
                // Drain whatever is available
                while (!requests_.empty() && static_cast<int>(batch.size()) < max_batch_) {
                    batch.push_back(std::move(requests_.front()));
                    requests_.pop();
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    break;
                };
            }

            if (batch.empty()) continue;
            process_batch(batch);
        }
    }

    void stop() {
        stopped_.store(true);
        req_cv_.notify_all();
    }

    /// Reload model weights from file.
    void reload_weights(const std::string& path) {
        try {
            torch::load(model_, path);
            model_->to(device_);
            model_->eval();
            std::cout << "[InferenceServer] Reloaded model from " << path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[InferenceServer] Failed to reload: " << e.what() << std::endl;
        }
    }

private:
    void process_batch(const std::vector<InferenceRequest>& batch) {
        const int bs = static_cast<int>(batch.size());

        // Build batch input
        std::vector<torch::Tensor> tensors;
        tensors.reserve(bs);
        for (const auto& req : batch) {
            tensors.push_back(model_->preprocess(req.state));
        }
        auto input = torch::stack(tensors).to(device_);

        // Inference
        torch::NoGradGuard no_grad;
        auto [logits, values] = model_->forward(input);

        // Apply softmax to get probabilities
        auto probs = torch::softmax(logits, /*dim=*/1).cpu();
        auto vals = values.cpu();

        // Dispatch results to workers
        for (int i = 0; i < bs; i++) {
            const int wid = batch[i].worker_id;
            auto prob_acc = probs[i];
            std::vector<float> policy(prob_acc.data_ptr<float>(),
                                      prob_acc.data_ptr<float>() + prob_acc.numel());
            float value = vals[i].item<float>();

            (void)wid;
            std::lock_guard<std::mutex> lock(resp_mu_);
            responses_[batch[i].request_id] = {std::move(policy), value};
        }
    }

    AlphaZeroBitNet& model_;
    torch::Device device_;
    int max_batch_;

    // Request queue
    std::queue<InferenceRequest> requests_;
    std::mutex req_mu_;
    std::condition_variable req_cv_;

    // Completed responses
    std::unordered_map<uint64_t, InferenceResponse> responses_;
    std::mutex resp_mu_;
    std::atomic<uint64_t> next_id_{1};

    std::atomic<bool> stopped_{false};
};

}  // namespace azb
