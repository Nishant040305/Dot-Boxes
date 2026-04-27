#pragma once
/// Thread-safe replay buffer for training data.

#include <algorithm>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace azb {

/// A single training sample: (state_features, policy_target, legal_mask, value_target).
struct TrainSample {
    torch::Tensor state;       // shape: (input_size,)
    torch::Tensor policy;      // shape: (action_size,) — MCTS visit frequencies
    torch::Tensor legal_mask;  // shape: (action_size,) — 1=legal, 0=illegal
    float value = 0.0f;
};

/// Circular replay buffer with thread-safe push/sample.
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity) : capacity_(capacity) {}
    int capacity() const { return capacity_; }
    bool save(const std::string& path) const {
        std::vector<TrainSample> snapshot;
        int capacity_snapshot = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            capacity_snapshot = capacity_;
            snapshot.assign(buffer_.begin(), buffer_.end());
        }

        torch::serialize::OutputArchive archive;
        archive.write("capacity",
                      torch::tensor({static_cast<int64_t>(capacity_snapshot)}, torch::kInt64));
        archive.write("size",
                      torch::tensor({static_cast<int64_t>(snapshot.size())}, torch::kInt64));

        if (!snapshot.empty()) {
            std::vector<torch::Tensor> states, policies, masks;
            std::vector<float> values;
            states.reserve(snapshot.size());
            policies.reserve(snapshot.size());
            masks.reserve(snapshot.size());
            values.reserve(snapshot.size());
            for (const auto& sample : snapshot) {
                states.push_back(sample.state.cpu());
                policies.push_back(sample.policy.cpu());
                masks.push_back(sample.legal_mask.cpu());
                values.push_back(sample.value);
            }

            archive.write("states", torch::stack(states));
            archive.write("policies", torch::stack(policies));
            archive.write("masks", torch::stack(masks));
            archive.write("values", torch::tensor(values, torch::kFloat32));
        }

        try {
            archive.save_to(path);
        } catch (const std::exception& e) {
            std::cerr << "[ReplayBuffer] Failed to save: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    bool load(const std::string& path, int min_capacity = -1) {
        torch::serialize::InputArchive archive;
        try {
            archive.load_from(path);
        } catch (const std::exception& e) {
            std::cerr << "[ReplayBuffer] Failed to load: " << e.what() << std::endl;
            return false;
        }

        torch::Tensor cap_t;
        torch::Tensor size_t;
        try {
            archive.read("capacity", cap_t);
            archive.read("size", size_t);
        } catch (const std::exception& e) {
            std::cerr << "[ReplayBuffer] Missing metadata: " << e.what() << std::endl;
            return false;
        }

        const int64_t saved_capacity = cap_t.item<int64_t>();
        const int64_t saved_size = size_t.item<int64_t>();
        int target_capacity = static_cast<int>(saved_capacity);
        if (min_capacity > target_capacity) {
            target_capacity = min_capacity;
        }

        std::deque<TrainSample> new_buffer;
        if (saved_size > 0) {
            torch::Tensor states;
            torch::Tensor policies;
            torch::Tensor masks;
            torch::Tensor values;
            try {
                archive.read("states", states);
                archive.read("policies", policies);
                archive.read("masks", masks);
                archive.read("values", values);
            } catch (const std::exception& e) {
                std::cerr << "[ReplayBuffer] Missing tensors: " << e.what() << std::endl;
                return false;
            }

            if (states.size(0) != saved_size || policies.size(0) != saved_size ||
                masks.size(0) != saved_size || values.size(0) != saved_size) {
                std::cerr << "[ReplayBuffer] Tensor sizes do not match saved size."
                          << std::endl;
                return false;
            }

            const int64_t start =
                std::max<int64_t>(0, saved_size - static_cast<int64_t>(target_capacity));
            for (int64_t i = start; i < saved_size; i++) {
                TrainSample sample;
                sample.state = states[i].clone();
                sample.policy = policies[i].clone();
                sample.legal_mask = masks[i].clone();
                sample.value = values[i].item<float>();
                new_buffer.push_back(std::move(sample));
            }
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            capacity_ = target_capacity;
            buffer_ = std::move(new_buffer);
        }
        return true;
    }

    /// Resize the buffer while p1taining existing samples (if possible).
    void set_capacity(int new_capacity) {
        std::lock_guard<std::mutex> lock(mu_);
        if (new_capacity == capacity_) return;
        
        while (static_cast<int>(buffer_.size()) > new_capacity) {
            buffer_.pop_front();
        }
        capacity_ = new_capacity;
    }

    void push(TrainSample sample) {
        std::lock_guard<std::mutex> lock(mu_);
        if (static_cast<int>(buffer_.size()) >= capacity_) {
            buffer_.pop_front();
        }
        buffer_.push_back(std::move(sample));
    }

    /// Push a batch of samples at once (fewer lock acquisitions).
    void push_batch(std::vector<TrainSample>& samples) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : samples) {
            if (static_cast<int>(buffer_.size()) >= capacity_) {
                buffer_.pop_front();
            }
            buffer_.push_back(std::move(s));
        }
    }

    /// Sample a random mini-batch. Returns empty vector if not enough data.
    std::vector<TrainSample> sample(int batch_size) {
        std::lock_guard<std::mutex> lock(mu_);
        if (static_cast<int>(buffer_.size()) < batch_size) {
            return {};
        }
        // Fisher-Yates partial shuffle for O(batch_size) sampling
        std::vector<int> indices(buffer_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::uniform_int_distribution<int> dist;
        for (int i = 0; i < batch_size; i++) {
            dist = std::uniform_int_distribution<int>(i, static_cast<int>(indices.size()) - 1);
            std::swap(indices[i], indices[dist(rng_)]);
        }
        std::vector<TrainSample> result;
        result.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
            result.push_back(buffer_[indices[i]]);
        }
        return result;
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return static_cast<int>(buffer_.size());
    }

private:
    int capacity_;
    std::deque<TrainSample> buffer_;
    mutable std::mutex mu_;
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace azb
