#pragma once
/// Thread-safe replay buffer for training data.

#include <deque>
#include <mutex>
#include <random>
#include <vector>

#include <torch/torch.h>

namespace azb {

/// A single training sample: (state_features, policy_target, value_target).
struct TrainSample {
    torch::Tensor state;   // shape: (input_size,)
    torch::Tensor policy;  // shape: (action_size,)
    float value = 0.0f;
};

/// Circular replay buffer with thread-safe push/sample.
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity) : capacity_(capacity) {}
    int capacity() const { return capacity_; }

    /// Resize the buffer while maintaining existing samples (if possible).
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
