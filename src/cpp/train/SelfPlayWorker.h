#pragma once
/// Self-play worker — plays complete games using AlphaZeroBitAgent
/// and submits training data to the replay buffer.

#include <atomic>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <torch/torch.h>
#include "AlphaZeroBitAgent.h"
#include "AlphaZeroBitNet.h"
#include "BitBoardEnv.h"
#include "InferenceServer.h"
#include "ReplayBuffer.h"
#include "TrainConfig.h"

namespace azb {

/// PolicyValueFn implementation that delegates to the InferenceServer.
class RemotePolicy : public PolicyValueFn {
public:
    RemotePolicy(InferenceServer& server, int worker_id)
        : server_(server), worker_id_(worker_id) {}

    PolicyValue operator()(const StateSnapshot& state) override {
        auto resp = server_.infer(worker_id_, state);
        return {std::move(resp.policy), resp.value};
    }

private:
    InferenceServer& server_;
    int worker_id_;
};

/// Runs self-play games and pushes training samples to the replay buffer.
/// Designed to run in its own std::thread.
class SelfPlayWorker {
public:
    SelfPlayWorker(int worker_id, const TrainConfig& cfg, const Phase& phase,
                   InferenceServer& server, ReplayBuffer& buffer,
                   std::atomic<int>& games_count, std::atomic<bool>& stop_flag)
        : worker_id_(worker_id), cfg_(cfg), phase_(phase),
          server_(server), buffer_(buffer),
          games_count_(games_count), stop_(stop_flag) {}

    void run() {
        RemotePolicy remote(server_, worker_id_);
        BitBoardEnv env(cfg_.rows, cfg_.cols);
        AlphaZeroBitAgent agent(env, remote, phase_.mcts_sims, cfg_.c_puct,
                                cfg_.dirichlet_alpha, cfg_.dirichlet_epsilon,
                                cfg_.fpu_reduction, true);

        // We need a separate AlphaZeroBitNet just for preprocess() calls.
        // Since preprocess is a pure function of the snapshot, a local lightweight
        // copy (never trained) is fine.
        AlphaZeroBitNet preprocess_net(cfg_.rows, cfg_.cols, /*hidden*/ 8, /*blocks*/ 1);
        preprocess_net->eval();

        while (!stop_.load()) {
            env.reset();
            struct HistoryItem {
                StateSnapshot snapshot;
                int player;
                std::unordered_map<uint32_t, int> visits;
            };
            std::vector<HistoryItem> game_history;

            int move_count = 0;
            try {
                while (!env.done() && !stop_.load()) {
                    StateSnapshot snap = env.get_state_snapshot(Action{-1, -1, -1});
                    float temp = (move_count < phase_.temp_threshold) 
                                 ? phase_.temp_explore : phase_.temp_exploit;

                    Action action = agent.act(env, /*return_probs=*/true, temp);
                    auto visits = agent.last_visit_counts();

                    game_history.push_back({std::move(snap), env.current_player(), std::move(visits)});
                    env.step(action);
                    move_count++;
                }
            } catch (const ShutdownException&) {
                break;
            }

            if (stop_.load()) break;

            // Determine result
            float result = 0.0f;
            if (env.score_p1() > env.score_p2())      result = 1.0f;
            else if (env.score_p2() > env.score_p1()) result = -1.0f;

            // Convert game history to training samples
            const int action_size = env.action_size();
            std::vector<TrainSample> samples;
            samples.reserve(game_history.size());

            for (const auto& item : game_history) {
                auto state_tensor = preprocess_net->preprocess(item.snapshot);

                auto policy_tensor = torch::zeros({action_size}, torch::kFloat32);
                auto pacc = policy_tensor.accessor<float, 1>();
                int total_visits = 0;
                for (const auto& [idx, v] : item.visits) total_visits += v;
                if (total_visits > 0) {
                    for (const auto& [idx, v] : item.visits) {
                        pacc[idx] = static_cast<float>(v) / total_visits;
                    }
                }

                float z = (item.player == 1) ? result : -result;
                samples.push_back({std::move(state_tensor), std::move(policy_tensor), z});
            }

            buffer_.push_batch(samples);
            games_count_.fetch_add(1);
        }
    }

private:
    int worker_id_;
    const TrainConfig& cfg_;
    const Phase& phase_;
    InferenceServer& server_;
    ReplayBuffer& buffer_;
    std::atomic<int>& games_count_;
    std::atomic<bool>& stop_;
};

}  // namespace azb
