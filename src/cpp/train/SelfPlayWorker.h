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
#include "SymmetryAugmentation.h"
#include "TrainConfig.h"

namespace azb {

/// Async policy-value implementation that delegates to the InferenceServer.
class RemotePolicy : public AsyncPolicyValueFn {
public:
    RemotePolicy(InferenceServer& server, int worker_id)
        : server_(server), worker_id_(worker_id) {}

    uint64_t submit(const StateSnapshot& state) override {
        return server_.submit(worker_id_, state);
    }

    bool try_get(uint64_t request_id, PolicyValue& out) override {
        InferenceResponse resp;
        if (!server_.try_get(request_id, resp)) return false;
        out = {std::move(resp.policy), resp.value};
        return true;
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
          games_count_(games_count), stop_(stop_flag),
          geo_(cfg.rows, cfg.cols),
          sym_transforms_(build_transforms(geo_)) {}

    void run() {
        RemotePolicy remote(server_, worker_id_);
        BitBoardEnv env(cfg_.rows, cfg_.cols);
        AlphaZeroBitAgent agent(env, remote, phase_.mcts_sims, cfg_.c_puct,
                                cfg_.dirichlet_alpha, cfg_.dirichlet_epsilon,
                                cfg_.fpu_reduction, true, cfg_.use_dag,
                                cfg_.value_eval);

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

            const int total_boxes = cfg_.rows * cfg_.cols;
            const int action_size = env.action_size();
            std::vector<TrainSample> samples;
            samples.reserve(game_history.size() * (1 + sym_transforms_.size()));

            for (const auto& item : game_history) {
                // Feature tensor via static preprocess
                auto state_tensor = AlphaZeroBitNetImpl::preprocess(
                    item.snapshot, cfg_.rows, cfg_.cols);

                // Build policy tensor (MCTS visit counts, normalised)
                auto policy_tensor = torch::zeros({action_size}, torch::kFloat32);
                auto legal_mask = torch::zeros({action_size}, torch::kFloat32);
                auto pacc = policy_tensor.accessor<float, 1>();
                auto macc = legal_mask.accessor<float, 1>();
                int total_visits = 0;
                for (const auto& [idx, v] : item.visits) total_visits += v;
                if (total_visits > 0) {
                    for (const auto& [idx, v] : item.visits) {
                        pacc[idx] = static_cast<float>(v) / total_visits;
                        macc[idx] = 1.0f;
                    }
                }

                float z = value_from_scores(cfg_.value_eval, item.player,
                                            env.score_p1(), env.score_p2(),
                                            total_boxes);

                samples.push_back({std::move(state_tensor),
                                   std::move(policy_tensor),
                                   std::move(legal_mask), z});

                // ── Symmetry Augmentation ─────────────────────────────────
                // Delegates to SymmetryAugmentation.h — generates all
                // valid symmetry transforms (2 for rectangular, 7 for square).
                augment_position(sym_transforms_, geo_, item.snapshot,
                                 item.visits, total_visits, z, samples);
            }

            buffer_.push_batch(samples);
            games_count_.fetch_add(1);
        }  // while !stop
    }  // run()

private:
    int worker_id_;
    const TrainConfig& cfg_;
    const Phase& phase_;
    InferenceServer& server_;
    ReplayBuffer& buffer_;
    std::atomic<int>& games_count_;
    std::atomic<bool>& stop_;

    // Pre-computed board geometry and symmetry transforms
    BoardGeometry geo_;
    std::vector<SymmetryTransform> sym_transforms_;
};

}  // namespace azb
