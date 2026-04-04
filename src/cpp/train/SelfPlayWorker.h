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
          games_count_(games_count), stop_(stop_flag) {}

    void run() {
        RemotePolicy remote(server_, worker_id_);
        BitBoardEnv env(cfg_.rows, cfg_.cols);
        AlphaZeroBitAgent agent(env, remote, phase_.mcts_sims, cfg_.c_puct,
                                cfg_.dirichlet_alpha, cfg_.dirichlet_epsilon,
                                cfg_.fpu_reduction, true);

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
                // Feature tensor via static preprocess — no model instance needed
                auto state_tensor = AlphaZeroBitNetImpl::preprocess(item.snapshot, cfg_.rows, cfg_.cols);

                // Build policy tensor (MCTS visit counts, normalised)
                auto policy_tensor = torch::zeros({action_size}, torch::kFloat32);
                // Build legal move mask (1=legal, 0=illegal) for masked cross-entropy loss
                auto legal_mask = torch::zeros({action_size}, torch::kFloat32);
                auto pacc = policy_tensor.accessor<float, 1>();
                auto macc = legal_mask.accessor<float, 1>();
                int total_visits = 0;
                for (const auto& [idx, v] : item.visits) total_visits += v;
                if (total_visits > 0) {
                    for (const auto& [idx, v] : item.visits) {
                        pacc[idx] = static_cast<float>(v) / total_visits;
                        macc[idx] = 1.0f;  // this action was legal (it got visits)
                    }
                }

                float z = (item.player == 1) ? result : -result;
                // z must be exactly ±1 or 0 — never a raw score difference.
                // The value head uses tanh output in [-1,+1]; training against
                // unnormalized scores produces enormous MSE gradients and makes
                // the value head saturate without learning game structure.
                samples.push_back({std::move(state_tensor), std::move(policy_tensor),
                                   std::move(legal_mask), z});

                // ── Symmetry Augmentation ─────────────────────────────────────
                // A Dots-and-Boxes board is symmetric under horizontal flip
                // (left↔right) and vertical flip (top↔bottom).
                // Each game position generates additional training samples for
                // free — effectively multiplying data 4× with zero extra NN calls.

                // Helper: flip h-edge bit index left↔right
                //   h-edge (r,c) → (r, cols-1-c)
                //   linear: r*(cols) + c  →  r*(cols) + (cols-1-c)
                auto flip_h_idx = [&](size_t i, bool /*h_edge*/) -> size_t {
                    const int r = static_cast<int>(i) / cfg_.cols;
                    const int c = static_cast<int>(i) % cfg_.cols;
                    return static_cast<size_t>(r * cfg_.cols + (cfg_.cols - 1 - c));
                };
                // Helper: flip v-edge bit index left↔right
                //   v-edge (r,c) → (r, cols-c)   [cols+1 columns]
                auto flip_v_hflip_idx = [&](size_t i) -> size_t {
                    const int cols1 = cfg_.cols + 1;
                    const int r = static_cast<int>(i) / cols1;
                    const int c = static_cast<int>(i) % cols1;
                    return static_cast<size_t>(r * cols1 + (cfg_.cols - c));
                };
                // Helper: flip h-edge bit index top↔bottom
                //   h-edge (r,c) → (rows-r, c)   [rows+1 rows]
                auto flip_h_vflip_idx = [&](size_t i) -> size_t {
                    const int r = static_cast<int>(i) / cfg_.cols;
                    const int c = static_cast<int>(i) % cfg_.cols;
                    return static_cast<size_t>((cfg_.rows - r) * cfg_.cols + c);
                };
                // Helper: flip v-edge bit index top↔bottom
                //   v-edge (r,c) → (rows-1-r, c)
                auto flip_v_vflip_idx = [&](size_t i) -> size_t {
                    const int cols1 = cfg_.cols + 1;
                    const int r = static_cast<int>(i) / cols1;
                    const int c = static_cast<int>(i) % cols1;
                    return static_cast<size_t>((cfg_.rows - 1 - r) * cols1 + c);
                };
                // Helper: flip box bit index left↔right
                auto flip_box_hflip = [&](size_t i) -> size_t {
                    const int r = static_cast<int>(i) / cfg_.cols;
                    const int c = static_cast<int>(i) % cfg_.cols;
                    return static_cast<size_t>(r * cfg_.cols + (cfg_.cols - 1 - c));
                };
                // Helper: flip box bit index top↔bottom
                auto flip_box_vflip = [&](size_t i) -> size_t {
                    const int r = static_cast<int>(i) / cfg_.cols;
                    const int c = static_cast<int>(i) % cfg_.cols;
                    return static_cast<size_t>((cfg_.rows - 1 - r) * cfg_.cols + c);
                };

                const int n_h_e = (cfg_.rows + 1) * cfg_.cols;
                const int n_v_e = cfg_.rows * (cfg_.cols + 1);

                // Lambda: apply a pair of index-remapping functions to build a
                // flipped StateSnapshot + (policy, legal_mask) pair.
                auto make_augmented = [&](
                    auto h_remap, auto v_remap, auto box_remap,
                    auto policy_h_remap, auto policy_v_remap
                ) -> TrainSample {
                    // Build flipped StateSnapshot
                    StateSnapshot flipped_snap;
                    flipped_snap.h_edges    = azb::FastBitset(n_h_e);
                    flipped_snap.v_edges    = azb::FastBitset(n_v_e);
                    flipped_snap.boxes_p1   = azb::FastBitset(cfg_.rows * cfg_.cols);
                    flipped_snap.boxes_p2   = azb::FastBitset(cfg_.rows * cfg_.cols);
                    flipped_snap.current_player = item.snapshot.current_player;
                    flipped_snap.done       = item.snapshot.done;
                    flipped_snap.score_p1   = item.snapshot.score_p1;
                    flipped_snap.score_p2   = item.snapshot.score_p2;
                    flipped_snap.last_action = item.snapshot.last_action;

                    item.snapshot.h_edges.for_each_set_bit([&](size_t i) {
                        flipped_snap.h_edges.set(h_remap(i));
                    });
                    item.snapshot.v_edges.for_each_set_bit([&](size_t i) {
                        flipped_snap.v_edges.set(v_remap(i));
                    });
                    item.snapshot.boxes_p1.for_each_set_bit([&](size_t i) {
                        flipped_snap.boxes_p1.set(box_remap(i));
                    });
                    item.snapshot.boxes_p2.for_each_set_bit([&](size_t i) {
                        flipped_snap.boxes_p2.set(box_remap(i));
                    });

                    // Build flipped policy + mask
                    auto flipped_policy = torch::zeros({action_size}, torch::kFloat32);
                    auto flipped_mask   = torch::zeros({action_size}, torch::kFloat32);
                    auto fp_acc = flipped_policy.accessor<float, 1>();
                    auto fm_acc = flipped_mask.accessor<float, 1>();
                    for (const auto& [idx, v] : item.visits) {
                        int new_idx;
                        if (static_cast<int>(idx) < n_h_e) {
                            new_idx = static_cast<int>(policy_h_remap(static_cast<size_t>(idx)));
                        } else {
                            new_idx = n_h_e + static_cast<int>(
                                policy_v_remap(static_cast<size_t>(idx) - n_h_e));
                        }
                        fp_acc[new_idx] += static_cast<float>(v) / total_visits;
                        fm_acc[new_idx]  = 1.0f;
                    }

                    auto flipped_state = AlphaZeroBitNetImpl::preprocess(
                        flipped_snap, cfg_.rows, cfg_.cols);
                    return {std::move(flipped_state), std::move(flipped_policy),
                            std::move(flipped_mask), z};
                };

                if (total_visits > 0) {
                    // Horizontal flip (left↔right)
                    samples.push_back(make_augmented(
                        [&](size_t i){ return flip_h_idx(i, true); },
                        [&](size_t i){ return flip_v_hflip_idx(i); },
                        [&](size_t i){ return flip_box_hflip(i); },
                        [&](size_t i){ return flip_h_idx(i, true); },
                        [&](size_t i){ return flip_v_hflip_idx(i); }
                    ));
                    // Vertical flip (top↔bottom)
                    samples.push_back(make_augmented(
                        [&](size_t i){ return flip_h_vflip_idx(i); },
                        [&](size_t i){ return flip_v_vflip_idx(i); },
                        [&](size_t i){ return flip_box_vflip(i); },
                        [&](size_t i){ return flip_h_vflip_idx(i); },
                        [&](size_t i){ return flip_v_vflip_idx(i); }
                    ));
                }  // if total_visits > 0
            }  // for item in game_history

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
};

}  // namespace azb
