#pragma once
/// Self-play worker — plays complete games using AlphaZeroBitAgent
/// and submits training data to the replay buffer.
/// Uses ModelHandle for model-agnostic preprocessing.

#include <atomic>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <torch/torch.h>
#include "AlphaZeroBitAgent.h"
#include "BitBoardEnv.h"
#include "AlphaZeroCNNNet.h"
#include "InferenceServer.h"
#include "ModelHandle.h"
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
        if (!server_.try_get(worker_id_, request_id, resp)) return false;
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
                                cfg_.value_eval, phase_.capture_boost);

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
                // Feature tensor — use the right preprocessing based on model type.
                // For standard model: AlphaZeroBitNetImpl::preprocess
                // For CNN model: AlphaZeroCNNNetImpl::preprocess (spatial 2D)
                // For PatchNet: PatchNetImpl::preprocess (includes patch features)
                torch::Tensor state_tensor;
                if (cfg_.use_patch_net) {
                    state_tensor = PatchNetImpl::preprocess(
                        item.snapshot, cfg_.rows, cfg_.cols,
                        cfg_.patch_rows, cfg_.patch_cols,
                        precomputed_patches());
                } else if (cfg_.use_cnn_net) {
                    state_tensor = AlphaZeroCNNNetImpl::preprocess(
                        item.snapshot, cfg_.rows, cfg_.cols);
                } else {
                    state_tensor = AlphaZeroBitNetImpl::preprocess(
                        item.snapshot, cfg_.rows, cfg_.cols);
                }

                // Reconstruct legality from the stored snapshot so we always have a valid legal mask,
                // even if MCTS visit counts are missing/empty (e.g., due to inference timeouts).
                BitBoardEnv env_tmp(cfg_.rows, cfg_.cols);
                env_tmp.set_state(item.snapshot.h_edges, item.snapshot.v_edges,
                                  item.snapshot.boxes_p1, item.snapshot.boxes_p2,
                                  item.snapshot.current_player, item.snapshot.done,
                                  item.snapshot.score_p1, item.snapshot.score_p2);
                const azb::FastBitset legal_bits = env_tmp.get_legal_actions_mask();

                // Build policy tensor (MCTS visit counts, normalised)
                auto policy_tensor = torch::zeros({action_size}, torch::kFloat32);
                auto legal_mask = torch::zeros({action_size}, torch::kFloat32);
                auto pacc = policy_tensor.accessor<float, 1>();
                auto macc = legal_mask.accessor<float, 1>();

                int n_legal = 0;
                legal_bits.for_each_set_bit([&](size_t idx) {
                    if (idx < static_cast<size_t>(action_size)) {
                        macc[static_cast<int64_t>(idx)] = 1.0f;
                        n_legal++;
                    }
                });

                int total_visits = 0;
                for (const auto& [idx, v] : item.visits) total_visits += v;
                if (total_visits > 0) {
                    const float inv_temp = 1.0f / std::max(cfg_.policy_target_temp, 0.01f);
                    if (std::abs(inv_temp - 1.0f) < 1e-6f) {
                        // No sharpening — fast path
                        for (const auto& [idx, v] : item.visits) {
                            pacc[idx] = static_cast<float>(v) / total_visits;
                        }
                    } else {
                        // Sharpen: raise counts to power 1/τ, then re-normalize.
                        // Use log-space arithmetic to avoid float overflow when
                        // inv_temp is large (e.g. τ=0.05 → inv_temp=20):
                        //   sharp(v) = v^inv_temp = exp(inv_temp * log(v))
                        // Subtract max before exp for numerical stability (log-sum-exp trick).
                        std::vector<std::pair<int, float>> log_vals;
                        log_vals.reserve(item.visits.size());
                        float max_log = -std::numeric_limits<float>::infinity();
                        for (const auto& [idx, v] : item.visits) {
                            if (v > 0) {
                                float lv = inv_temp * std::log(static_cast<float>(v));
                                log_vals.push_back({idx, lv});
                                if (lv > max_log) max_log = lv;
                            }
                        }
                        float sharp_sum = 0.0f;
                        for (const auto& [idx, lv] : log_vals) {
                            float s = std::exp(lv - max_log);
                            pacc[idx] = s;
                            sharp_sum += s;
                        }
                        if (sharp_sum > 0.0f) {
                            for (const auto& [idx, lv] : log_vals) {
                                pacc[idx] /= sharp_sum;
                            }
                        }
                    }
                } else if (n_legal > 0) {
                    const float uniform = 1.0f / static_cast<float>(n_legal);
                    legal_bits.for_each_set_bit([&](size_t idx) {
                        if (idx < static_cast<size_t>(action_size)) {
                            pacc[static_cast<int64_t>(idx)] = uniform;
                        }
                    });
                }

                float z = value_from_scores(cfg_.value_eval, item.player,
                                            env.score_p1(), env.score_p2(),
                                            total_boxes);

                samples.push_back({std::move(state_tensor),
                                   std::move(policy_tensor),
                                   std::move(legal_mask), z});

                // ── Symmetry Augmentation ─────────────────────────────────
                // NOTE: For PatchNet, augmentation operates on the full
                // combined feature vector. The symmetry class handles
                // the mapping correctly since edges/boxes have the same
                // ordering in the global portion of the feature vector.
                // We disable augmentation for PatchNet for now since
                // patch features would need separate permutation tables.
                if (cfg_.use_augmentation && !cfg_.use_patch_net && !cfg_.use_cnn_net) {
                    augment_position(sym_transforms_, geo_, item.snapshot,
                                     item.visits, total_visits, z, samples,
                                     cfg_.policy_target_temp);
                }
            }

            // ── NaN validation: drop any contaminated samples ─────
            {
                size_t before = samples.size();
                samples.erase(
                    std::remove_if(samples.begin(), samples.end(),
                        [](const TrainSample& s) {
                            return s.policy.isnan().any().item<bool>() ||
                                   s.state.isnan().any().item<bool>() ||
                                   std::isnan(s.value) || std::isinf(s.value);
                        }),
                    samples.end());
                if (samples.size() < before) {
                    std::cerr << "[Worker " << worker_id_
                              << "] WARNING: dropped " << (before - samples.size())
                              << "/" << before << " NaN-contaminated samples"
                              << std::endl;
                }
            }

            buffer_.push_batch(samples);
            games_count_.fetch_add(1);
        }  // while !stop
    }  // run()

private:
    /// Lazy-init patch descriptors (only needed if use_patch_net is true).
    const std::vector<PatchDesc>& precomputed_patches() {
        if (patches_.empty() && cfg_.use_patch_net) {
            // Recreate the PatchNet temporarily just to get the patch descriptors.
            // This is called once per worker, so the overhead is negligible.
            PatchNet tmp(cfg_.rows, cfg_.cols,
                          cfg_.patch_rows, cfg_.patch_cols,
                          cfg_.local_hidden_size, cfg_.local_num_res_blocks,
                          cfg_.global_hidden_size, cfg_.global_num_res_blocks,
                          cfg_.dropout);
            patches_ = tmp->patches;
        }
        return patches_;
    }

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

    // Cached patch descriptors for PatchNet preprocessing
    std::vector<PatchDesc> patches_;
};

}  // namespace azb
