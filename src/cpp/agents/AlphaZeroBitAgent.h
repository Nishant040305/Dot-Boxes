#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BitBoardEnv.h"

namespace azb {

/// Result from a policy-value network evaluation.
struct PolicyValue {
    std::vector<float> policy;  // size = action_size
    float value = 0.0f;         // in [-1, 1]
};

/// Abstract interface for async policy-value function.
class AsyncPolicyValueFn {
public:
    virtual ~AsyncPolicyValueFn() = default;
    virtual uint64_t submit(const StateSnapshot& state) = 0;
    virtual bool try_get(uint64_t request_id, PolicyValue& out) = 0;
};

// ── 128-bit cache key ──────────────────────────────────────────────────
// Using a pair<uint64_t, uint64_t> as map key.
// Practically zero collision probability for boards up to 20x20.

using StateKey128 = std::pair<uint64_t, uint64_t>;

struct StateKey128Hasher {
    size_t operator()(const StateKey128& k) const noexcept {
        // XOR-fold both halves with golden-ratio mixing
        uint64_t h = k.first ^ (k.second * 0x9e3779b97f4a7c15ULL);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        return static_cast<size_t>(h);
    }
};

using InferenceCache = std::unordered_map<StateKey128, PolicyValue, StateKey128Hasher>;

// ── AlphaZero Bit Agent ───────────────────────────────────────────────

/// MCTS-based AlphaZero agent for Dots-and-Boxes.
/// Supports rectangular boards (rows x cols).
class AlphaZeroBitAgent {
public:
    AlphaZeroBitAgent(const BitBoardEnv& env,
                      AsyncPolicyValueFn& model,
                      int n_simulations = 400,
                      float c_puct = 1.5f,
                      float dirichlet_alpha = 0.3f,
                      float dirichlet_epsilon = 0.25f,
                      float fpu_reduction = 0.25f,
                      bool add_noise = true);

    Action act(const BitBoardEnv& env, bool return_probs = false, float temperature = 1e-3f);
    const std::unordered_map<uint32_t, int>& last_visit_counts() const {
        return last_visit_counts_;
    }

private:
    struct NodeState {
        azb::FastBitset h_edges;
        azb::FastBitset v_edges;
        azb::FastBitset boxes_p1;
        azb::FastBitset boxes_p2;
        int current_player = 1;
        bool done = false;
        int score_p1 = 0;
        int score_p2 = 0;
    };

    struct Node {
        NodeState state;
        Node* parent = nullptr;
        std::unordered_map<uint32_t, Node*> children;
        int visits = 0;
        float value_sum = 0.0f;
        float prior = 0.0f;
        enum class Status { kUnexpanded, kPending, kExpanded } status = Status::kUnexpanded;
        uint64_t pending_id = 0;
    };

    float node_value(const Node& node) const {
        return node.visits == 0 ? 0.0f : node.value_sum / static_cast<float>(node.visits);
    }

    /// Compute a 128-bit cache key from the full board state.
    /// Combines hash128() of all four FastBitsets + current_player.
    StateKey128 state_key(const NodeState& s) const {
        auto [h0a, h1a] = s.h_edges.hash128();
        auto [h0b, h1b] = s.v_edges.hash128();
        auto [h0c, h1c] = s.boxes_p1.hash128();
        auto [h0d, h1d] = s.boxes_p2.hash128();
        // Mix all four hash pairs + player into two 64-bit values
        uint64_t lo = h0a ^ (h0b * 0x517cc1b727220a95ULL)
                          ^ (h0c * 0x6c62272e07bb0142ULL)
                          ^ (h0d * 0x94d049bb133111ebULL)
                          ^ static_cast<uint64_t>(s.current_player);
        uint64_t hi = h1a ^ (h1b * 0x94d049bb133111ebULL)
                          ^ (h1c * 0x517cc1b727220a95ULL)
                          ^ (h1d * 0x6c62272e07bb0142ULL);
        return {lo, hi};
    }

    void precompute_edge_bits();
    NodeState apply_action(const NodeState& state, const Action& action) const;
    std::vector<Action> legal_actions(const NodeState& state) const;
    std::vector<float> build_features(const NodeState& state) const;
    bool try_expand(Node& node, bool is_root, float& out_value);
    std::pair<Action, Node*> select_child(Node& node);
    void backpropagate(Node* node, float value);
    Action best_action(const Node& root, float temperature);
    uint32_t action_to_index(const Action& action) const;

    int rows_;
    int cols_;
    int n_h_edges_;
    int n_v_edges_;
    int action_size_;
    int n_simulations_;
    float c_puct_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    float fpu_reduction_;
    bool add_noise_;

    std::vector<std::pair<int, int>> box_h_bits_;
    std::vector<std::pair<int, int>> box_v_bits_;

    AsyncPolicyValueFn& model_;
    std::mt19937 rng_;
    std::unordered_map<uint32_t, int> last_visit_counts_;

    // Inference cache: board state → (policy, value)
    // Cleared at the start of each act() call.
    InferenceCache inference_cache_;
    std::unordered_map<StateKey128, uint64_t, StateKey128Hasher> pending_cache_;
};

}  // namespace azb
