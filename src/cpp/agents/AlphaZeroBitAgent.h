#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BitBoardEnv.h"

namespace azb {

struct PolicyValue {
    std::vector<float> policy;  // size = action_size
    float value = 0.0f;         // in [-1, 1]
};

class PolicyValueFn {
public:
    virtual ~PolicyValueFn() = default;
    virtual PolicyValue operator()(const StateSnapshot& state) = 0;
};

class AlphaZeroBitAgent {
public:
    AlphaZeroBitAgent(const BitBoardEnv& env,
                      PolicyValueFn& model,
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
        uint64_t h_edges = 0;
        uint64_t v_edges = 0;
        uint64_t boxes_p1 = 0;
        uint64_t boxes_p2 = 0;
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
    };

    float node_value(const Node& node) const {
        return node.visits == 0 ? 0.0f : node.value_sum / static_cast<float>(node.visits);
    }

    void precompute_edge_bits();
    NodeState apply_action(const NodeState& state, const Action& action) const;
    std::vector<Action> legal_actions(const NodeState& state) const;
    std::vector<float> build_features(const NodeState& state) const;
    float evaluate_and_expand(Node& node, bool is_root);
    std::pair<Action, Node*> select_child(Node& node);
    void backpropagate(Node* node, float value);
    Action best_action(const Node& root, float temperature);
    uint32_t action_to_index(const Action& action) const;

    int n_;
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

    PolicyValueFn& model_;
    std::mt19937 rng_;
    std::unordered_map<uint32_t, int> last_visit_counts_;
};

}  // namespace azb
