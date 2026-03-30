#include "AlphaZeroBitAgent.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace azb {

AlphaZeroBitAgent::AlphaZeroBitAgent(const BitBoardEnv& env,
                                     PolicyValueFn& model,
                                     int n_simulations,
                                     float c_puct,
                                     float dirichlet_alpha,
                                     float dirichlet_epsilon,
                                     float fpu_reduction,
                                     bool add_noise)
    : rows_(env.rows()),
      cols_(env.cols()),
      n_h_edges_((rows_ + 1) * cols_),
      n_v_edges_(rows_ * (cols_ + 1)),
      action_size_(n_h_edges_ + n_v_edges_),
      n_simulations_(n_simulations),
      c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_epsilon_(dirichlet_epsilon),
      fpu_reduction_(fpu_reduction),
      add_noise_(add_noise),
      model_(model),
      rng_(std::random_device{}()) {
    precompute_edge_bits();
}

void AlphaZeroBitAgent::precompute_edge_bits() {
    box_h_bits_.resize(rows_ * cols_);
    box_v_bits_.resize(rows_ * cols_);
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            const int h_top = r * cols_ + c;
            const int h_bot = (r + 1) * cols_ + c;
            const int v_left = r * (cols_ + 1) + c;
            const int v_right = r * (cols_ + 1) + (c + 1);
            box_h_bits_[r * cols_ + c] = {h_top, h_bot};
            box_v_bits_[r * cols_ + c] = {v_left, v_right};
        }
    }
}

uint32_t AlphaZeroBitAgent::action_to_index(const Action& action) const {
    if (action.edge_type == 0) {
        return static_cast<uint32_t>(action.r * cols_ + action.c);
    }
    return static_cast<uint32_t>(n_h_edges_ + action.r * (cols_ + 1) + action.c);
}

std::vector<float> AlphaZeroBitAgent::build_features(const NodeState& state) const {
    const int input_size = n_h_edges_ + n_v_edges_ + rows_ * cols_ + 1;
    std::vector<float> features(static_cast<size_t>(input_size), 0.0f);

    for (int i = 0; i < n_h_edges_; i++) {
        if (state.h_edges & (1ULL << i)) features[static_cast<size_t>(i)] = 1.0f;
    }
    for (int i = 0; i < n_v_edges_; i++) {
        const int idx = n_h_edges_ + i;
        if (state.v_edges & (1ULL << i)) features[static_cast<size_t>(idx)] = 1.0f;
    }
    for (int i = 0; i < rows_ * cols_; i++) {
        const uint64_t bit = 1ULL << i;
        if (state.boxes_p1 & bit) {
            features[static_cast<size_t>(n_h_edges_ + n_v_edges_ + i)] =
                (state.current_player == 1) ? 1.0f : -1.0f;
        } else if (state.boxes_p2 & bit) {
            features[static_cast<size_t>(n_h_edges_ + n_v_edges_ + i)] =
                (state.current_player == 2) ? 1.0f : -1.0f;
        }
    }
    features.back() = (state.current_player == 1) ? 1.0f : -1.0f;
    return features;
}

std::vector<Action> AlphaZeroBitAgent::legal_actions(const NodeState& state) const {
    std::vector<Action> actions;
    const uint64_t all_h = (n_h_edges_ == 0) ? 0ULL : ((1ULL << n_h_edges_) - 1ULL);
    const uint64_t all_v = (n_v_edges_ == 0) ? 0ULL : ((1ULL << n_v_edges_) - 1ULL);

    uint64_t free_h = all_h & ~state.h_edges;
    uint64_t free_v = all_v & ~state.v_edges;

    while (free_h) {
        const int idx = __builtin_ctzll(free_h);
        actions.push_back({0, idx / cols_, idx % cols_});
        free_h &= (free_h - 1);
    }
    while (free_v) {
        const int idx = __builtin_ctzll(free_v);
        actions.push_back({1, idx / (cols_ + 1), idx % (cols_ + 1)});
        free_v &= (free_v - 1);
    }
    return actions;
}

AlphaZeroBitAgent::NodeState AlphaZeroBitAgent::apply_action(const NodeState& state,
                                                             const Action& action) const {
    NodeState next = state;
    const int edge_type = action.edge_type;
    const int r = action.r;
    const int c = action.c;

    std::vector<std::pair<int, int>> adj_boxes;
    if (edge_type == 0) {
        next.h_edges |= (1ULL << (r * cols_ + c));
        if (r > 0) adj_boxes.push_back({r - 1, c});
        if (r < rows_) adj_boxes.push_back({r, c});
    } else {
        next.v_edges |= (1ULL << (r * (cols_ + 1) + c));
        if (c > 0) adj_boxes.push_back({r, c - 1});
        if (c < cols_) adj_boxes.push_back({r, c});
    }

    bool box_made = false;
    for (const auto& box : adj_boxes) {
        const int box_r = box.first;
        const int box_c = box.second;
        const uint64_t h_mask =
            (1ULL << box_h_bits_[box_r * cols_ + box_c].first) |
            (1ULL << box_h_bits_[box_r * cols_ + box_c].second);
        const uint64_t v_mask =
            (1ULL << box_v_bits_[box_r * cols_ + box_c].first) |
            (1ULL << box_v_bits_[box_r * cols_ + box_c].second);
        const uint64_t box_bit = 1ULL << (box_r * cols_ + box_c);
        if ((next.h_edges & h_mask) == h_mask && (next.v_edges & v_mask) == v_mask) {
            if ((next.boxes_p1 & box_bit) == 0 && (next.boxes_p2 & box_bit) == 0) {
                if (next.current_player == 1) {
                    next.boxes_p1 |= box_bit;
                    next.score_p1 += 1;
                } else {
                    next.boxes_p2 |= box_bit;
                    next.score_p2 += 1;
                }
                box_made = true;
            }
        }
    }

    if (!box_made) {
        next.current_player = 3 - next.current_player;
    }
    next.done = (next.score_p1 + next.score_p2) == (rows_ * cols_);
    return next;
}

float AlphaZeroBitAgent::evaluate_and_expand(Node& node, bool is_root) {
    if (node.state.done) {
        if (node.state.score_p1 == node.state.score_p2) return 0.0f;
        const int my_score = (node.state.current_player == 1) ? node.state.score_p1
                                                              : node.state.score_p2;
        const int opp_score = (node.state.current_player == 1) ? node.state.score_p2
                                                               : node.state.score_p1;
        return (my_score > opp_score) ? 1.0f : -1.0f;
    }

    PolicyValue pv = model_(StateSnapshot{
        node.state.h_edges,
        node.state.v_edges,
        node.state.boxes_p1,
        node.state.boxes_p2,
        node.state.current_player,
        node.state.done,
        node.state.score_p1,
        node.state.score_p2,
        Action{-1, -1, -1},
    });

    if (static_cast<int>(pv.policy.size()) != action_size_) {
        throw std::runtime_error("Policy size mismatch");
    }

    const std::vector<Action> actions = legal_actions(node.state);
    if (actions.empty()) return pv.value;

    std::vector<float> priors;
    priors.reserve(actions.size());
    float total = 0.0f;
    for (const auto& action : actions) {
        const uint32_t idx = action_to_index(action);
        const float p = pv.policy[idx];
        priors.push_back(p);
        total += p;
    }

    if (total <= 0.0f) {
        const float uniform = 1.0f / static_cast<float>(actions.size());
        std::fill(priors.begin(), priors.end(), uniform);
    } else {
        for (float& p : priors) p /= total;
    }

    if (is_root && add_noise_) {
        std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
        std::vector<float> noise(actions.size(), 0.0f);
        float sum = 0.0f;
        for (float& v : noise) {
            v = gamma(rng_);
            sum += v;
        }
        for (size_t i = 0; i < actions.size(); i++) {
            const float n = (sum > 0.0f) ? noise[i] / sum : 1.0f / noise.size();
            priors[i] = (1.0f - dirichlet_epsilon_) * priors[i] + dirichlet_epsilon_ * n;
        }
    }

    for (size_t i = 0; i < actions.size(); i++) {
        Node* child = new Node();
        child->state = apply_action(node.state, actions[i]);
        child->parent = &node;
        child->prior = priors[i];
        node.children[action_to_index(actions[i])] = child;
    }

    return pv.value;
}

std::pair<Action, AlphaZeroBitAgent::Node*> AlphaZeroBitAgent::select_child(Node& node) {
    float best_score = -1e30f;
    uint32_t best_idx = 0;
    Node* best_child = nullptr;

    int sum_visits = 0;
    for (const auto& kv : node.children) sum_visits += kv.second->visits;
    const float sqrt_sum = std::sqrt(static_cast<float>(sum_visits + 1));
    const float parent_q = (node.visits > 0) ? node_value(node) : 0.0f;
    const float fpu_value = parent_q - fpu_reduction_;

    for (const auto& kv : node.children) {
        const uint32_t idx = kv.first;
        Node* child = kv.second;
        const float q_value = (child->visits == 0) ? fpu_value : node_value(*child);
        const float action_value =
            (child->state.current_player == node.state.current_player) ? q_value : -q_value;
        const float u_score = c_puct_ * child->prior * sqrt_sum / (1.0f + child->visits);
        const float score = action_value + u_score;
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
            best_child = child;
        }
    }

    Action action;
    if (best_idx < static_cast<uint32_t>(n_h_edges_)) {
        action.edge_type = 0;
        action.r = static_cast<int>(best_idx / cols_);
        action.c = static_cast<int>(best_idx % cols_);
    } else {
        const uint32_t rel = best_idx - static_cast<uint32_t>(n_h_edges_);
        action.edge_type = 1;
        action.r = static_cast<int>(rel / (cols_ + 1));
        action.c = static_cast<int>(rel % (cols_ + 1));
    }

    return {action, best_child};
}

void AlphaZeroBitAgent::backpropagate(Node* node, float value) {
    while (node) {
        node->visits += 1;
        node->value_sum += value;
        if (node->parent && node->parent->state.current_player != node->state.current_player) {
            value = -value;
        }
        node = node->parent;
    }
}

Action AlphaZeroBitAgent::best_action(const Node& root, float temperature) {
    std::vector<uint32_t> actions;
    std::vector<float> weights;
    actions.reserve(root.children.size());
    weights.reserve(root.children.size());

    for (const auto& kv : root.children) {
        actions.push_back(kv.first);
        weights.push_back(static_cast<float>(kv.second->visits));
    }

    if (actions.empty()) return Action{-1, -1, -1};

    auto idx_to_action = [&](uint32_t idx) -> Action {
        if (idx < static_cast<uint32_t>(n_h_edges_)) {
            return {0, static_cast<int>(idx / cols_), static_cast<int>(idx % cols_)};
        }
        const uint32_t rel = idx - static_cast<uint32_t>(n_h_edges_);
        return {1, static_cast<int>(rel / (cols_ + 1)), static_cast<int>(rel % (cols_ + 1))};
    };

    if (temperature <= 1e-3f) {
        size_t best = 0;
        for (size_t i = 1; i < weights.size(); i++) {
            if (weights[i] > weights[best]) best = i;
        }
        return idx_to_action(actions[best]);
    }

    std::vector<float> logits(weights.size(), 0.0f);
    float max_log = -1e30f;
    for (size_t i = 0; i < weights.size(); i++) {
        const float v = std::log(weights[i] + 1e-10f) / temperature;
        logits[i] = v;
        if (v > max_log) max_log = v;
    }
    float sum = 0.0f;
    for (float& v : logits) {
        v = std::exp(v - max_log);
        sum += v;
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float pick = dist(rng_) * sum;
    float acc = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        acc += logits[i];
        if (acc >= pick) {
            return idx_to_action(actions[i]);
        }
    }
    return idx_to_action(actions.back());
}

Action AlphaZeroBitAgent::act(const BitBoardEnv& env, bool return_probs, float temperature) {
    last_visit_counts_.clear();
    Node root;
    root.state = NodeState{
        env.h_edges(), env.v_edges(), env.boxes_p1(), env.boxes_p2(),
        env.current_player(), env.done(), env.score_p1(), env.score_p2()
    };

    evaluate_and_expand(root, true);
    for (int i = 0; i < n_simulations_; i++) {
        Node* node = &root;
        while (!node->children.empty()) {
            auto sel = select_child(*node);
            node = sel.second;
        }
        const float value = evaluate_and_expand(*node, false);
        backpropagate(node, value);
    }

    for (const auto& kv : root.children) {
        last_visit_counts_[kv.first] = kv.second->visits;
    }

    Action chosen = best_action(root, temperature);

    // Cleanup tree
    std::vector<Node*> stack;
    stack.push_back(&root);
    while (!stack.empty()) {
        Node* cur = stack.back();
        stack.pop_back();
        for (auto& kv : cur->children) stack.push_back(kv.second);
        if (cur != &root) delete cur;
    }
    return chosen;
}

}  // namespace azb
