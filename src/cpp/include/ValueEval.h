#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace azb {

enum class ValueEval {
    kWinLoss,
    kScoreDiff,
    kScoreDiffSqrt,
    kScoreDiffTanh,
    kScoreDiffScaled,
};

inline float value_winloss(int current_player, int score_p1, int score_p2) {
    const int my_score = (current_player == 1) ? score_p1 : score_p2;
    const int opp_score = (current_player == 1) ? score_p2 : score_p1;
    if (my_score == opp_score) return 0.0f;
    return (my_score > opp_score) ? 1.0f : -1.0f;
}

inline float value_score_diff(int current_player,
                              int score_p1,
                              int score_p2,
                              int total_boxes) {
    const int my_score = (current_player == 1) ? score_p1 : score_p2;
    const int opp_score = (current_player == 1) ? score_p2 : score_p1;
    const float total_f = (total_boxes > 0) ? static_cast<float>(total_boxes) : 1.0f;
    return static_cast<float>(my_score - opp_score) / total_f;
}

inline float value_score_diff_sqrt(int current_player,
                                   int score_p1,
                                   int score_p2,
                                   int total_boxes) {
    const float base = value_score_diff(current_player, score_p1, score_p2, total_boxes);
    if (base == 0.0f) return 0.0f;
    const float mag = std::sqrt(std::fabs(base));
    return (base > 0.0f) ? mag : -mag;
}

inline float value_score_diff_tanh(int current_player,
                                   int score_p1,
                                   int score_p2,
                                   int total_boxes) {
    const float base = value_score_diff(current_player, score_p1, score_p2, total_boxes);
    // Slightly emphasize clear leads while keeping outputs in (-1, 1).
    return std::tanh(2.0f * base);
}

/// Board-size-independent score evaluation.
/// Uses raw score difference (NOT divided by total_boxes) with a fixed
/// scaling factor.  This avoids the signal-crushing problem of kScoreDiffTanh
/// on large boards where diff/total → 0.
///   1-box lead ≈ 0.46,  2-box ≈ 0.76,  3-box ≈ 0.91,  5+ ≈ ~1.0
inline float value_score_diff_scaled(int current_player,
                                     int score_p1,
                                     int score_p2,
                                     int /*total_boxes*/) {
    const int my_score  = (current_player == 1) ? score_p1 : score_p2;
    const int opp_score = (current_player == 1) ? score_p2 : score_p1;
    const float diff = static_cast<float>(my_score - opp_score);
    return std::tanh(0.5f * diff);
}

inline float value_from_scores(ValueEval eval,
                               int current_player,
                               int score_p1,
                               int score_p2,
                               int total_boxes) {
    switch (eval) {
        case ValueEval::kWinLoss:
            return value_winloss(current_player, score_p1, score_p2);
        case ValueEval::kScoreDiff:
            return value_score_diff(current_player, score_p1, score_p2, total_boxes);
        case ValueEval::kScoreDiffSqrt:
            return value_score_diff_sqrt(current_player, score_p1, score_p2, total_boxes);
        case ValueEval::kScoreDiffTanh:
            return value_score_diff_tanh(current_player, score_p1, score_p2, total_boxes);
        case ValueEval::kScoreDiffScaled:
            return value_score_diff_scaled(current_player, score_p1, score_p2, total_boxes);
    }
    return value_winloss(current_player, score_p1, score_p2);
}

inline const char* value_eval_name(ValueEval eval) {
    switch (eval) {
        case ValueEval::kWinLoss: return "winloss";
        case ValueEval::kScoreDiff: return "score_diff";
        case ValueEval::kScoreDiffSqrt: return "score_diff_sqrt";
        case ValueEval::kScoreDiffTanh: return "score_diff_tanh";
        case ValueEval::kScoreDiffScaled: return "score_diff_scaled";
    }
    return "winloss";
}

inline bool parse_value_eval(const std::string& raw, ValueEval& out) {
    std::string s = raw;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (s == "winloss" || s == "win_loss" || s == "win-loss" || s == "wl") {
        out = ValueEval::kWinLoss;
        return true;
    }
    if (s == "score" || s == "score_diff" || s == "score-diff" ||
        s == "box" || s == "boxscore" || s == "box_score") {
        out = ValueEval::kScoreDiff;
        return true;
    }
    if (s == "score_sqrt" || s == "score-diff-sqrt" || s == "score_diff_sqrt" ||
        s == "box_sqrt" || s == "boxscore_sqrt") {
        out = ValueEval::kScoreDiffSqrt;
        return true;
    }
    if (s == "score_tanh" || s == "score-diff-tanh" || s == "score_diff_tanh" ||
        s == "box_tanh" || s == "boxscore_tanh") {
        out = ValueEval::kScoreDiffTanh;
        return true;
    }
    if (s == "scaled" || s == "score_scaled" || s == "score_diff_scaled" ||
        s == "score-diff-scaled") {
        out = ValueEval::kScoreDiffScaled;
        return true;
    }
    return false;
}

}  // namespace azb
