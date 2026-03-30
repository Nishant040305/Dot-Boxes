/// AlphaZero Agent Server — stdin/stdout bridge for Python interop.
///
/// Protocol (JSON lines, one per line):
///
///   Python → C++:
///     {"cmd":"act","h_edges":0,"v_edges":0,"boxes_p1":0,"boxes_p2":0,
///      "current_player":1,"score_p1":0,"score_p2":0,"done":false}
///
///   C++ → Python:
///     {"edge_type":0,"r":1,"c":2}
///
///   Python → C++:
///     {"cmd":"quit"}
///
/// Launch:
///   ./alphazero_server --rows 3 --cols 3 --model path.pt --sims 400

#include <iostream>
#include <string>

#include <torch/torch.h>
#include "AlphaZeroBitAgent.h"
#include "AlphaZeroBitNet.h"
#include "BitBoardEnv.h"

namespace azb {

// ─── Minimal JSON parser for our flat protocol ───────────────────

class SimpleJSON {
public:
    explicit SimpleJSON(const std::string& s) : raw_(s) {}

    std::string get_string(const std::string& key) const {
        auto pos = raw_.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = raw_.find(':', pos);
        if (pos == std::string::npos) return "";
        pos = raw_.find('"', pos + 1);
        if (pos == std::string::npos) return "";
        auto end = raw_.find('"', pos + 1);
        return raw_.substr(pos + 1, end - pos - 1);
    }

    int64_t get_int(const std::string& key) const {
        auto pos = raw_.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = raw_.find(':', pos);
        if (pos == std::string::npos) return 0;
        pos++;
        while (pos < raw_.size() && (raw_[pos] == ' ' || raw_[pos] == '\t')) pos++;
        return std::stoll(raw_.substr(pos));
    }

    bool get_bool(const std::string& key) const {
        auto pos = raw_.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = raw_.find(':', pos);
        if (pos == std::string::npos) return false;
        auto next_delim = std::min(raw_.find(',', pos), raw_.find('}', pos));
        return raw_.find("true", pos) < next_delim;
    }

private:
    std::string raw_;
};

// ─── NN Policy for the server ────────────────────────────────────

class ServerNNPolicy : public PolicyValueFn {
public:
    ServerNNPolicy(AlphaZeroBitNet& model, torch::Device device)
        : model_(model), device_(device) {}

    PolicyValue operator()(const StateSnapshot& state) override {
        auto feat = model_->preprocess(state).unsqueeze(0).to(device_);
        torch::NoGradGuard no_grad;
        auto [logits, value] = model_->forward(feat);
        auto probs = torch::softmax(logits, 1).cpu();
        std::vector<float> policy(probs[0].data_ptr<float>(),
                                  probs[0].data_ptr<float>() + probs[0].numel());
        return {policy, value.item<float>()};
    }

private:
    AlphaZeroBitNet& model_;
    torch::Device device_;
};

}  // namespace azb


int main(int argc, char* argv[]) {
    int rows = 3, cols = -1, mcts_sims = 400, hidden = 256, blocks = 6;
    std::string model_path;
    float temperature = 0.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) break;
        std::string val = argv[++i];
        if      (arg == "--rows")    rows = std::stoi(val);
        else if (arg == "--cols")    cols = std::stoi(val);
        else if (arg == "--sims")    mcts_sims = std::stoi(val);
        else if (arg == "--model")   model_path = val;
        else if (arg == "--hidden")  hidden = std::stoi(val);
        else if (arg == "--blocks")  blocks = std::stoi(val);
        else if (arg == "--temp")    temperature = std::stof(val);
    }
    if (cols < 0) cols = rows;
    if (model_path.empty()) {
        model_path = "../models/alphazero_" +
                     std::to_string(rows) + "x" + std::to_string(cols) + ".pt";
    }

    // Initialize model
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);

    azb::AlphaZeroBitNet model(rows, cols, hidden, blocks);
    try {
        torch::load(model, model_path);
        model->to(device);
        model->eval();
        std::cerr << "[server] Loaded model: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[server] WARNING: " << e.what()
                  << " — using random weights." << std::endl;
        model->to(device);
        model->eval();
    }

    // Create agent
    azb::ServerNNPolicy nn_policy(model, device);
    azb::BitBoardEnv env(rows, cols);
    azb::AlphaZeroBitAgent agent(env, nn_policy, mcts_sims,
                                 1.5f, 0.3f, 0.25f, 0.25f, false);

    // Signal ready (stderr for diagnostics, stdout for protocol)
    std::cout << "{\"status\":\"ready\",\"rows\":" << rows
              << ",\"cols\":" << cols
              << ",\"action_size\":" << env.action_size()
              << "}" << std::endl;
    std::cerr << "[server] Ready (rows=" << rows << ", cols=" << cols
              << ", sims=" << mcts_sims << ")" << std::endl;

    // Main loop
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        azb::SimpleJSON json(line);
        std::string cmd = json.get_string("cmd");

        if (cmd == "quit") break;

        if (cmd == "reset") {
            env.reset();
            std::cout << "{\"status\":\"ok\"}" << std::endl;
            continue;
        }

        if (cmd == "act") {
            // Restore state from Python's authoritative representation
            uint64_t h_edges = static_cast<uint64_t>(json.get_int("h_edges"));
            uint64_t v_edges = static_cast<uint64_t>(json.get_int("v_edges"));
            uint64_t boxes_p1 = static_cast<uint64_t>(json.get_int("boxes_p1"));
            uint64_t boxes_p2 = static_cast<uint64_t>(json.get_int("boxes_p2"));
            int current_player = static_cast<int>(json.get_int("current_player"));
            int score_p1 = static_cast<int>(json.get_int("score_p1"));
            int score_p2 = static_cast<int>(json.get_int("score_p2"));
            bool done = json.get_bool("done");

            env.set_state(h_edges, v_edges, boxes_p1, boxes_p2,
                          current_player, done, score_p1, score_p2);

            if (env.done()) {
                std::cout << "{\"edge_type\":-1,\"r\":-1,\"c\":-1}" << std::endl;
                continue;
            }

            azb::Action action = agent.act(env, false, temperature);
            std::cout << "{\"edge_type\":" << action.edge_type
                      << ",\"r\":" << action.r
                      << ",\"c\":" << action.c << "}" << std::endl;
        }
    }

    std::cerr << "[server] Shutdown." << std::endl;
    return 0;
}
