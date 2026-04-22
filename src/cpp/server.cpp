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
#include <mutex>
#include <string>
#include <unordered_map>

#include <torch/torch.h>
#include "AlphaZeroBitAgent.h"
#include "AlphaZeroBitNet.h"
#include "BitBoardEnv.h"
#include "PatchNet.h"

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

template <typename ModelHolder>
class ServerNNPolicyT : public AsyncPolicyValueFn {
public:
    ServerNNPolicyT(ModelHolder& model, torch::Device device)
        : model_(model), device_(device) {}

    uint64_t submit(const StateSnapshot& state) override {
        auto feat = model_->preprocess(state).unsqueeze(0).to(device_);
        torch::NoGradGuard no_grad;
        auto [logits, value] = model_->forward(feat);
        auto probs = torch::softmax(logits, 1).cpu();
        std::vector<float> policy(probs[0].template data_ptr<float>(),
                                  probs[0].template data_ptr<float>() + probs[0].numel());
        PolicyValue pv{std::move(policy), value.template item<float>()};
        std::lock_guard<std::mutex> lock(mu_);
        const uint64_t req_id = next_id_++;
        ready_[req_id] = std::move(pv);
        return req_id;
    }

    bool try_get(uint64_t request_id, PolicyValue& out) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = ready_.find(request_id);
        if (it == ready_.end()) return false;
        out = std::move(it->second);
        ready_.erase(it);
        return true;
    }

private:
    ModelHolder& model_;
    torch::Device device_;
    std::unordered_map<uint64_t, PolicyValue> ready_;
    uint64_t next_id_ = 1;
    std::mutex mu_;
};

}  // namespace azb


int main(int argc, char* argv[]) {
    // LibTorch internal threads for matrix ops in inference.
    // Tuned for i7-13700H: use 4 threads for DNNL/OpenMP matrix ops.
    torch::set_num_threads(4);
    torch::set_num_interop_threads(1);

    int rows = 3, cols = -1, mcts_sims = 400, hidden = 256, blocks = 6;
    std::string model_path;
    float temperature = 0.0f;
    bool use_dag = true;
    float c_puct = 1.6f;
    azb::ValueEval value_eval = azb::ValueEval::kScoreDiffScaled;
    bool use_patch_net = false;
    int patch_rows = 3, patch_cols = 3;
    int local_hidden = 128, local_blocks = 6;
    int global_hidden = 192, global_blocks = 4;
    std::string local_model_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dag") {
            use_dag = true;
            continue;
        }
        if (arg == "--no-dag") {
            use_dag = false;
            continue;
        }
        if (arg == "--patch") {
            use_patch_net = true;
            continue;
        }
        if (i + 1 >= argc) break;
        std::string val = argv[++i];
        if      (arg == "--rows")    rows = std::stoi(val);
        else if (arg == "--cols")    cols = std::stoi(val);
        else if (arg == "--sims")    mcts_sims = std::stoi(val);
        else if (arg == "--model")   model_path = val;
        else if (arg == "--hidden")  hidden = std::stoi(val);
        else if (arg == "--blocks")  blocks = std::stoi(val);
        else if (arg == "--temp")    temperature = std::stof(val);
        else if (arg == "--c-puct")  c_puct = std::stof(val);
        else if (arg == "--patch-rows")    patch_rows = std::stoi(val);
        else if (arg == "--patch-cols")    patch_cols = std::stoi(val);
        else if (arg == "--local-hidden")  local_hidden = std::stoi(val);
        else if (arg == "--local-blocks")  local_blocks = std::stoi(val);
        else if (arg == "--global-hidden") global_hidden = std::stoi(val);
        else if (arg == "--global-blocks") global_blocks = std::stoi(val);
        else if (arg == "--local-model")   local_model_path = val;
        else if (arg == "--value-eval") {
            if (!azb::parse_value_eval(val, value_eval)) {
                std::cerr << "[server] Unknown value eval: " << val << std::endl;
                return 1;
            }
        }
    }
    if (cols < 0) cols = rows;
    if (model_path.empty()) {
        model_path = "../models/alphazero_" +
                     std::to_string(rows) + "x" + std::to_string(cols) + ".pt";
    }

    // Initialize model
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);

    azb::BitBoardEnv env(rows, cols);
    auto run_loop = [&](azb::AlphaZeroBitAgent& agent) {
        std::cerr << "[server] Value eval: " << azb::value_eval_name(value_eval) << std::endl;

        // Signal ready (stderr for diagnostics, stdout for protocol)
        std::cout << "{\"status\":\"ready\",\"rows\":" << rows
                  << ",\"cols\":" << cols
                  << ",\"action_size\":" << env.action_size()
                  << "}" << std::endl;
        std::cerr << "[server] Ready (rows=" << rows << ", cols=" << cols
                  << ", sims=" << mcts_sims
                  << ", dag=" << (use_dag ? "on" : "off") << ")" << std::endl;

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
                azb::FastBitset h_edges((rows + 1) * cols);
                azb::FastBitset v_edges(rows * (cols + 1));
                azb::FastBitset boxes_p1(rows * cols);
                azb::FastBitset boxes_p2(rows * cols);

                if (h_edges.num_words() > 0) h_edges.raw()[0] = static_cast<uint64_t>(json.get_int("h_edges"));
                if (v_edges.num_words() > 0) v_edges.raw()[0] = static_cast<uint64_t>(json.get_int("v_edges"));
                if (boxes_p1.num_words() > 0) boxes_p1.raw()[0] = static_cast<uint64_t>(json.get_int("boxes_p1"));
                if (boxes_p2.num_words() > 0) boxes_p2.raw()[0] = static_cast<uint64_t>(json.get_int("boxes_p2"));

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
    };

    if (use_patch_net) {
        azb::PatchNet model(rows, cols, patch_rows, patch_cols,
                            local_hidden, local_blocks,
                            global_hidden, global_blocks);
        if (!local_model_path.empty()) {
            try {
                model->load_and_freeze_local(local_model_path);
            } catch (const std::exception& e) {
                std::cerr << "[server] WARNING: Failed to load local model: "
                          << local_model_path << " (" << e.what() << ")"
                          << std::endl;
            }
        }
        try {
            torch::load(model, model_path);
            model->freeze_local();
            model->to(device);
            model->eval();
            std::cerr << "[server] Loaded PatchNet model: " << model_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[server] WARNING: " << e.what()
                      << " — using random weights." << std::endl;
            model->freeze_local();
            model->to(device);
            model->eval();
        }

        azb::ServerNNPolicyT<azb::PatchNet> nn_policy(model, device);
        azb::AlphaZeroBitAgent agent(env, nn_policy, mcts_sims,
                                     c_puct, 0.3f, 0.25f, 0.20f,
                                     false, use_dag, value_eval);
        run_loop(agent);
    } else {
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

        azb::ServerNNPolicyT<azb::AlphaZeroBitNet> nn_policy(model, device);
        azb::AlphaZeroBitAgent agent(env, nn_policy, mcts_sims,
                                     c_puct, 0.3f, 0.25f, 0.20f,
                                     false, use_dag, value_eval);
        run_loop(agent);
    }

    std::cerr << "[server] Shutdown." << std::endl;
    return 0;
}
