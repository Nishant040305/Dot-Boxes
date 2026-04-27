#pragma once
/// ModelHandle — Type-erased model wrapper for the training pipeline.
///
/// Allows AlphaZeroBitNet and PatchNet (and future models) to be used
/// interchangeably by the Trainer, InferenceServer, and SelfPlayWorker
/// without templates or inheritance changes.

#include <functional>
#include <string>
#include <torch/torch.h>
#include "AlphaZeroBitNet.h"
#include "AlphaZeroCNNNet.h"
#include "PatchNet.h"
#include "TrainConfig.h"

namespace azb {

struct ModelHandle {
    // ── Core inference ──
    std::function<torch::Tensor(const StateSnapshot&)> preprocess;
    std::function<std::pair<torch::Tensor, torch::Tensor>(torch::Tensor)> forward;

    // ── Lifecycle ──
    std::shared_ptr<torch::nn::Module> module;

    void to(torch::Device d)        { module->to(d); }
    void train_mode()               { module->train(); }
    void eval_mode()                { module->eval(); }
    auto parameters()               { return module->parameters(); }
    auto named_parameters()         { return module->named_parameters(); }
    auto named_buffers()            { return module->named_buffers(); }

    // ── Serialisation ──
    std::function<void(const std::string&)> save;
    std::function<void(const std::string&)> load;

    // ── Clone: create a separate inference copy ──
    std::function<ModelHandle(const TrainConfig&)> clone_fn;
    ModelHandle clone(const TrainConfig& cfg) const { return clone_fn(cfg); }
};

// ─── Factory: standard AlphaZeroBitNet ───────────────────────────────

inline ModelHandle make_standard_model(int rows, int cols,
                                        int hidden, int blocks, double dropout) {
    AlphaZeroBitNet net(rows, cols, hidden, blocks, dropout);
    ModelHandle h;
    h.module     = net.ptr();
    h.preprocess = [net](const StateSnapshot& s) mutable { return net->preprocess(s); };
    h.forward    = [net](torch::Tensor x) mutable { return net->forward(x); };
    h.save       = [net](const std::string& p) { torch::save(net, p); };
    h.load       = [net](const std::string& p) mutable {
        torch::load(net, p);
    };
    h.clone_fn   = [net](const TrainConfig& cfg) -> ModelHandle {
        // Create an independent copy for inference
        AlphaZeroBitNet copy(cfg.rows, cfg.cols, cfg.hidden_size,
                              cfg.num_res_blocks, cfg.dropout);
        {
            torch::NoGradGuard no_grad;
            auto src_params = net->named_parameters();
            for (auto& p : copy->named_parameters()) {
                auto* found = src_params.find(p.key());
                if (found) p.value().copy_(*found);
            }
            auto src_bufs = net->named_buffers();
            for (auto& b : copy->named_buffers()) {
                auto* found = src_bufs.find(b.key());
                if (found) b.value().copy_(*found);
            }
        }
        ModelHandle ch;
        ch.module     = copy.ptr();
        ch.preprocess = [copy](const StateSnapshot& s) mutable { return copy->preprocess(s); };
        ch.forward    = [copy](torch::Tensor x) mutable { return copy->forward(x); };
        ch.save       = [copy](const std::string& p) { torch::save(copy, p); };
        ch.load       = [copy](const std::string& p) mutable { torch::load(copy, p); };
        ch.clone_fn   = nullptr;  // inference copies don't need to clone
        return ch;
    };
    return h;
}

// ─── Factory: PatchNet (hierarchical) ────────────────────────────────

inline ModelHandle make_patch_model(int big_rows, int big_cols,
                                     int patch_rows, int patch_cols,
                                     int local_hidden, int local_blocks,
                                     int global_hidden, int global_blocks,
                                     double dropout,
                                     const std::string& local_model_path) {
    PatchNet net(big_rows, big_cols, patch_rows, patch_cols,
                 local_hidden, local_blocks,
                 global_hidden, global_blocks, dropout);

    // Load and freeze pre-trained local model
    if (!local_model_path.empty()) {
        net->load_and_freeze_local(local_model_path);
    }

    ModelHandle h;
    h.module     = net.ptr();
    h.preprocess = [net](const StateSnapshot& s) mutable { return net->preprocess(s); };
    h.forward    = [net](torch::Tensor x) mutable { return net->forward(x); };
    h.save       = [net](const std::string& p) { torch::save(net, p); };
    h.load       = [net](const std::string& p) mutable {
        torch::load(net, p);
        // Re-freeze local model after loading
        net->freeze_local();
    };
    h.clone_fn   = [net](const TrainConfig& cfg) -> ModelHandle {
        // Create an independent PatchNet copy for inference
        PatchNet copy(cfg.rows, cfg.cols, cfg.patch_rows, cfg.patch_cols,
                       cfg.local_hidden_size, cfg.local_num_res_blocks,
                       cfg.global_hidden_size, cfg.global_num_res_blocks,
                       cfg.dropout);
        {
            torch::NoGradGuard no_grad;
            auto src_params = net->named_parameters(/*recurse=*/true);
            for (auto& p : copy->named_parameters(/*recurse=*/true)) {
                auto* found = src_params.find(p.key());
                if (found) p.value().copy_(*found);
            }
            auto src_bufs = net->named_buffers(/*recurse=*/true);
            for (auto& b : copy->named_buffers(/*recurse=*/true)) {
                auto* found = src_bufs.find(b.key());
                if (found) b.value().copy_(*found);
            }
        }
        copy->freeze_local();
        ModelHandle ch;
        ch.module     = copy.ptr();
        ch.preprocess = [copy](const StateSnapshot& s) mutable { return copy->preprocess(s); };
        ch.forward    = [copy](torch::Tensor x) mutable { return copy->forward(x); };
        ch.save       = [copy](const std::string& p) { torch::save(copy, p); };
        ch.load       = [copy](const std::string& p) mutable {
            torch::load(copy, p);
            copy->freeze_local();
        };
        ch.clone_fn   = nullptr;
        return ch;
    };
    return h;
}

// ─── Factory: CNN (convolutional) ───────────────────────────────────────

inline ModelHandle make_cnn_model(int rows, int cols,
                                  int cnn_channels, int blocks, double dropout) {
    AlphaZeroCNNNet net(rows, cols, cnn_channels, blocks, dropout);
    ModelHandle h;
    h.module     = net.ptr();
    h.preprocess = [net](const StateSnapshot& s) mutable { return net->preprocess(s); };
    h.forward    = [net](torch::Tensor x) mutable { return net->forward(x); };
    h.save       = [net](const std::string& p) { torch::save(net, p); };
    h.load       = [net](const std::string& p) mutable {
        torch::load(net, p);
    };
    h.clone_fn   = [net](const TrainConfig& cfg) -> ModelHandle {
        AlphaZeroCNNNet copy(cfg.rows, cfg.cols, cfg.cnn_channels,
                              cfg.num_res_blocks, cfg.dropout);
        {
            torch::NoGradGuard no_grad;
            auto src_params = net->named_parameters();
            for (auto& p : copy->named_parameters()) {
                auto* found = src_params.find(p.key());
                if (found) p.value().copy_(*found);
            }
            auto src_bufs = net->named_buffers();
            for (auto& b : copy->named_buffers()) {
                auto* found = src_bufs.find(b.key());
                if (found) b.value().copy_(*found);
            }
        }
        ModelHandle ch;
        ch.module     = copy.ptr();
        ch.preprocess = [copy](const StateSnapshot& s) mutable { return copy->preprocess(s); };
        ch.forward    = [copy](torch::Tensor x) mutable { return copy->forward(x); };
        ch.save       = [copy](const std::string& p) { torch::save(copy, p); };
        ch.load       = [copy](const std::string& p) mutable { torch::load(copy, p); };
        ch.clone_fn   = nullptr;  // inference copies don't need to clone
        return ch;
    };
    return h;
}

}  // namespace azb
