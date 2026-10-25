#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <random>
#include <fstream>

#include "go/board/go_board_cuda.h"

namespace spacego {
namespace go {

enum class ExpertModelType : uint8_t {
    CA_MOE_NN = 0
};

struct LinkNeuronOutput {
    std::vector<float> prior_probs;
    float value;
    std::vector<float> state_encoding;
    float uncertainty;

    LinkNeuronOutput() : value(0.0f), uncertainty(0.0f) {}
};

class CAMOE {
public:
    CAMOE();
    ~CAMOE();

    LinkNeuronOutput forward(const GoBoardCPU& board);
    LinkNeuronOutput forward_simple(const GoBoardCPU& board);

    bool save_model(const std::string& path);
    bool load_model(const std::string& path);

private:
    static const int STATE_ENCODING_DIM = 64;

    std::vector<float> network_weights_;
    std::vector<float> network_bias_;

    std::unordered_map<std::string, float> performance_history_;

    void initialize_default_parameters();
    void initialize_network();

    float sigmoid(float x) const;
    float relu(float x) const;
    float tanh_activation(float x) const;
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;

    std::vector<float> apply_network(const std::vector<float>& input);
    std::vector<float> prepare_input(const GoBoardCPU& board) const;
};

std::shared_ptr<CAMOE> create_ca_moe();

} // namespace go
} // namespace spacego
