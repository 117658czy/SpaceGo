#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include "go/heterogeneous/heterogeneous_compute_engine.h"

// 前向声明

namespace spacego {
namespace go {
namespace neural {

// 神经网络输出结构
struct PolicyValueOutput {
    std::vector<float> policy_probs;  // 着法概率分布（361+1个位置，+1表示pass）
    float value_estimate;             // 价值估计（-1到1，黑棋胜率）
    
    PolicyValueOutput() : value_estimate(0.0) {}
    PolicyValueOutput(const std::vector<float>& probs, float value)
        : policy_probs(probs), value_estimate(value) {}
};

// CPU端神经网络接口（与现有HMGS-DP兼容）
class NeuralNetworkCPU {
public:
    NeuralNetworkCPU();
    virtual ~NeuralNetworkCPU();
    
    // 加载模型
    virtual bool load_model(const std::string& model_path);
    
    // 单次推理
    virtual PolicyValueOutput inference(const std::string& board_state) const;
    
    // 批量推理（CPU实现）
    virtual std::vector<PolicyValueOutput> batch_inference(
        const std::vector<std::string>& board_states) const;
    
    // 模型信息
    virtual std::string get_model_info() const;
    virtual bool is_loaded() const { return model_loaded_; }
    
protected:
    bool model_loaded_;
    std::string model_path_;
#ifdef USE_LIBTORCH
    mutable torch::jit::script::Module model_;  // LibTorch模型
#endif
    
private:
    // 将棋盘字符串转换为张量
#ifdef USE_LIBTORCH
    torch::Tensor board_string_to_tensor(const std::string& board_state) const;
    
    // 将批量棋盘字符串转换为张量
    torch::Tensor batch_board_strings_to_tensor(
        const std::vector<std::string>& board_states) const;
#endif
};

#if defined(USE_CUDA) && defined(USE_LIBTORCH)

// ============================================================================
// CUDA事件RAII包装器
// 自动管理CUDA事件生命周期，确保资源在任何情况下都能正确释放
// 避免手动cudaEventCreate/cudaEventDestroy导致的资源泄漏
// ============================================================================
class CudaEventRAII {
public:
    // 默认构造：创建空状态，不分配CUDA事件
    CudaEventRAII() : event_(nullptr) {}

    // 带参数构造：创建CUDA事件
    explicit CudaEventRAII(unsigned int flags) {
        cudaError_t error = cudaEventCreate(&event_, flags);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA事件创建失败: " +
                                    std::string(cudaGetErrorString(error)));
        }
    }

    ~CudaEventRAII() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    // 禁止拷贝
    CudaEventRAII(const CudaEventRAII&) = delete;
    CudaEventRAII& operator=(const CudaEventRAII&) = delete;

    // 允许移动
    CudaEventRAII(CudaEventRAII&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEventRAII& operator=(CudaEventRAII&& other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    // 获取原始事件句柄
    cudaEvent_t get() const { return event_; }

    // 释放所有权（用于返回给调用者，调用者负责销毁）
    cudaEvent_t release() {
        cudaEvent_t tmp = event_;
        event_ = nullptr;
        return tmp;
    }

    // 记录事件到指定流
    void record(cudaStream_t stream = 0) {
        if (event_ == nullptr) {
            throw std::runtime_error("CUDA事件记录失败: 事件未初始化");
        }
        cudaError_t error = cudaEventRecord(event_, stream);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA事件记录失败: " +
                                    std::string(cudaGetErrorString(error)));
        }
    }

    // 同步等待事件完成
    void synchronize() {
        if (event_ == nullptr) {
            throw std::runtime_error("CUDA事件同步失败: 事件未初始化");
        }
        cudaError_t error = cudaEventSynchronize(event_);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA事件同步失败: " +
                                    std::string(cudaGetErrorString(error)));
        }
    }

    // 检查事件是否已初始化
    bool is_valid() const { return event_ != nullptr; }

private:
    cudaEvent_t event_ = nullptr;
};

// ============================================================================
// CUDA流RAII包装器
// 自动管理CUDA流生命周期，确保资源在任何情况下都能正确释放
// 避免手动cudaStreamCreate/cudaStreamDestroy导致的资源泄漏
// ============================================================================
#ifdef USE_CUDA
class CudaStreamRAII {
public:
    // 默认构造：创建空状态，不分配CUDA流
    CudaStreamRAII() : stream_(nullptr) {}

    // 带参数构造：创建CUDA流
    explicit CudaStreamRAII(unsigned int flags) {
        cudaError_t error = cudaStreamCreateWithFlags(&stream_, flags);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA流创建失败: " +
                                    std::string(cudaGetErrorString(error)));
        }
    }

    ~CudaStreamRAII() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // 禁止拷贝
    CudaStreamRAII(const CudaStreamRAII&) = delete;
    CudaStreamRAII& operator=(const CudaStreamRAII&) = delete;

    // 允许移动
    CudaStreamRAII(CudaStreamRAII&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStreamRAII& operator=(CudaStreamRAII&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    // 获取原始流句柄
    cudaStream_t get() const { return stream_; }

    // 释放所有权（调用者负责销毁）
    cudaStream_t release() {
        cudaStream_t tmp = stream_;
        stream_ = nullptr;
        return tmp;
    }

    // 同步等待流中所有操作完成
    void synchronize() {
        if (stream_ == nullptr) {
            throw std::runtime_error("CUDA流同步失败: 流未初始化");
        }
        cudaError_t error = cudaStreamSynchronize(stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA流同步失败: " +
                                    std::string(cudaGetErrorString(error)));
        }
    }

    // 检查流是否已初始化
    bool is_valid() const { return stream_ != nullptr; }

private:
    cudaStream_t stream_ = nullptr;
};
#endif  // USE_CUDA

// GPU端神经网络（CUDA优化）
class NeuralNetworkCUDA {
public:
    NeuralNetworkCUDA();
    ~NeuralNetworkCUDA();
    
    // 禁止拷贝
    NeuralNetworkCUDA(const NeuralNetworkCUDA&) = delete;
    NeuralNetworkCUDA& operator=(const NeuralNetworkCUDA&) = delete;
    
    // 移动构造
    NeuralNetworkCUDA(NeuralNetworkCUDA&& other) noexcept;
    NeuralNetworkCUDA& operator=(NeuralNetworkCUDA&& other) noexcept;
    
    // 模型管理
    bool load_model(const std::string& model_path);
    bool load_model_with_optimization(const std::string& model_path, 
                                      bool use_fp16 = true,
                                      bool use_tensor_core = true);
    void unload_model();
    
    // 推理接口
    PolicyValueOutput inference(const std::string& board_state);
    std::vector<PolicyValueOutput> batch_inference(
        const std::vector<std::string>& board_states);
    
    // 张量输入推理（高性能接口）
    PolicyValueOutput inference_tensor(const torch::Tensor& input_tensor);
    std::vector<PolicyValueOutput> batch_inference_tensor(
        const torch::Tensor& batch_tensor);
    
    // 异步推理接口
    cudaEvent_t async_inference_begin(const torch::Tensor& input_tensor);
    PolicyValueOutput async_inference_end(cudaEvent_t start_event);
    
    // 性能优化
    void set_batch_size(size_t batch_size);
    void enable_mixed_precision(bool enable);
    void enable_tensor_core(bool enable);
    void set_stream(cudaStream_t stream);

    // 异构推理调度接口
    void set_heterogeneous_enabled(bool enabled);
    bool is_heterogeneous_enabled() const;
    bool should_use_cpu_inference(size_t batch_size) const;
    void set_cpu_inference_threshold(size_t threshold);
    size_t get_cpu_inference_threshold() const;
    std::vector<PolicyValueOutput> infer_heterogeneous(
        const std::vector<std::string>& board_states);
    
    // 状态查询
    bool is_loaded() const { return model_loaded_; }
    bool is_using_fp16() const { return use_fp16_; }
    bool is_using_tensor_core() const { return use_tensor_core_; }
    size_t get_batch_size() const { return optimal_batch_size_; }
    std::string get_performance_info() const;
    
    // 内存管理
    size_t get_gpu_memory_usage() const;
    void release_unused_memory();
    
private:
    // 模型和优化
    bool model_loaded_;
    bool use_fp16_;
    bool use_tensor_core_;
    size_t optimal_batch_size_;
    
    // LibTorch模型（支持GPU）
    torch::jit::script::Module model_;
    torch::Device device_;
    
    // CUDA流管理
    cudaStream_t inference_stream_;
    // 使用RAII包装器管理CUDA事件，自动释放资源
    CudaEventRAII inference_start_event_;
    CudaEventRAII inference_end_event_;

    // 输入/输出缓冲区（设备内存）
    // 使用Pimpl模式将thrust::device_vector移至cpp文件，
    // 避免MSVC编译cpp时实例化thrust模板触发C2338错误
    struct CUDABuffers;
    std::unique_ptr<CUDABuffers> cuda_buffers_;
    
    // 临时张量（避免重复分配）
    torch::Tensor input_tensor_cache_;
    torch::Tensor output_policy_cache_;
    torch::Tensor output_value_cache_;

    // 异构推理调度配置
    bool enable_heterogeneous_;              // 异构推理总开关
    size_t cpu_inference_threshold_;         // CPU推理批次阈值（默认16）

    // 性能统计
    mutable size_t total_inferences_;
    mutable double total_inference_time_ms_;
    mutable size_t batch_inference_count_;
    mutable double batch_inference_time_ms_;
    
    // 私有方法
    void initialize_buffers(size_t batch_size);
    void convert_board_to_tensor(const std::string& board_state, 
                                float* tensor_data, size_t batch_idx) const;
    void launch_batch_conversion_kernel(const std::vector<std::string>& board_states,
                                       float* tensor_data_device);
    PolicyValueOutput extract_output_from_buffers(size_t batch_idx) const;
    
    // 性能优化方法
    void optimize_model_for_inference();
    void configure_cudnn_optimizations();
    size_t calculate_optimal_batch_size() const;
};

// 神经网络批处理管理器（用于HMGS-DP并行搜索）
class NeuralNetworkBatchManager {
public:
    NeuralNetworkBatchManager(std::shared_ptr<NeuralNetworkCUDA> network);
    ~NeuralNetworkBatchManager();
    
    // 批次管理
    void set_max_batch_size(size_t max_batch_size);
    void set_timeout_ms(int timeout_ms);
    
    // 异步推理请求
    struct InferenceRequest {
        std::string board_state;
        uint64_t request_id;
        cudaEvent_t completion_event;
        bool completed;
        PolicyValueOutput output;
        
        InferenceRequest() : request_id(0), completed(false) {
            cudaEventCreate(&completion_event, cudaEventDisableTiming);
        }
        
        ~InferenceRequest() {
            cudaEventDestroy(completion_event);
        }
    };
    
    // 提交推理请求
    uint64_t submit_inference_request(const std::string& board_state);
    
    // 批量提交请求
    std::vector<uint64_t> submit_batch_inference_requests(
        const std::vector<std::string>& board_states);
    
    // 获取结果
    bool try_get_result(uint64_t request_id, PolicyValueOutput& output);
    void wait_for_result(uint64_t request_id, PolicyValueOutput& output);
    void wait_for_all_results();
    
    // 批次处理
    void process_pending_requests();
    
    // 状态查询
    size_t get_pending_request_count() const;
    size_t get_processed_request_count() const;
    double get_average_latency_ms() const;
    double get_throughput_inferences_per_sec() const;
    
private:
    std::shared_ptr<NeuralNetworkCUDA> network_;
    size_t max_batch_size_;
    int timeout_ms_;
    
    // 请求管理
    std::unordered_map<uint64_t, std::shared_ptr<InferenceRequest>> pending_requests_;
    std::unordered_map<uint64_t, std::shared_ptr<InferenceRequest>> completed_requests_;
    std::vector<uint64_t> request_queue_;
    uint64_t next_request_id_;
    
    // 性能统计
    size_t total_processed_requests_;
    double total_processing_time_ms_;
    
    // 线程管理
    std::thread processing_thread_;
    std::atomic<bool> stop_processing_;
    std::mutex request_mutex_;
    std::condition_variable request_cv_;
    
    // 私有方法
    void processing_loop();
    void process_batch(const std::vector<uint64_t>& batch_request_ids);
    uint64_t generate_request_id();
};

// CUDA内核声明（棋盘到张量转换）
#ifdef __CUDACC__
extern "C" {
    __global__ void batch_board_strings_to_tensor_kernel(
        const char* board_strings, float* tensor_data,
        size_t batch_size, size_t stride, size_t tensor_stride);

    __global__ void generate_go_features_kernel(
        const uint64_t* black_bits, const uint64_t* white_bits,
        float* feature_tensor, size_t batch_size,
        int feature_planes, int board_size);
}
#endif

// 神经网络性能分析器
class NeuralNetworkProfiler {
public:
    struct ProfileResult {
        double inference_latency_ms;
        double batch_throughput_inferences_per_sec;
        double gpu_utilization;
        double memory_bandwidth_utilization;
        double tensor_core_utilization;
        std::vector<double> layer_timings_ms;
        std::map<std::string, double> detailed_metrics;
    };
    
    NeuralNetworkProfiler(std::shared_ptr<NeuralNetworkCUDA> network);
    
    // 性能分析
    ProfileResult profile_single_inference(int warmup_iterations = 10, 
                                          int measurement_iterations = 100);
    ProfileResult profile_batch_inference(size_t batch_size,
                                         int warmup_iterations = 5,
                                         int measurement_iterations = 50);
    
    // 扩展性分析
    std::map<size_t, ProfileResult> profile_scaling_behavior(
        const std::vector<size_t>& batch_sizes);
    
    // 瓶颈分析
    struct BottleneckAnalysis {
        enum class BottleneckType {
            MEMORY_BOUND,
            COMPUTE_BOUND,
            LATENCY_BOUND,
            KERNEL_LAUNCH_BOUND,
            DATA_TRANSFER_BOUND
        };
        
        BottleneckType primary_bottleneck;
        double bottleneck_severity;  // 0-1
        std::vector<std::string> optimization_suggestions;
    };
    
    BottleneckAnalysis analyze_bottlenecks(const ProfileResult& profile);
    
    // 优化建议
    std::vector<std::string> get_optimization_recommendations();
    
    // 报告生成
    std::string generate_html_report(const ProfileResult& profile);
    std::string generate_csv_report(const std::map<size_t, ProfileResult>& scaling_results);
    
private:
    std::shared_ptr<NeuralNetworkCUDA> network_;
    
    // 性能测量工具（使用RAII包装器自动管理CUDA事件生命周期）
    CudaEventRAII profile_start_event_;
    CudaEventRAII profile_end_event_;
    
    // 私有方法
    void measure_layer_performance(std::vector<double>& layer_timings);
    double measure_gpu_utilization_during_inference();
    double measure_memory_bandwidth_utilization();
};

#endif  // USE_CUDA && USE_LIBTORCH

}  // namespace neural
}  // namespace go
}  // namespace spacego