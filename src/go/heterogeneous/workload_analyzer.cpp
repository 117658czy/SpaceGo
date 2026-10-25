// ============================================================================
// 工作负载分析器 - 实现文件
// 命名空间: spacego::go::heterogeneous
// 功能: 实现任务特征提取、设备亲和度计算、专用分析策略
// ============================================================================

#define NOMINMAX
#include "go/heterogeneous/workload_analyzer.h"

#include <algorithm>   // std::max, std::min, std::clamp (C++17)
#include <cmath>       // std::exp, std::abs, std::pow

#undef max
#undef min

namespace spacego {
namespace go {
namespace heterogeneous {

// ============================================================================
// 构造函数与析构函数
// ============================================================================

/**
 * @brief 构造工作负载分析器
 *
 * 初始化设备能力缓存：
 *   - CPU能力通过 detect_cpu() 自动检测
 *   - GPU能力初始化为默认值（无GPU时的安全回退）
 */
WorkloadAnalyzer::WorkloadAnalyzer()
    : cpu_capability_(DeviceCapability::detect_cpu())
    , gpu_capability_()  // GPU能力使用默认构造函数（空设备）
{
    // GPU能力将在 update_device_capabilities() 中被覆盖
    // 或在 calculate_device_affinity() 中作为参数传入
}

/**
 * @brief 析构函数
 *
 * 当前无动态资源需要释放。
 */
WorkloadAnalyzer::~WorkloadAnalyzer()
{
}

// ============================================================================
// 综合任务分析 - 按类别分派到专用策略
// ============================================================================

/**
 * @brief 综合分析任务特征
 *
 * 根据任务类别选择相应的专用分析策略。每种策略都针对该类任务的
 * 计算特征（矩阵密度、访存模式、并行度）进行了优化设计。
 */
TaskFeature WorkloadAnalyzer::analyze_task(TaskCategory category,
                                           size_t batch_size,
                                           size_t element_count,
                                           bool is_tensor_op,
                                           int tensor_dims)
{
    // 防止批次大小为零导致除零错误
    if (batch_size == 0) {
        batch_size = 1;
    }

    // 防止元素数量为零
    if (element_count == 0) {
        element_count = 1;
    }

    // 根据任务类别分派到专用分析策略
    switch (category) {
        case TaskCategory::BOARD_EVALUATION: {
            // 棋盘评估：以棋盘数量作为元素数
            return analyze_board_evaluation(batch_size, element_count);
        }

        case TaskCategory::NEURAL_INFERENCE: {
            // 神经网络推理：使用张量维度信息
            // 如果 tensor_dims 未指定（为0），使用 element_count 作为默认维度
            int effective_dims = (tensor_dims > 0) ? tensor_dims
                                 : static_cast<int>(element_count);
            return analyze_neural_inference(batch_size, effective_dims);
        }

        case TaskCategory::TOPOLOGY_CALCULATION: {
            // 拓扑计算：以区域数量作为数据规模
            return analyze_topology_calculation(batch_size, element_count);
        }

        case TaskCategory::SEARCH_SIMULATION: {
            // 搜索仿真：批次大小是主要参数
            return analyze_search_simulation(batch_size);
        }

        case TaskCategory::GENERIC_COMPUTE: {
            // 通用计算：传递 is_tensor_op 标志以区分张量运算
            return analyze_generic_compute(batch_size, element_count, is_tensor_op);
        }

        default: {
            // 未知类别：回退到通用计算分析
            return analyze_generic_compute(batch_size, element_count);
        }
    }
}

// ============================================================================
// 四维特征单独计算方法
// ============================================================================

/**
 * @brief 计算任务的矩阵/张量密度
 *
 * 矩阵密度表示数据结构的填充率（非零元素占比）：
 *   - 神经网络推理(0.9)：全连接层和卷积层的权重矩阵几乎完全密集
 *                       即使有稀疏权重，占比通常也很小
 *   - 拓扑计算(0.6)：区域连通性矩阵比棋盘状态密集，
 *                    每个区域需要计算与邻居的关系
 *   - 棋盘评估(0.3)：围棋棋盘361个点，实际有子的位置通常不到一半
 *                     开局阶段密度最低（空棋盘），终局密度升高
 *   - 搜索仿真(0.2)：MCTS树节点通常很稀疏，只有已探索的分支有数据
 *   - 通用计算：张量操作默认为高密度(0.7)，非张量操作为中密度(0.5)
 *
 * 批次大小对密度的微调：
 *   - 大批次不会改变数据结构的密度，但可以让GPU更多地利用
 *     合并访存，因此在密度计算中给予小幅加权
 */
double WorkloadAnalyzer::compute_matrix_density(TaskCategory category,
                                                size_t batch_size,
                                                bool is_tensor_op)
{
    double base_density = 0.5;  // 默认中性值

    switch (category) {
        case TaskCategory::BOARD_EVALUATION:
            // 棋盘状态评估：棋盘上非空点比例通常不高
            base_density = 0.30;
            break;

        case TaskCategory::NEURAL_INFERENCE:
            // 神经网络推理：权重矩阵高度密集
            // FP16/FP32混合精度下，权重矩阵填充率接近100%
            base_density = 0.90;
            break;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // 拓扑计算：区域连通性矩阵的密度高于纯棋盘状态
            // 每个交叉点最多有4个邻居，密度约为 4/361 ~= 1%
            // 但以区域为单位时，区域间连接性更高
            base_density = 0.60;
            break;

        case TaskCategory::SEARCH_SIMULATION:
            // 搜索仿真：MCTS树的节点高度稀疏
            // 只有被探索过的路径节点才有数据
            base_density = 0.20;
            break;

        case TaskCategory::GENERIC_COMPUTE:
            // 通用计算：根据是否为张量操作判断
            if (is_tensor_op) {
                // 张量操作通常涉及密集矩阵运算
                base_density = 0.70;
            } else {
                // 非张量通用操作，保守估计
                base_density = 0.50;
            }
            break;

        default:
            base_density = 0.50;
            break;
    }

    // 批次大小的微调：大批次任务在GPU上可以更好地利用合并访存
    // 在密度基值上给予小幅加成（最高+0.05，避免喧宾夺主）
    double batch_bonus = 0.0;
    if (batch_size >= GPU_MIN_BATCH_THRESHOLD) {
        // 大批次：密度加成按对数增长，上限0.05
        batch_bonus = std::min(0.05, std::log2(static_cast<double>(batch_size)) * 0.005);
    } else if (batch_size <= SMALL_BATCH_THRESHOLD) {
        // 小批次：密度小幅惩罚，因为有效计算密度下降
        batch_bonus = -0.02;
    }

    // 最终密度值限制在 [0.0, 1.0] 范围内
    return std::clamp(base_density + batch_bonus, 0.0, 1.0);
}

/**
 * @brief 计算任务的数据规模
 *
 * 数据规模 = 基本元素数 * 批次大小 * 维度放大因子
 *
 * 维度放大因子的含义：
 *   - 对于一维数据（向量）：数据量 = N
 *   - 对于二维数据（矩阵）：数据量 = N^2
 *   - 对于三维数据（张量，如卷积特征图）：数据量 = N^3
 *   - 维度越高，实际数据量呈指数增长
 *
 * 针对不同任务类别的特殊处理：
 *   - 棋盘评估：每个棋盘361个交叉点，但实际有棋子的点更少
 *   - 神经网络推理：张量维度决定矩阵乘法规模
 *   - 搜索仿真：按每步仿真3-5倍分支因子估算
 */
size_t WorkloadAnalyzer::compute_data_scale(TaskCategory category,
                                            size_t batch_size,
                                            size_t element_count)
{
    // 防止输入为零
    if (batch_size == 0) {
        batch_size = 1;
    }
    if (element_count == 0) {
        element_count = 1;
    }

    size_t base_scale = element_count * batch_size;

    switch (category) {
        case TaskCategory::BOARD_EVALUATION: {
            // 棋盘评估：每个棋盘361个点
            // 如果传入的 element_count 大于361（如批量棋盘数），使用实际值
            // 否则使用围棋标准棋盘大小
            size_t points_per_board = (element_count >= 361)
                                      ? element_count
                                      : 361;  // 默认19路棋盘
            return points_per_board * batch_size;
        }

        case TaskCategory::NEURAL_INFERENCE: {
            // 神经网络推理：element_count 可能代表张量维度
            // 数据量与维度平方成正比（全连接层矩阵乘法：N x batch x N）
            // 例如：element_count=256 表示 256x256 的权重矩阵
            // 数据规模 = 维度 * 维度 * 批次大小
            // 防止溢出：使用 double 进行中间计算，然后截断为 size_t
            double scale = static_cast<double>(element_count)
                         * static_cast<double>(element_count)
                         * static_cast<double>(batch_size);
            // 限制上限，避免溢出
            constexpr double MAX_SCALE = 1e12;  // 1万亿元素
            return static_cast<size_t>(std::min(scale, MAX_SCALE));
        }

        case TaskCategory::TOPOLOGY_CALCULATION: {
            // 拓扑计算：每个区域需要检查与周围区域的连接关系
            // 估算每个区域平均有4个邻居 + 1个自身 = 5个操作
            // 数据规模 = 区域数 * 批次大小 * 5
            return element_count * batch_size * 5;
        }

        case TaskCategory::SEARCH_SIMULATION: {
            // 搜索仿真：每个仿真包含多个步骤
            // 平均每步生成约4-6个候选着法
            // 估算数据规模 = batch_size * 平均仿真步数(200) * 平均分支因子(5)
            // 但实际的batch_size直接代表了仿真数量
            // 使用保守估算：batch_size * 200步 * 平均5个候选 = batch_size * 1000
            return batch_size * 1000;
        }

        case TaskCategory::GENERIC_COMPUTE:
        default: {
            // 通用计算：直接使用 batch_size * element_count
            return base_scale;
        }
    }
}

/**
 * @brief 推断任务的访存模式
 *
 * 访存模式对设备选择的影响：
 *
 * SEQUENTIAL（顺序访问）：
 *   GPU优势：合并访存 (coalesced access) 使单个warp的32个线程
 *            在一次内存事务中完成数据读取，带宽利用率极高
 *   CPU优势：硬件预取器 (prefetcher) 能提前加载连续地址的数据
 *
 * STRIDED（跨步访问）：
 *   GPU劣势：跨步大于缓存行大小时，合并访存失效，
 *            每个线程独立访问内存，带宽利用率急剧下降
 *   CPU劣势：跨步超出L1缓存行时同样失效
 *   总体上：步长越大，两种设备效率都越低，GPU下降更显著
 *
 * RANDOM（随机访问）：
 *   GPU劣势：warp内各线程访问分散地址，需要多次内存事务
 *            GPU显存延迟远高于CPU主存（~400 cycles vs ~200 cycles）
 *   CPU优势：多级缓存层次（L1/L2/L3）更适合处理随机访问
 *            乱序执行可以隐藏部分延迟
 *
 * MIXED（混合模式）：
 *   需要动态分析，保守采用MIXED标记
 */
MemoryAccessPattern WorkloadAnalyzer::compute_memory_pattern(TaskCategory category,
                                                              bool is_tensor_op)
{
    switch (category) {
        case TaskCategory::BOARD_EVALUATION:
            // 棋盘评估：需要扫描棋盘上各位置的棋子状态
            // 评估函数可能需要访问对角线、星位、边角等非连续位置
            // 走子生成需要检查每个空点的合法性，访问模式偏随机
            return MemoryAccessPattern::RANDOM;

        case TaskCategory::NEURAL_INFERENCE:
            // 神经网络推理：矩阵乘法按行/列顺序访问权重
            // 卷积运算按通道顺序访问特征图
            // 批归一化按批次顺序处理
            // 整体上以顺序访问为主
            return MemoryAccessPattern::SEQUENTIAL;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // 拓扑计算：以连通区域为单位进行跨步访问
            // 例如：气计数需要检查相邻4个点
            //       区域遍历按棋盘行扫描，但每行只处理属于该区域的点
            //       跨步大小为棋盘宽度19的倍数
            return MemoryAccessPattern::STRIDED;

        case TaskCategory::SEARCH_SIMULATION:
            // 搜索仿真：混合了多种访存模式
            // - 走子生成：棋盘上的规则扫描（顺序/跨步）
            // - 树节点回溯：沿树路径向上更新（随机）
            // - 叶节点评估：神经网络推理（顺序）
            return MemoryAccessPattern::MIXED;

        case TaskCategory::GENERIC_COMPUTE:
        default:
            // 通用计算：根据 is_tensor_op 标志判断
            if (is_tensor_op) {
                // 张量操作默认顺序访问
                return MemoryAccessPattern::SEQUENTIAL;
            } else {
                // 非张量操作保守使用混合模式
                return MemoryAccessPattern::MIXED;
            }
    }
}

/**
 * @brief 计算任务的内在并行度
 *
 * 内在并行度（Intrinsic Parallelism Degree）表示任务中可同时执行的
 * 独立工作单元占总工作量的比例。
 *
 * 并行度分级参考：
 *   0.95+ : 极高并行 - 每个数据元素独立计算，无需同步
 *           （如：神经网络大批次推理、向量逐元素运算）
 *   0.85  : 高并行   - 大部分工作可并行，少量需要汇总
 *           （如：棋盘评估，各棋盘独立但最终需要聚合结果）
 *   0.75  : 中高并行 - 存在轻度依赖，多数路径可并行
 *           （如：树搜索中不同分支的并行探索）
 *   0.70  : 中并行   - 存在区域间依赖，部分任务必须串行
 *           （如：拓扑计算中某些区域依赖相邻区域的结果）
 *   0.50  : 低并行   - 小批次或串行主导的任务
 *           （如：单次或少量推理请求）
 *
 * 批次大小对并行度的影响：
 *   - 批次大小越大，每个元素独立计算的机会越多
 *   - 使用 sigmoid 函数平滑过渡：从0.5到基础并行度
 */
double WorkloadAnalyzer::compute_parallelism_degree(TaskCategory category,
                                                    size_t batch_size)
{
    // 防止批次大小为零
    if (batch_size == 0) {
        batch_size = 1;
    }

    // 第一步：确定任务类别的基准并行度
    double base_parallelism = 0.5;  // 默认值

    switch (category) {
        case TaskCategory::BOARD_EVALUATION:
            // 棋盘评估：每块棋盘独立评估，互不依赖
            // 评估函数内部可能包含全局特征（如领地估算），
            // 但这些可以在单块棋盘内完成，不需要棋盘间通信
            base_parallelism = 0.85;
            break;

        case TaskCategory::NEURAL_INFERENCE:
            // 神经网络推理：每个样本独立前向传播
            // 矩阵乘法天然高度并行
            // 大批次时每个样本完全独立，并行度接近1.0
            base_parallelism = 0.95;
            break;

        case TaskCategory::TOPOLOGY_CALCULATION:
            // 拓扑计算：区域间可能存在依赖关系
            // 例如：判断某块棋的死活需要先知道周围区域的状态
            // 但不同棋盘的拓扑计算是完全独立的
            base_parallelism = 0.70;
            break;

        case TaskCategory::SEARCH_SIMULATION:
            // 搜索仿真：MCTS的不同根节点分支可以并行探索
            // 但树内节点有父子依赖，叶节点评估后需要回溯更新
            // 根并行化（Root Parallelization）可以达到较高并行度
            base_parallelism = 0.75;
            break;

        case TaskCategory::GENERIC_COMPUTE:
        default:
            // 通用计算：保守估算
            base_parallelism = 0.65;
            break;
    }

    // 第二步：根据批次大小调整并行度
    // 使用 sigmoid 函数平滑过渡：
    //   批次很小时    -> 并行度接近0.5（受限于少量任务）
    //   批次中等时    -> 并行度逐渐接近基础值
    //   批次很大时    -> 并行度接近基础值（已达上限）
    //
    // Sigmoid 参数设计：
    //   中心点 = SMALL_BATCH_THRESHOLD (8)
    //   斜率 = 0.3（适中过渡速度）
    //
    // 当 batch_size = 1:  sigmoid ~= 0.1, final ~= 0.5（低并行）
    // 当 batch_size = 8:  sigmoid ~= 0.5, final ~= 0.7（中等）
    // 当 batch_size = 64: sigmoid ~= 0.99, final ~= base（高并行）

    double batch_factor_x = static_cast<double>(batch_size)
                          - static_cast<double>(SMALL_BATCH_THRESHOLD);
    // Sigmoid: 1 / (1 + exp(-x / slope))
    double sigmoid = 1.0 / (1.0 + std::exp(-batch_factor_x / 3.0));

    // 最终并行度 = 0.5 + (base - 0.5) * sigmoid
    // 这样当 batch_size 极小时，并行度接近0.5
    // 当 batch_size 极大时，并行度接近基础值
    double adjusted = 0.50 + (base_parallelism - 0.50) * sigmoid;

    // 限制在 [0.0, 1.0] 范围内
    return std::clamp(adjusted, 0.0, 1.0);
}

// ============================================================================
// 设备亲和度计算 - 核心决策算法
// ============================================================================

/**
 * @brief 计算任务对CPU和GPU的设备亲和度评分
 *
 * 这是整个工作负载分析器的核心算法。采用加权评分模型，
 * 综合任务特征和硬件能力，输出量化的设备选择建议。
 *
 * ## GPU亲和度计算公式（详细分解）：
 *
 * Step 1 - 计算 GPU 基础分（不含数据规模调整）：
 *   gpu_base = matrix_density    * 0.40    // 矩阵密度：权重40%
 *            + parallelism_degree * 0.30   // 并行度：  权重30%
 *            + seq_mem_bonus     * 0.15    // 顺序访存：权重15%
 *            + tensor_core_bonus * 0.15    // TensorCore: 权重15%
 *
 * 权重设计原理：
 *   - 矩阵密度权重最高(40%)：GPU架构(大量CUDA核心做密集运算)的核心理念
 *   - 并行度次高(30%)：GPU拥有数千核心，高并行度任务才能充分利用
 *   - 顺序访存(15%)：GPU合并访存对有规则访问模式的加成明显
 *   - Tensor Core(15%)：专用硬件加速器对特定工作负载的额外加成
 *
 * Step 2 - 应用数据规模因子：
 *   data_scale_factor = 1.0 - exp(-data_scale / GPU_MIN_DATA_SCALE)
 *
 *   - 数据量很小时(<< threshold): factor ~= 0.1（GPU不占优势）
 *   - 数据量中等时(= threshold): factor ~= 0.63
 *   - 数据量很大时(>> threshold): factor ~= 1.0（GPU优势充分发挥）
 *
 * Step 3 - GPU亲和度最终值：
 *   gpu_affinity = gpu_base * data_scale_factor
 *
 * ## CPU亲和度计算公式：
 *
 *   cpu_affinity = (1.0 - gpu_affinity)
 *                * random_mem_bonus    // 随机访存给CPU加分
 *                * small_batch_bonus   // 小批次任务CPU更优
 *
 * 逻辑：GPU不适合做的任务(1-gpu_affinity高) + CPU擅长的方面(bonus高)
 *       = CPU亲和度高
 *
 * ## 决策与置信度：
 *
 *   推荐设备 = argmax(cpu_affinity, gpu_affinity)
 *   置信度   = |gpu_affinity - cpu_affinity| / (gpu_affinity + cpu_affinity)
 *
 *   当GPU亲和度和CPU亲和度差距很大时，置信度高（决策明确）
 *   当两者接近时，置信度低（需要调度器进一步综合判断）
 */
DeviceAffinity WorkloadAnalyzer::calculate_device_affinity(
    const TaskFeature& feature,
    const DeviceCapability& cpu_cap,
    const DeviceCapability& gpu_cap)
{
    DeviceAffinity result;

    // ====================================================================
    // 安全检查：GPU是否可用
    // 如果GPU核心数为0（未检测到GPU或GPU不可用），直接返回CPU推荐
    // ====================================================================
    if (gpu_cap.compute_cores == 0 || gpu_cap.total_memory_mb == 0) {
        result.cpu_affinity_score = 1.0;
        result.gpu_affinity_score = 0.0;
        result.recommended_device = ComputeDeviceType::CPU;
        result.confidence = 1.0;
        return result;
    }

    // ====================================================================
    // Step 1: 计算顺序访存奖励 (sequential_mem_bonus)
    // ====================================================================
    // GPU对顺序访存的合并访问极为高效，给予额外奖励
    // 分值范围：0.3(随机访问) ~ 1.0(顺序访问)
    double seq_mem_bonus = 0.3;  // 默认：随机访问的低奖励
    switch (feature.mem_pattern) {
        case MemoryAccessPattern::SEQUENTIAL:
            // 顺序访问：GPU可合并访存，一个warp的32线程只需1-2次内存事务
            seq_mem_bonus = 1.0;
            break;
        case MemoryAccessPattern::STRIDED:
            // 跨步访问：如果步长是缓存行对齐的，仍有部分合并效率
            // 不如纯顺序访问，但优于随机访问
            seq_mem_bonus = 0.65;
            break;
        case MemoryAccessPattern::RANDOM:
            // 随机访问：GPU的合并访存几乎完全失效
            // 显存延迟高（~400 cycles），CPU缓存层次更优
            seq_mem_bonus = 0.30;
            break;
        case MemoryAccessPattern::MIXED:
            // 混合模式：保守取中间值
            seq_mem_bonus = 0.55;
            break;
        default:
            seq_mem_bonus = 0.50;
            break;
    }

    // ====================================================================
    // Step 2: 计算 Tensor Core 奖励 (tensor_core_bonus)
    // ====================================================================
    // Tensor Core 是 NVIDIA Volta+ 架构的专用矩阵运算硬件
    // 在 FP16 精度下可提供数倍于标准 CUDA Core 的吞吐量
    // 但仅对密集矩阵运算有效（稀疏矩阵获益不明显）
    double tensor_core_bonus = 0.30;  // 默认：无Tensor Core或低收益

    if (gpu_cap.supports_tensor_core) {
        if (feature.matrix_density > 0.80) {
            // 高密度矩阵：Tensor Core效果最好
            // 例如：全连接层的权重矩阵乘法
            tensor_core_bonus = 1.0;
        } else if (feature.matrix_density > 0.60) {
            // 中高密度：Tensor Core仍有明显收益
            // 例如：部分稀疏的卷积层
            tensor_core_bonus = 0.75;
        } else if (feature.matrix_density > 0.40) {
            // 中等密度：Tensor Core收益有限
            tensor_core_bonus = 0.50;
        } else {
            // 低密度：Tensor Core的专用硬件设计不适用于稀疏结构
            tensor_core_bonus = 0.35;
        }
    }
    // 如果GPU不支持Tensor Core，保持默认值0.30

    // ====================================================================
    // Step 3: 计算 GPU 基础分
    // ====================================================================
    double gpu_base = feature.matrix_density    * 0.40
                    + feature.parallelism_degree * 0.30
                    + seq_mem_bonus              * 0.15
                    + tensor_core_bonus          * 0.15;

    // ====================================================================
    // Step 4: 计算数据规模因子 (data_scale_factor)
    // ====================================================================
    // GPU的并行优势需要足够大的数据规模才能体现。
    // 数据量过小时，GPU内核启动开销、数据传输开销会超过计算收益。
    //
    // 使用指数衰减函数：factor = 1 - exp(-scale / threshold)
    //
    // GPU_MIN_DATA_SCALE 设计为 GPU_MIN_BATCH_THRESHOLD * 基准元素数
    // 基准元素数取 1000（中等规模任务的典型值）
    static constexpr double GPU_MIN_DATA_SCALE = 16000.0;  // 16 * 1000
    double scale_ratio = static_cast<double>(feature.data_scale)
                       / GPU_MIN_DATA_SCALE;
    double data_scale_factor = 1.0 - std::exp(-scale_ratio);

    // ====================================================================
    // Step 5: 计算最终 GPU 亲和度
    // ====================================================================
    double gpu_affinity = gpu_base * data_scale_factor;

    // 限制在 [0.0, 1.0] 范围内
    gpu_affinity = std::clamp(gpu_affinity, 0.0, 1.0);

    // ====================================================================
    // Step 6: 计算随机访存奖励 (random_mem_bonus，为CPU加分)
    // ====================================================================
    // CPU拥有多级缓存层次（L1/L2/L3），对随机访问有更好的容忍度。
    // 乱序执行可以进一步隐藏内存延迟。
    // GPU显存延迟高，随机访问会导致warp停顿。
    double random_mem_bonus = 1.0;  // 默认：无特殊加成
    switch (feature.mem_pattern) {
        case MemoryAccessPattern::RANDOM:
            // 随机访问：CPU缓存层次 + 乱序执行优于GPU
            random_mem_bonus = 1.35;
            break;
        case MemoryAccessPattern::MIXED:
            // 混合模式：CPU有一定优势
            random_mem_bonus = 1.20;
            break;
        case MemoryAccessPattern::STRIDED:
            // 跨步访问：小步长时CPU缓存可命中
            random_mem_bonus = 1.10;
            break;
        case MemoryAccessPattern::SEQUENTIAL:
            // 顺序访问：CPU和GPU都高效，但GPU合并访存更优
            // 不给CPU额外奖励
            random_mem_bonus = 1.0;
            break;
        default:
            random_mem_bonus = 1.0;
            break;
    }

    // ====================================================================
    // Step 7: 计算小批次奖励 (small_batch_bonus，为CPU加分)
    // ====================================================================
    // 批次过小时，GPU的内核启动开销（~20us）可能超过实际计算时间
    // 数据传输开销（CPU->GPU->CPU）也无法被批量处理平摊
    double small_batch_bonus = 1.0;  // 默认：无特殊加成

    // 注意：feature.data_scale 已经包含了批次信息
    // 但这里需要更细粒度的批次判断
    // 使用 data_scale 反推等效批次大小
    // 等效批次 = data_scale / (平均元素数，保守取1000)
    size_t effective_batch = static_cast<size_t>(
        static_cast<double>(feature.data_scale) / 1000.0);
    // 如果等效批次太小，使用默认保守值
    if (effective_batch < 1) {
        effective_batch = 1;
    }

    if (effective_batch <= SMALL_BATCH_THRESHOLD) {
        // 极小批次：GPU开销远大于收益
        // 强烈推荐CPU执行
        small_batch_bonus = 2.0;
    } else if (effective_batch <= GPU_MIN_BATCH_THRESHOLD) {
        // 小批次：仍推荐CPU，但程度减轻
        // 使用线性衰减
        double ratio = static_cast<double>(effective_batch - SMALL_BATCH_THRESHOLD)
                     / static_cast<double>(GPU_MIN_BATCH_THRESHOLD - SMALL_BATCH_THRESHOLD);
        small_batch_bonus = 2.0 - ratio * 0.8;  // 从2.0线性衰减到1.2
    } else {
        // 足够大的批次：GPU启动开销被充分平摊
        small_batch_bonus = 1.0;
    }

    // ====================================================================
    // Step 8: 计算最终 CPU 亲和度
    // ====================================================================
    // CPU亲和度 = (1 - GPU亲和度的互补部分)
    //            * 随机访存奖励（CPU的优势）
    //            * 小批次奖励（CPU的优势）
    double cpu_affinity = (1.0 - gpu_affinity)
                        * random_mem_bonus
                        * small_batch_bonus;

    // 限制在 [0.0, 1.0] 范围内
    cpu_affinity = std::clamp(cpu_affinity, 0.0, 1.0);

    // ====================================================================
    // Step 9: 归一化处理
    // ====================================================================
    // 确保CPU和GPU的分数之和在合理范围内
    // 防止两个分数都很高（如gpu=0.9, cpu=0.9）的情况
    double total = cpu_affinity + gpu_affinity;
    if (total > 1.5) {
        // 总和过高时按比例缩放
        double scale = 1.5 / total;
        gpu_affinity *= scale;
        cpu_affinity *= scale;
    }

    // ====================================================================
    // Step 10: 确定推荐设备和置信度
    // ====================================================================
    double score_diff = std::abs(gpu_affinity - cpu_affinity);
    double score_sum = gpu_affinity + cpu_affinity;

    if (score_sum > 0.001) {
        // 置信度 = 两分数的|差值| / 两分数之和
        // 差值越大，决策越明确，置信度越高
        result.confidence = score_diff / score_sum;

        // 额外考虑：如果数据规模非常大且GPU可用，
        // 略微提升置信度（大规模并行GPU几乎总是最优）
        if (feature.data_scale > 1000000) {  // 1M元素以上
            result.confidence = std::min(1.0, result.confidence * 1.15);
        }
    } else {
        // 两个分数都接近0（极少见的情况）
        result.confidence = 0.5;
    }

    // 根据分数高低决定推荐设备
    if (gpu_affinity > cpu_affinity) {
        result.recommended_device = ComputeDeviceType::GPU;
        // GPU推荐但分数接近时降低置信度
        if (score_diff < 0.15) {
            result.confidence = std::min(result.confidence, 0.65);
        }
    } else if (cpu_affinity > gpu_affinity) {
        result.recommended_device = ComputeDeviceType::CPU;
        // CPU推荐但分数接近时降低置信度
        if (score_diff < 0.15) {
            result.confidence = std::min(result.confidence, 0.65);
        }
    } else {
        // 分数完全相等：选择AUTO让调度器决定
        result.recommended_device = ComputeDeviceType::AUTO;
        result.confidence = 0.30;  // 低置信度
    }

    // 赋值最终分数
    result.cpu_affinity_score = std::clamp(cpu_affinity, 0.0, 1.0);
    result.gpu_affinity_score = std::clamp(gpu_affinity, 0.0, 1.0);

    return result;
}

// ============================================================================
// 设备能力管理
// ============================================================================

/**
 * @brief 更新内部缓存的设备能力信息
 *
 * 典型用法：
 * @code
 *   WorkloadAnalyzer analyzer;
 *   analyzer.update_device_capabilities(
 *       DeviceCapability::detect_cpu(),
 *       DeviceCapability::detect_gpu(0));
 * @endcode
 */
void WorkloadAnalyzer::update_device_capabilities(
    const DeviceCapability& cpu_cap,
    const DeviceCapability& gpu_cap)
{
    cpu_capability_ = cpu_cap;
    gpu_capability_ = gpu_cap;
}

// ============================================================================
// 专用分析策略 - 棋盘评估
// ============================================================================

/**
 * @brief 棋盘评估专用特征分析
 *
 * 围棋棋盘评估的核心特征：
 *   1. 访存模式：随机 - 评估函数需要访问棋盘中任意位置
 *      （对角线、星位、边角等），这些位置在内存中不连续
 *   2. 并行度：高 - 不同棋盘完全独立，可以在GPU上并行评估
 *      数千块棋盘
 *   3. 数据规模：中等 - 单个棋盘361点 * 批次大小
 *      乘以特征通道数（约17-20通道）后会增大
 *   4. 矩阵密度：低 - 棋盘上大部分位置为空，或只有简单标记
 *
 * GPU适用性分析：
 *   - 大批次(>=32)棋盘评估适合GPU：CUDA核心可同时处理大量独立棋盘
 *   - 需要开发专用的CUDA内核，优化随机访问模式
 *   - 使用位棋盘(Bitboard)表示可减少全局内存访问量
 *
 * CPU适用性分析：
 *   - 小批次(<16)棋盘评估适合CPU：避免GPU启动开销
 *   - CPU的复杂逻辑判断更适合围棋规则处理（劫、眼等）
 *
 * @param batch_size  批次大小（同时评估的棋盘数量）
 * @param board_count 每批棋盘数量（通常与batch_size相同）
 */
TaskFeature WorkloadAnalyzer::analyze_board_evaluation(size_t batch_size,
                                                        size_t board_count)
{
    TaskFeature feature;

    // ---- 1. 矩阵密度 ----
    // 棋盘评估的数据结构（棋盘状态）是稀疏的
    // 位棋盘表示中，uint64_t的每个bit代表一个交叉点
    // 通常只有不到一半的点有棋子
    feature.matrix_density = compute_matrix_density(
        TaskCategory::BOARD_EVALUATION, batch_size, false);

    // ---- 2. 数据规模 ----
    // 每个棋盘需要评估的特征包括：
    //   - 361个交叉点的状态信息
    //   - 每个交叉点约17-20个特征通道
    // 基准计算：batch_size * board_count * 361
    // 但实际评估中会使用位棋盘压缩表示，减少内存占用
    feature.data_scale = compute_data_scale(
        TaskCategory::BOARD_EVALUATION, batch_size,
        // 使用361（标准围棋棋盘交叉点数）作为每个实例的元素数
        std::max(board_count, static_cast<size_t>(361)));

    // ---- 3. 访存模式 ----
    // 随机访问：评估函数遍历棋盘时访问模式不规则
    feature.mem_pattern = compute_memory_pattern(
        TaskCategory::BOARD_EVALUATION, false);

    // ---- 4. 并行度 ----
    // 每块棋盘独立评估，互不依赖 -> 高并行度
    feature.parallelism_degree = compute_parallelism_degree(
        TaskCategory::BOARD_EVALUATION, batch_size);

    return feature;
}

// ============================================================================
// 专用分析策略 - 神经网络推理
// ============================================================================

/**
 * @brief 神经网络推理专用特征分析
 *
 * 神经网络推理的核心特征：
 *   1. 访存模式：顺序 - 权重矩阵连续存储，按行/列顺序读取
 *      卷积核也可以连续读取
 *   2. 并行度：极高 - 矩阵乘法可以分解为大量独立的内积运算
 *      每个输出元素都是所有输入元素的加权和，这些加权和互相独立
 *   3. 数据规模：可大可小 - 从单次推理到数千批次推理
 *      tensor_dims参数控制模型规模大小
 *   4. 矩阵密度：最高 - 全连接层权重矩阵密集度接近100%
 *
 * GPU适用性分析：
 *   - GPU是神经网络推理的自然选择：
 *     * CUDA核心非常适合矩阵乘法的并行化
 *     * Tensor Core可以将FP16矩阵乘法提速数倍
 *     * cuDNN/cuBLAS库提供了高度优化的实现
 *   - FP16混合精度在GPU上可以进一步提升吞吐量
 *   - 大批次推理可充分利用GPU的高内存带宽
 *
 * CPU适用性分析：
 *   - 仅在以下情况考虑CPU：
 *     * 极小型网络（tensor_dims < 32）
 *     * 单次推理（batch_size = 1），GPU启动开销 > 计算收益
 *     * 需要频繁的设备间数据传输
 *
 * @param batch_size  批次大小（同时推理的样本数量）
 * @param tensor_dims 张量维度（网络的输入/隐藏层大小）
 */
TaskFeature WorkloadAnalyzer::analyze_neural_inference(size_t batch_size,
                                                        int tensor_dims)
{
    TaskFeature feature;

    // 确保张量维度有效
    if (tensor_dims <= 0) {
        tensor_dims = 256;  // 使用典型的神经网络维度作为默认值
    }

    // ---- 1. 矩阵密度 ----
    // 神经网络权重矩阵高度密集
    // 即使有少量的Dropout或稀疏化，整体密度仍 > 0.85
    feature.matrix_density = compute_matrix_density(
        TaskCategory::NEURAL_INFERENCE, batch_size, true);

    // ---- 2. 数据规模 ----
    // 神经网络推理的数据规模取决于：
    //   - 模型大小（tensor_dims决定了权重矩阵的大小）
    //   - 批次大小（batch_size决定了同时处理的样本数）
    //   - 层数（多层的累积数据量）
    //
    // 对于推理任务，主要数据是：
    //   - 输入特征：batch_size * tensor_dims
    //   - 权重矩阵：tensor_dims * tensor_dims（每层）
    //   - 输出：batch_size * tensor_dims
    //
    // 矩阵乘法的计算量：O(batch_size * tensor_dims^2)
    // 使用 tensor_dims 作为 element_count，compute_data_scale
    // 内部会将 element_count^2 * batch_size 作为数据规模
    feature.data_scale = compute_data_scale(
        TaskCategory::NEURAL_INFERENCE, batch_size,
        static_cast<size_t>(tensor_dims));

    // ---- 3. 访存模式 ----
    // 权重矩阵按行/列顺序读取 -> 顺序访问
    feature.mem_pattern = compute_memory_pattern(
        TaskCategory::NEURAL_INFERENCE, true);

    // ---- 4. 并行度 ----
    // 神经网络推理的极高并行度：
    //   每个样本独立前向传播
    //   矩阵乘法每个内积独立计算
    feature.parallelism_degree = compute_parallelism_degree(
        TaskCategory::NEURAL_INFERENCE, batch_size);

    return feature;
}

// ============================================================================
// 专用分析策略 - 拓扑计算
// ============================================================================

/**
 * @brief 拓扑计算专用特征分析
 *
 * 拓扑计算的核心特征：
 *   1. 访存模式：跨步 - 以连通区域（棋串）为单位访问棋盘
 *      访问相邻4个方向，步长=1或19（棋盘宽度）
 *   2. 并行度：中等 - 同一棋盘内的不同区域可能有拓扑依赖
 *      （如两块棋共享气），但不同棋盘的拓扑计算完全独立
 *   3. 数据规模：region_count * batch_size * 邻居因子(5)
 *   4. 矩阵密度：中等 - 拓扑关系比纯棋盘状态密集，
 *      但不如矩阵乘法的密度高
 *
 * 拓扑计算包含的操作：
 *   - 连通性分析：判断棋串是否连通
 *   - 气计数：计算棋串的当前气数
 *   - 眼位检测：识别真正的眼和假眼
 *   - 死活判断：判断棋串的生死状态
 *   - 区域划分：将棋盘划分为不同的势力范围
 *
 * GPU适用性分析：
 *   - 大批量棋盘拓扑分析：GPU可以并行处理
 *   - 但不适合作为GPU的主要优势场景
 *
 * @param batch_size   批次大小（同时分析的棋盘数量）
 * @param region_count 每块棋盘的区域/棋串数量
 */
TaskFeature WorkloadAnalyzer::analyze_topology_calculation(size_t batch_size,
                                                            size_t region_count)
{
    TaskFeature feature;

    // 确保区域数量有效
    if (region_count == 0) {
        region_count = 1;
    }

    // ---- 1. 矩阵密度 ----
    // 拓扑计算的区域连通性矩阵密度高于纯棋盘评估
    // 每个区域需要检查与所有其他区域的关系
    feature.matrix_density = compute_matrix_density(
        TaskCategory::TOPOLOGY_CALCULATION, batch_size, false);

    // ---- 2. 数据规模 ----
    // 拓扑计算规模：每个区域平均检查约5个邻居（上下左右+自身）
    // 再加上区域间的连接关系检查
    feature.data_scale = compute_data_scale(
        TaskCategory::TOPOLOGY_CALCULATION, batch_size, region_count);

    // ---- 3. 访存模式 ----
    // 跨步访问：沿棋盘扫描连通区域，步长受棋盘宽度(19)影响
    feature.mem_pattern = compute_memory_pattern(
        TaskCategory::TOPOLOGY_CALCULATION, false);

    // ---- 4. 并行度 ----
    // 拓扑计算并行度：不同棋盘完全独立，但同一棋盘内区域有依赖
    // 整体并行度高于串行任务，但低于完全独立的任务
    feature.parallelism_degree = compute_parallelism_degree(
        TaskCategory::TOPOLOGY_CALCULATION, batch_size);

    return feature;
}

// ============================================================================
// 专用分析策略 - 搜索仿真
// ============================================================================

/**
 * @brief 搜索仿真专用特征分析
 *
 * MCTS搜索仿真的核心特征：
 *   1. 访存模式：混合 -
 *      - 走子生成：规则性扫描棋盘空点（顺序/跨步）
 *      - 叶节点评估：神经网络推理（顺序）
 *      - 回溯更新：沿树路径向上传播（随机）
 *   2. 并行度：中高 -
 *      - 根并行化(Root Parallelization)：不同根节点分支独立探索
 *      - 叶并行化(Leaf Parallelization)：不同仿真独立进行
 *      - 树并行化(Tree Parallelization)：需要全局锁或事务内存
 *   3. 数据规模：batch_size * 仿真步数 * 分支因子
 *   4. 矩阵密度：低 - MCTS树节点结构稀疏
 *
 * 搜索仿真的典型数据流：
 *   1. 从根节点选择路径（树策略，如UCB）
 *   2. 在叶节点展开（添加新节点）
 *   3. 执行快速走子（rollout policy）
 *   4. 回溯更新路径上所有节点的统计信息
 *
 * GPU适用性分析：
 *   - 大批量独立仿真适合GPU：数千个仿真可并行
 *   - 但树结构更新需要同步，可能成为GPU瓶颈
 *   - GPU端MCTS实现复杂度高于CPU端
 *
 * @param batch_size 并行仿真的数量
 */
TaskFeature WorkloadAnalyzer::analyze_search_simulation(size_t batch_size)
{
    TaskFeature feature;

    // ---- 1. 矩阵密度 ----
    // MCTS树节点数据高度稀疏：
    //   - 每个节点只有少数几个已探索的子节点
    //   - 节点统计信息（访问次数/胜率）是标量数据
    feature.matrix_density = compute_matrix_density(
        TaskCategory::SEARCH_SIMULATION, batch_size, false);

    // ---- 2. 数据规模 ----
    // 搜索仿真数据规模估算：
    //   每个仿真平均执行约200步走子
    //   每步需评估约5-10个候选着法
    //   每个候选着法有若干特征通道
    // 使用 batch_size * 1000 作为保守估算
    feature.data_scale = compute_data_scale(
        TaskCategory::SEARCH_SIMULATION, batch_size, 0);

    // ---- 3. 访存模式 ----
    // 混合访存：走子生成（规则）+ 树回溯（随机）+ 叶节点评估（顺序）
    feature.mem_pattern = compute_memory_pattern(
        TaskCategory::SEARCH_SIMULATION, false);

    // ---- 4. 并行度 ----
    // 搜索仿真并行度：
    //   根并行化策略下，不同仿真高度独立
    //   但在树内更新时需要轻量同步
    feature.parallelism_degree = compute_parallelism_degree(
        TaskCategory::SEARCH_SIMULATION, batch_size);

    return feature;
}

// ============================================================================
// 专用分析策略 - 通用计算
// ============================================================================

/**
 * @brief 通用计算特征分析
 *
 * 当任务无法归类到围棋专用类别时使用的通用分析策略。
 * 基于传入的 element_count 和 is_tensor_op 标志进行特征推断。
 *
 * 通用策略的保守性：
 *   - 矩阵密度使用中等值（张量->0.70，非张量->0.50）
 *   - 访存模式按张量标志判断（张量->顺序，否则->混合）
 *   - 并行度使用保守值
 *
 * @param batch_size    批次大小
 * @param element_count 元素数量
 */
TaskFeature WorkloadAnalyzer::analyze_generic_compute(size_t batch_size,
                                                       size_t element_count,
                                                       bool is_tensor_op)
{
    TaskFeature feature;

    // 确定是否为张量/矩阵运算
    // 在通用计算中，通过 is_tensor_op 标志区分张量运算和普通计算
    // 张量运算：高矩阵密度 + 顺序访存（GPU友好）
    // 普通计算：中矩阵密度 + 混合访存（保守估算）

    // ---- 1. 矩阵密度 ----
    // 使用传入的 is_tensor_op 标志进行精确判断
    feature.matrix_density = compute_matrix_density(
        TaskCategory::GENERIC_COMPUTE, batch_size, is_tensor_op);

    // ---- 2. 数据规模 ----
    // 直接使用 batch_size * element_count
    feature.data_scale = compute_data_scale(
        TaskCategory::GENERIC_COMPUTE, batch_size, element_count);

    // ---- 3. 访存模式 ----
    // 使用 is_tensor_op 标志：张量运算偏顺序访问，普通运算偏混合
    feature.mem_pattern = compute_memory_pattern(
        TaskCategory::GENERIC_COMPUTE, is_tensor_op);

    // ---- 4. 并行度 ----
    // 通用计算并行度保守估算
    // 如果是张量操作，并行度可适当提升（矩阵乘法天然并行）
    feature.parallelism_degree = compute_parallelism_degree(
        TaskCategory::GENERIC_COMPUTE, batch_size);

    return feature;
}

// ============================================================================
// 辅助函数 - GPU启动开销估算
// ============================================================================

/**
 * @brief 估算GPU内核启动开销（微秒级）
 *
 * GPU启动CUDA内核的开销来源：
 *   1. CPU端驱动调用开销（~2-5us）：CUDA Runtime API在CPU端的处理
 *   2. 内核参数准备和传输（~1-3us）：将内核参数打包并通过PCIe发送
 *   3. GPU端调度延迟（~5-10us）：GPU的Warp Scheduler接收并调度内核
 *   4. 线程块启动延迟（~5-20us）：SM逐个启动线程块的延迟
 *
 * 估算模型：
 *   基础开销：DEFAULT_GPU_LAUNCH_US = 20us
 *     - 覆盖CPU端驱动调用 + 参数传输 + GPU调度延迟
 *   批次相关开销：batch_size * 0.1us
 *     - 线程块越多，启动延迟越高
 *     - 0.1us/块是简化的估算（实际与SM数量、线程块大小相关）
 *   上限：MAX_GPU_LAUNCH_US = 100us
 *     - 驱动程序层面有优化，不会无限增长
 *
 * 该估算值用于在 calculate_device_affinity 中辅助判断
 * 小批次任务是否值得在GPU上执行。
 *
 * @param batch_size 批次大小（对应线程块数量的量级）
 * @return double GPU内核启动开销估算值（微秒）
 */
double WorkloadAnalyzer::estimate_gpu_launch_overhead(size_t batch_size) const
{
    if (batch_size == 0) {
        return DEFAULT_GPU_LAUNCH_US;
    }

    // 基础启动开销：20微秒
    // 涵盖CUDA Runtime的固定开销
    double overhead = DEFAULT_GPU_LAUNCH_US;

    // 批次相关开销：每个逻辑批次增加0.1微秒
    // 实际线程块启动的延迟累积
    // 注意：这里使用对数关系，因为GPU调度器能批量处理线程块启动
    // batch_size=16  -> 增加约 0.1*4=0.4us
    // batch_size=256 -> 增加约 0.1*8=0.8us
    // batch_size=4096 -> 增加约 0.1*12=1.2us
    overhead += std::log2(static_cast<double>(std::max(batch_size, size_t(1)))) * 0.1;

    // 限制上限为100微秒
    return std::min(overhead, MAX_GPU_LAUNCH_US);
}

}  // namespace heterogeneous
}  // namespace go
}  // namespace spacego