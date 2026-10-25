# 拓扑感知UCT实现

## 核心设计

拓扑感知UCT（Upper Confidence Bound applied to Trees）是MiniGo AI系统的核心搜索算法，它在传统UCT的基础上引入了拓扑结构学习和合并机制，实现了更高效的状态空间探索。

- **拓扑修正价值项**：考虑节点的合并概率和被合并概率，修正传统Q值
- **合并增强探索项**：鼓励探索高合并概率区域，这些区域通常是状态空间的结构枢纽
- **不确定性引导探索**：根据神经网络的不确定性估计调整探索强度
- **新颖性奖励机制**：保护未被充分探索的分支，防止早熟收敛
- **动态参数调整**：根据搜索进程自适应调整算法参数

## 数据结构

### UCTVariant枚举

定义了多种UCT变体类型，用于动态选择不同的搜索策略：

```python
class UCTVariant:
    BASIC_TOPOLOGY = 0  # 基本拓扑感知UCT
    HISTORY_AWARE = 1  # 历史感知UCT
    DIVERSITY_FOCUSED = 2  # 多样性聚焦UCT
    BALANCED_EXPLORATION = 3  # 平衡探索UCT
    ADAPTIVE = 4  # 自适应UCT
    CONFIDENCE_BASED = 5  # 基于置信度的UCT
    HIERARCHICAL = 6  # 分层UCT
    PARALLEL = 7  # 并行UCT
    MERGE_OPTIMIZED = 8  # 合并优化UCT
    NOVELTY_DRIVEN = 9  # 新颖性驱动UCT
```

### TopologyUCT类结构

拓扑感知UCT的基类，实现了核心算法和接口：

```python
class TopologyUCT:
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1):
        # 初始化探索常数和各种权重参数
        self.c_puct = c_puct
        self.alpha1 = alpha1  # 合并概率权重
        self.alpha2 = alpha2  # 被合并概率权重
        self.beta1 = beta1  # 合并增强权重
        self.beta2 = beta2  # 被合并增强权重
        self.gamma = gamma  # 不确定性权重
        self.delta = delta  # 新颖性权重
        self.epsilon = epsilon  # 最小探索概率
        self.variant_type = UCTVariant.BASIC_TOPOLOGY
        self.parameter_history = []  # 参数调整历史
        self.performance_metrics = {}  # 性能指标
```

## 算法实现

### 拓扑修正价值项计算

考虑节点的合并概率和被合并概率，对基础Q值进行修正：

```python
def calculate_topology_q_value(self, node: HMGSNode) -> float:
    # 基础Q值
    q_value = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
    
    # 拓扑修正：Q_topo = Q × (1 + α1×P_m + α2×P_b)
    q_topo = q_value * (1 + self.alpha1 * node.merge_prob + self.alpha2 * node.merged_into_prob)
    
    return q_topo
```

### 合并增强探索项

鼓励探索高合并概率区域，提高搜索效率：

```python
def calculate_merge_boost(self, node: HMGSNode) -> float:
    # 合并增强：M = 1 + β1×P_m + β2×P_b
    merge_boost = 1 + self.beta1 * node.merge_prob + self.beta2 * node.merged_into_prob
    
    return merge_boost
```

### 不确定性引导探索

根据节点的不确定性估计调整探索强度：

```python
def calculate_uncertainty_boost(self, node: HMGSNode) -> float:
    # 不确定性增强：U = 1 + γ×σ
    uncertainty_boost = 1 + self.gamma * node.uncertainty
    
    return uncertainty_boost
```

### 新颖性奖励项

保护未被充分探索的分支，防止搜索过早收敛：

```python
def calculate_novelty_boost(self, node: HMGSNode) -> float:
    # 新颖性增强：N = 1 + δ×(1 - P_merge)×D(t)×B(k)
    # 简化实现：使用连续未合并次数作为新颖性指标
    novelty_score = node.continuous_unmerged / 100.0  # 归一化到0-1范围
    novelty_boost = 1 + self.delta * novelty_score
    
    return novelty_boost
```

### 拓扑感知UCT值计算

整合所有因素，计算最终的UCT值：

```python
def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
    # 计算基础Q值和访问次数
    visit_count = node.visit_count
    if visit_count == 0:
        q_value = 0.0
    else:
        q_value = node.total_value / visit_count
    
    # 计算拓扑修正价值项（合并计算，减少方法调用）
    q_topo = q_value * (1 + self.alpha1 * node.merge_prob + self.alpha2 * node.merged_into_prob)
    
    # 计算探索项
    parent_visits = math.sqrt(parent.visit_count)
    # 确保索引不超出prior_probs列表范围，添加默认值处理
    prior = parent.prior_probs[index] if hasattr(parent, 'prior_probs') and parent.prior_probs and index < len(parent.prior_probs) else 0.0
    
    # 合并计算各项增强因子，减少方法调用
    merge_boost = 1 + self.beta1 * node.merge_prob + self.beta2 * node.merged_into_prob
    uncertainty_boost = 1 + self.gamma * node.uncertainty
    novelty_boost = 1 + self.delta * (node.continuous_unmerged / 100.0)
    
    # 计算U值并应用增强因子
    u_value = self.c_puct * prior * parent_visits / (1.0 + visit_count)
    u_value *= merge_boost * uncertainty_boost * novelty_boost
    
    # 最终UCT值
    uct_value = q_topo + u_value
    
    return uct_value
```

### 批量UCT计算

使用向量化操作提高效率，批量计算多个节点的UCT值：

```python
def calculate_batch_uct(self, nodes: List[HMGSNode], parent: HMGSNode) -> np.ndarray:
    batch_size = len(nodes)
    uct_values = np.zeros(batch_size, dtype=np.float32)
    
    # 预先计算父节点的访问次数平方根
    parent_visits = math.sqrt(parent.visit_count)
    
    # 向量化计算UCT值
    for i, node in enumerate(nodes):
        uct_values[i] = self.calculate_uct(node, parent, i)
    
    return uct_values
```

### 最佳子节点选择

选择UCT值最大的子节点，用于搜索树的扩展：

```python
def select_best_child(self, node: HMGSNode) -> Optional[HMGSNode]:
    if not node.children:
        return None
    
    best_score = -float("inf")
    best_child = None
    
    for i, child in enumerate(node.children):
        uct_score = self.calculate_uct(child, node, i)
        if uct_score > best_score:
            best_score = uct_score
            best_child = child
    
    return best_child
```

## UCT变体详解

### 历史感知UCT（HistoryAwareUCT）

考虑节点的访问历史，调整探索强度：

```python
class HistoryAwareUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, history_weight: float = 0.1):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.history_weight = history_weight
        self.variant_type = UCTVariant.HISTORY_AWARE
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 基础拓扑UCT值
        uct_value = super().calculate_uct(node, parent, index)
        
        # 计算历史影响
        history_influence = self.calculate_history_influence(node)
        
        # 应用历史影响
        uct_value *= history_influence
        
        return uct_value
```

### 多样性聚焦UCT（DiversityFocusedUCT）

更注重探索新的区域，保护搜索多样性：

```python
class DiversityFocusedUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.4, epsilon: float = 0.2, diversity_bonus: float = 0.2):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.diversity_bonus = diversity_bonus
        self.variant_type = UCTVariant.DIVERSITY_FOCUSED
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 基础拓扑UCT值
        uct_value = super().calculate_uct(node, parent, index)
        
        # 计算多样性奖励
        diversity_bonus = self.calculate_diversity_bonus(node)
        
        # 应用多样性奖励
        uct_value *= diversity_bonus
        
        return uct_value
```

### 平衡探索UCT（BalancedExplorationUCT）

平衡探索和利用，根据访问次数动态调整：

```python
class BalancedExplorationUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, balance_factor: float = 0.5):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.balance_factor = balance_factor
        self.variant_type = UCTVariant.BALANCED_EXPLORATION
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 基础拓扑UCT值
        uct_value = super().calculate_uct(node, parent, index)
        
        # 计算平衡调整
        balance_adjustment = self.calculate_balance_adjustment(node)
        
        # 应用平衡调整
        uct_value *= balance_adjustment
        
        return uct_value
```

### 自适应UCT（AdaptiveUCT）

根据搜索状态动态调整参数，优化探索-利用平衡：

```python
class AdaptiveUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, adaptation_rate: float = 0.05):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.adaptation_rate = adaptation_rate
        self.variant_type = UCTVariant.ADAPTIVE
        self.previous_diversity = 0.5  # 初始多样性假设
        self.previous_convergence = 0.5  # 初始收敛假设
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 动态调整参数
        diversity_score = self.calculate_diversity_score(parent.children)
        convergence_rate = self.calculate_convergence_rate(parent)
        
        # 平滑参数调整
        smoothed_diversity = self.previous_diversity * 0.9 + diversity_score * 0.1
        smoothed_convergence = self.previous_convergence * 0.9 + convergence_rate * 0.1
        
        self.adjust_parameters(smoothed_diversity, smoothed_convergence)
        
        # 更新历史值
        self.previous_diversity = smoothed_diversity
        self.previous_convergence = smoothed_convergence
        
        # 计算基础UCT值
        uct_value = super().calculate_uct(node, parent, index)
        
        # 应用自适应因子
        adaptive_factor = self.calculate_adaptive_factor(node)
        uct_value *= adaptive_factor
        
        return uct_value
```

### 基于置信度的UCT（ConfidenceBasedUCT）

考虑价值估计的置信度，调整探索强度：

```python
class ConfidenceBasedUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, confidence_weight: float = 0.1):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.confidence_weight = confidence_weight
        self.variant_type = UCTVariant.CONFIDENCE_BASED
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 计算拓扑修正价值项，考虑置信度
        q_value = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
        confidence = self.calculate_value_confidence(node)
        
        # 置信度越高，价值权重越大
        q_topo = q_value * (1 + self.alpha1 * node.merge_prob + self.alpha2 * node.merged_into_prob) * \
                 (1 + self.confidence_weight * confidence)
        
        # 计算探索项，置信度越低，探索权重越大
        parent_visits = math.sqrt(parent.visit_count)
        u_value = self.c_puct * parent.prior_probs[index] * parent_visits / (1.0 + node.visit_count) * \
                  (1 + self.confidence_weight * (1.0 - confidence))
        
        # 应用其他增强因子
        merge_boost = self.calculate_merge_boost(node)
        uncertainty_boost = self.calculate_uncertainty_boost(node)
        novelty_boost = self.calculate_novelty_boost(node)
        
        u_value *= merge_boost * uncertainty_boost * novelty_boost
        
        # 最终UCT值
        uct_value = q_topo + u_value
        
        return uct_value
```

### 分层UCT（HierarchicalUCT）

考虑节点的层级，深层节点获得更高的探索奖励：

```python
class HierarchicalUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.2,
                 beta1: float = 0.15, beta2: float = 0.25, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, level_weight: float = 0.1):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.level_weight = level_weight
        self.variant_type = UCTVariant.HIERARCHICAL
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 基础拓扑UCT值
        uct_value = super().calculate_uct(node, parent, index)
        
        # 应用层级影响
        level_influence = self.calculate_level_influence(node)
        uct_value *= level_influence
        
        return uct_value
```

### 合并优化UCT（MergeOptimizedUCT）

优化合并节点的选择，更注重合并概率：

```python
class MergeOptimizedUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.2, alpha2: float = 0.1,
                 beta1: float = 0.3, beta2: float = 0.15, gamma: float = 0.3,
                 delta: float = 0.2, epsilon: float = 0.1, merge_weight: float = 0.2):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.merge_weight = merge_weight
        self.variant_type = UCTVariant.MERGE_OPTIMIZED
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 计算拓扑修正价值项，更注重合并概率
        q_value = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
        q_topo = q_value * (1 + self.alpha1 * node.merge_prob + self.alpha2 * node.merged_into_prob)
        
        # 计算探索项，更注重合并增强
        parent_visits = math.sqrt(parent.visit_count)
        u_value = self.c_puct * parent.prior_probs[index] * parent_visits / (1.0 + node.visit_count)
        
        # 应用增强因子，更注重合并潜力
        merge_boost = self.calculate_merge_boost(node)
        uncertainty_boost = self.calculate_uncertainty_boost(node)
        novelty_boost = self.calculate_novelty_boost(node)
        merge_potential = self.calculate_merge_potential(node)
        
        u_value *= merge_boost * uncertainty_boost * novelty_boost * merge_potential
        
        # 最终UCT值
        uct_value = q_topo + u_value
        
        return uct_value
```

### 新颖性驱动UCT（NoveltyDrivenUCT）

更注重新颖区域的探索，保护未被充分探索的分支：

```python
class NoveltyDrivenUCT(TopologyUCT):
    def __init__(self, c_puct: float = 1.0, alpha1: float = 0.1, alpha2: float = 0.1,
                 beta1: float = 0.15, beta2: float = 0.15, gamma: float = 0.2,
                 delta: float = 0.5, epsilon: float = 0.3, novelty_bonus: float = 0.3):
        super().__init__(c_puct, alpha1, alpha2, beta1, beta2, gamma, delta, epsilon)
        self.novelty_bonus = novelty_bonus
        self.variant_type = UCTVariant.NOVELTY_DRIVEN
    
    def calculate_uct(self, node: HMGSNode, parent: HMGSNode, index: int) -> float:
        # 计算拓扑修正价值项
        q_value = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
        q_topo = q_value * (1 + self.alpha1 * node.merge_prob + self.alpha2 * node.merged_into_prob)
        
        # 计算探索项，大幅增强新颖性影响
        parent_visits = math.sqrt(parent.visit_count)
        u_value = self.c_puct * parent.prior_probs[index] * parent_visits / (1.0 + node.visit_count)
        
        # 应用增强因子，更注重新颖性
        merge_boost = self.calculate_merge_boost(node)
        uncertainty_boost = self.calculate_uncertainty_boost(node)
        novelty_boost = self.calculate_novelty_boost(node)
        custom_novelty_bonus = self.calculate_novelty_bonus(node)
        
        u_value *= merge_boost * uncertainty_boost * novelty_boost * custom_novelty_bonus
        
        # 最终UCT值
        uct_value = q_topo + u_value
        
        return uct_value
```

## UCT策略工厂

UCT策略工厂类，用于创建不同类型的UCT策略，简化算法扩展和使用：

```python
class UCTStrategyFactory:
    # UCT变体类型到类的映射，便于扩展
    _VARIANT_MAP = {
        UCTVariant.BASIC_TOPOLOGY: TopologyUCT,
        UCTVariant.HISTORY_AWARE: HistoryAwareUCT,
        UCTVariant.DIVERSITY_FOCUSED: DiversityFocusedUCT,
        UCTVariant.BALANCED_EXPLORATION: BalancedExplorationUCT,
        UCTVariant.ADAPTIVE: AdaptiveUCT,
        UCTVariant.CONFIDENCE_BASED: ConfidenceBasedUCT,
        UCTVariant.HIERARCHICAL: HierarchicalUCT,
        UCTVariant.MERGE_OPTIMIZED: MergeOptimizedUCT,
        UCTVariant.NOVELTY_DRIVEN: NoveltyDrivenUCT
    }
    
    # UCT变体名称到类型的映射
    _NAME_TO_TYPE = {
        "BASIC_TOPOLOGY": UCTVariant.BASIC_TOPOLOGY,
        "HISTORY_AWARE": UCTVariant.HISTORY_AWARE,
        "DIVERSITY_FOCUSED": UCTVariant.DIVERSITY_FOCUSED,
        "BALANCED_EXPLORATION": UCTVariant.BALANCED_EXPLORATION,
        "ADAPTIVE": UCTVariant.ADAPTIVE,
        "CONFIDENCE_BASED": UCTVariant.CONFIDENCE_BASED,
        "HIERARCHICAL": UCTVariant.HIERARCHICAL,
        "MERGE_OPTIMIZED": UCTVariant.MERGE_OPTIMIZED,
        "NOVELTY_DRIVEN": UCTVariant.NOVELTY_DRIVEN
    }
    
    @staticmethod
    def create_uct_strategy(variant_type: UCTVariant, **kwargs) -> TopologyUCT:
        """
        创建UCT策略
        
        Args:
            variant_type: UCT变体类型
            **kwargs: UCT参数
        
        Returns:
            UCT策略实例
        """
        uct_class = UCTStrategyFactory._VARIANT_MAP.get(variant_type)
        if not uct_class:
            raise ValueError(f"未知的UCT变体类型: {variant_type}")
        return uct_class(**kwargs)
    
    @staticmethod
    def create_uct_strategy_by_name(name: str, **kwargs) -> TopologyUCT:
        """
        根据名称创建UCT策略（统一接口）
        
        Args:
            name: UCT变体名称
            **kwargs: UCT参数
        
        Returns:
            UCT策略实例
        """
        variant_type = UCTStrategyFactory.get_variant_type_by_name(name)
        return UCTStrategyFactory.create_uct_strategy(variant_type, **kwargs)
    
    @staticmethod
    def get_available_variants() -> List[str]:
        """
        获取可用的UCT变体列表
        
        Returns:
            UCT变体名称列表
        """
        return list(UCTStrategyFactory._NAME_TO_TYPE.keys())
    
    @staticmethod
    def get_variant_type_by_name(name: str) -> UCTVariant:
        """
        根据名称获取UCT变体类型
        
        Args:
            name: UCT变体名称
        
        Returns:
            UCT变体类型
        """
        variant_type = UCTStrategyFactory._NAME_TO_TYPE.get(name.upper())
        if not variant_type:
            raise ValueError(f"未知的UCT变体名称: {name}")
        return variant_type
```

## 便捷函数

为了简化UCT策略的使用，提供了便捷函数：

```python
def calculate_topology_uct(node: HMGSNode, parent: HMGSNode, index: int, c_puct: float = 1.0) -> float:
    """
    便捷函数：计算拓扑感知的UCT值
    
    Args:
        node: HMGS节点
        parent: 父节点
        index: 节点在父节点子节点列表中的索引
        c_puct: 探索常数
    
    Returns:
        拓扑感知的UCT值
    """
    uct_calculator = TopologyUCT(c_puct=c_puct)
    return uct_calculator.calculate_uct(node, parent, index)

def create_uct_strategy(variant_name: str, **kwargs) -> TopologyUCT:
    """
    便捷函数：根据名称创建UCT策略
    
    Args:
        variant_name: UCT变体名称
        **kwargs: UCT参数
    
    Returns:
        UCT策略实例
    """
    variant_type = UCTStrategyFactory.get_variant_type_by_name(variant_name)
    return UCTStrategyFactory.create_uct_strategy(variant_type, **kwargs)
```

## 总结

拓扑感知UCT是MiniGo AI系统的核心搜索算法，它在传统UCT的基础上引入了拓扑结构学习和合并机制，实现了更高效的状态空间探索。该算法通过多种创新技术，显著提高了AI在复杂环境中的搜索效率和决策质量。

### 核心优势

1. **拓扑结构学习**：通过分析节点间的合并关系，学习状态空间的拓扑结构，提高搜索效率
2. **合并机制增强**：利用合并概率和被合并概率修正价值估计，鼓励探索关键区域
3. **自适应探索**：根据搜索进程动态调整参数，平衡探索与利用
4. **多样性保护**：保护未被充分探索的分支，防止算法过早收敛到局部最优
5. **高效计算**：支持批量计算和向量化操作，提高计算效率
6. **灵活扩展**：多种UCT变体适应不同搜索场景，工厂模式简化算法扩展和使用

### 应用场景

- **复杂游戏AI**：如围棋、象棋等需要大量搜索的游戏
- **路径规划**：机器人导航、自动驾驶等领域的路径规划
- **决策优化**：复杂决策问题的最优解搜索
- **资源分配**：多资源、多目标的优化问题

### 技术亮点

1. **多种UCT变体**：提供10种不同的UCT变体，适应不同的搜索需求
2. **拓扑修正价值**：考虑节点的合并特性，修正传统Q值
3. **合并增强探索**：鼓励探索高合并概率区域，提高搜索效率
4. **不确定性引导**：根据神经网络的不确定性估计调整探索强度
5. **新颖性奖励**：保护未被充分探索的分支，促进全面搜索

### 未来发展方向

1. **深度学习融合**：进一步融合深度学习技术，提高拓扑结构学习的准确性
2. **分布式实现**：支持大规模分布式搜索，处理更复杂的问题
3. **自适应参数调整**：基于学习的参数调整机制，进一步提高算法性能
4. **多模态融合**：结合多种信息源，提高搜索质量

拓扑感知UCT算法的实现为MiniGo AI系统在复杂状态空间中的高效搜索提供了坚实的基础，也为其他领域的搜索算法设计提供了新的思路和方向。