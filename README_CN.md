# RSBench-V3: 多目标推荐系统基准测试库

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/status-active-green.svg)](https://github.com/your-repo/rsbench-v3)

## 🎯 概述

RSBench-V2 是一个基于大语言模型（LLM）和进化算法的多目标推荐系统优化综合库。它为评估推荐系统在多个目标（包括准确性、多样性和公平性）上的表现提供了标准化基准。

> **⚠️ 注意：这是一个新版本** - 相较于原文章，本实现采用了**异步评估**机制，显著提升了执行效率。所有 LLM 调用通过 asyncio 并行执行，使得评估过程比原文章中的顺序执行方式更加高效。

## 🚀 为什么 RSBench-V3 是一个库

### 1. **模块化架构**
- **核心组件**: 算法、问题、数据集和工具的独立模块
- **可扩展设计**: 易于添加新算法、问题或数据集
- **插件系统**: LLM 算子可以交换和自定义

### 2. **标准化接口**
- **统一 API**: 所有组件的一致接口
- **类型提示**: 完整的类型注解，提供更好的 IDE 支持
- **文档**: 全面的文档字符串和示例

### 3. **生产就绪特性**
- **异步支持**: 高性能异步评估
- **错误处理**: 强大的错误处理和恢复机制
- **日志记录**: 全面的日志记录和进度跟踪
- **配置**: 灵活的配置系统

### 4. **研究开发工具**
- **可视化**: 内置绘图和分析工具
- **指标**: 全面的性能指标
- **基准测试**: 标准化评估协议
- **可重现性**: 通过种子控制实现确定性结果

### 5. **多目标优化**
- **NSGA-II**: 最先进的多目标进化算法
- **帕累托前沿**: 自动帕累托前沿识别
- **多样性指标**: 拥挤距离和分布指标
- **收敛分析**: 超体积和收敛跟踪

## 🏗️ 架构

```
RSBench-V3/
├── core/                    # 核心库组件
│   ├── algorithms.py       # 进化算法 (NSGA-II, MOEA/D, IBEA)
│   ├── problems.py         # 多目标问题 (AccDiv, AccFair, AccDivFair)
│   └── operators.py        # 基于 LLM 的遗传算子
├── datasets/               # 数据集处理
│   ├── loader.py          # 数据集加载器 (MovieLens, Game, Bundle)
│   └── __init__.py        # 数据集工厂函数
├── utils/                  # 工具模块
│   ├── plotting.py        # 可视化和绘图
│   ├── metrics.py         # 性能指标
│   ├── selection.py       # 选择算子
│   └── dominance.py       # 帕累托支配工具
├── examples/               # 示例脚本
│   ├── basic_example.py   # 基本使用示例
│   └── nsga2_with_plotting.py  # 带绘图的高级示例
└── Results/               # 生成的结果和图表
```

## 🎯 主要特性

### **多目标问题**
- **准确性与多样性**: 平衡推荐质量与物品多样性
- **准确性与公平性**: 确保跨群体的公平推荐
- **准确性与多样性与公平性**: 全面的三目标优化

### **进化算法**
- **NSGA-II**: 非支配排序遗传算法 II
- **MOEA/D**: 基于分解的多目标进化算法
- **IBEA**: 基于指标的进化算法

### **LLM 集成**
- **GPT 模型**: OpenAI GPT-3.5/4 集成
- **GLM 模型**: 智谱AI GLM 集成
- **自定义模型**: 可扩展支持其他 LLM 提供商
- **令牌跟踪**: 成本监控和优化

### **高性能评估**
- **异步处理**: 种群并行评估 - **所有提示和所有样本并行运行**
- **真正的异步**: 使用 asyncio 实现最大并行度以获得最佳性能
- **批处理**: 高效的批量评估
- **速率限制**: API 速率限制管理
- **错误恢复**: 强大的错误处理和回退机制
- **性能提升**: 相比原文章中的顺序评估方式，执行效率显著提升

### **可视化与分析**
- **进化图**: 目标空间进化可视化
- **帕累托前沿**: 非支配解高亮显示
- **指标跟踪**: 跨世代的性能指标
- **导出选项**: 多种输出格式 (PNG, TXT, NPY)

## 📦 安装

### 先决条件
- Python 3.8+
- OpenAI API 密钥（用于 GPT 模型）
- 智谱AI API 密钥（用于 GLM 模型，可选）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 环境设置
```bash
# 创建 .env 文件并添加您的 API 密钥
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ZHIPUAI_API_KEY=your_zhipuai_key_here" >> .env  # 可选
```

## 🚀 快速开始

### 基本使用
```python
import asyncio
from core.algorithms import NSGA2LLM
from core.problems import AccDivProblem
from datasets import create_dataset_loader

async def main():
    # 加载数据集
    dataset_loader = create_dataset_loader("MovieLens", "Dataset/Movie")
    train_data = dataset_loader.get_train_data()
    
    # 创建问题
    problem = AccDivProblem(
        train_data=train_data,
        batch_size=5,
        llm_model='gpt',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # 创建算法
    algorithm = NSGA2LLM(
        problem=problem,
        pop_size=10,
        max_iter=3,
        api_key=os.getenv('OPENAI_API_KEY'),
        llm_model='gpt'
    )
    
    # 运行优化
    final_population, final_objectives = await algorithm.run()
    
    print(f"优化完成！")
    print(f"最终种群大小: {len(final_population)}")
    print(f"目标形状: {final_objectives.shape}")

# 运行示例
asyncio.run(main())
```

### 带绘图功能
```python
from core.algorithms_with_plotting import NSGA2LLM

# 启用绘图
algorithm = NSGA2LLM(
    problem=problem,
    pop_size=10,
    max_iter=3,
    enable_plotting=True  # 启用可视化
)

# 运行优化并自动生成图表
final_population, final_objectives = await algorithm.run()
```

## 📊 示例输出

```
🚀 NSGA2-LLM 算法启动！
初始化种群...
🔄 评估初始种群...
🚀 真正的异步评估
⚡ 所有提示和所有样本通过 asyncio 并行运行！
🔧 处理 10 个提示，每个 5 个样本
🚀 创建 50 个异步 LLM 任务...
⚡ 并行执行 50 个 LLM 调用...
📊 评估摘要: 10 个提示，50 个样本，总计 19.09 秒
✅ 初始评估完成，用时: 0.59 分钟

> **性能说明**: 上述异步评估展示了相比顺序执行的效率提升。所有 LLM 调用并发执行，相比原文章中的方法，总执行时间显著减少。

进化开始！
🔄 第 1/3 代
⚡ 交叉完成，用时: 17.54 秒
🔄 评估第 1 代后代...
📊 评估摘要: 10 个提示，50 个样本，总计 20.22 秒
✅ 第 1 代评估完成，用时: 20.22 秒
⚡ 环境选择完成，用时: 0.00 秒
✅ 完成第 1 次迭代

🎯 进化完成！
📊 生成进化图表...
📊 图表已保存: Results/NSGA2-LLM_Evolution_20250913_205950.png
📊 指标图表已保存: Results/NSGA2-LLM_Metrics_20250913_205950.png
📋 摘要报告已保存: Results/NSGA2-LLM_Summary_20250913_205950.txt

📊 算法摘要:
算法: NSGA2-LLM
种群大小: 10
迭代次数: 3
最终种群大小: 10
目标数量: 2

=== 令牌使用摘要 ===
总令牌数: 379724
提示令牌: 237084
完成令牌: 142640
预估成本: $0.3325
```

## 🔧 配置

### 算法参数
```python
algorithm = NSGA2LLM(
    problem=problem,
    pop_size=20,           # 种群大小
    max_iter=10,           # 最大迭代次数
    api_key=api_key,
    llm_model='gpt',       # 'gpt' 或 'glm'
    enable_plotting=True   # 启用可视化
)
```

### 问题参数
```python
problem = AccDivProblem(
    train_data=train_data,
    batch_size=10,         # 评估批次大小
    llm_model='gpt',
    api_key=api_key
)
```

## 📈 性能指标

### 多目标指标
- **超体积**: 解支配的目标空间体积
- **分布**: 目标空间中解的多样性
- **收敛性**: 到参考帕累托前沿的距离
- **帕累托前沿大小**: 非支配解的数量

### 效率指标
- **评估时间**: 每代的时间
- **令牌使用**: LLM API 使用和成本
- **内存使用**: 内存消耗跟踪
- **并行效率**: 异步处理性能

## 🎨 可视化

库自动生成：
- **进化图**: 跨世代的目标空间进化
- **帕累托前沿**: 非支配解可视化
- **指标图**: 随时间变化的性能指标
- **摘要报告**: 全面的算法分析

## 🔬 研究应用

### 学术研究
- **多目标优化**: MOO 算法的基准测试
- **推荐系统**: 推荐策略的评估
- **LLM 集成**: 基于 LLM 的优化研究
- **AI 公平性**: 偏见和公平性分析

### 工业应用
- **推荐系统**: 生产推荐优化
- **内容策展**: 多目标内容选择
- **个性化**: 平衡的个性化策略
- **A/B 测试**: 自动化优化测试

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发设置
```bash
git clone https://github.com/your-repo/rsbench-v3.git
cd rsbench-v3
pip install -e .
pip install -r requirements-dev.txt
```

### 运行测试
```bash
python -m pytest tests/
```

## 📚 文档

- [API 参考](docs/api.md)
- [示例](examples/)
- [教程](docs/tutorials.md)
- [性能指南](docs/performance.md)

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **NSGA-II**: Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- **多目标优化**: Coello, C. A. C. "Evolutionary multi-objective optimization"
- **推荐系统**: Ricci, F., et al. "Recommender Systems Handbook"
