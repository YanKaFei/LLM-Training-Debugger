<div align="center">

<img src="../../assets/banner.svg" alt="LLM Training Debugger" width="100%">

<br>

[English](../../README.md) · **简体中文** · [한국어](README.ko.md) · [日本語](README.ja.md) · [Français](README.fr.md)

<br>

[![tests](https://img.shields.io/badge/tests-584_passing-3fb950?style=flat-square)](../../tests)
[![python](https://img.shields.io/badge/python-3.9%2B-58a6ff?style=flat-square)](../../pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-8b949e?style=flat-square)](../../LICENSE)
[![status](https://img.shields.io/badge/status-alpha-d29922?style=flat-square)](#项目状态)
[![deepseek-harness](https://img.shields.io/badge/built_with-DeepSeek_Harness-4d6bfe?style=flat-square)](https://github.com/topics/deepseek-harness)

</div>

---

## 这是什么

一个**以证据为先的 PyTorch 训练监控库**。它观察训练循环，把指标异常转化为事件（incident），
并且 —— 这一点最关键 —— **在证据不足以支撑结论时拒绝给出结论**。

<div align="center">
<img src="../../assets/architecture.svg" alt="架构" width="100%">
</div>

多数训练工具回答的是「发生了什么」。这个库问一个更窄也更难的问题：

> **我到底知不知道？如果不知道，我手上究竟有什么？**

## 这不是什么

先讲边界，因为诚实的边界比功能列表更有用：

- **不是实验追踪工具的替代品。** 它向追踪工具输出数据，但它本身不是仪表盘。
- **不是根因分析。** 它报告的是*某一步、某个指标*。它无法告诉你是哪个 kernel、哪个张量、哪一行代码。
- **不是控制器。** 它只观察。它不会裁剪梯度、不会重置优化器状态、不会中止你的训练。
- **未在大规模上验证。** 本仓库中的每一个数字都是在 4–6 层模型上测得的。
- **未经验证。** 没有盲测留出集。`validated_capabilities` 按设计为空。

## 为什么需要它

一次训练会产生一条 loss 曲线。当曲线出问题时，曲线不会告诉你问题在哪，而重跑往往复现不出来。

本库的立场是：**产品不是那条曲线，产品是事件** —— 一个**有边界、可审计、带时间戳**的陈述，
说明观察到了什么，附上支撑它的证据，并明确写出**无法得出什么结论**。

这个立场贯穿整个设计：

| 原则 | 在代码中的体现 |
|---|---|
| 观察绝不能破坏训练 | `emit()` 永不抛异常；P0–P3 有界队列宁可丢弃也不阻塞 |
| 机制必须被观察到，才能被断言 | `OBSERVED` / `INFERRED` / `HYPOTHESIS` / `UNKNOWN`；只有 `OBSERVED` 可提升 |
| 「没观察到」不等于「没发生」 | 每个判定都携带 `data_quality` 与 `completeness_warning` |
| 宁可弃权，也不编造关系 | `INSUFFICIENT_EVIDENCE` 是一等判定，不是错误 |
| 某些因果主张应当无法做出 | `CAUSED_BY` 与 `ROOT_CAUSE_OF` **在 schema 中不存在**，且有测试断言其永不出现 |

## 环境要求

| | |
|---|---|
| **Python** | 3.9 – 3.13，每个版本都在 CI 中实测。 |
| **核心依赖** | `numpy>=1.23`、`pyarrow>=12`、`duckdb>=0.10,<1.5` |
| **PyTorch 适配器** | `torch>=2.0` —— 可选，通过 `torch` extra 安装。observer、storage、detectors、episodes、fusion 与 CLI 均不导入 torch。 |
| **开发** | `.[dev]` = `pytest` **和** torch，因为测试套件会导入它。 |

> **`duckdb` 的上界是权宜之计，不是偏好。** duckdb 1.5.6 会让 `fetch_arrow_table()`
> 抛出 `SystemError: <built-in function __import__> returned a result with an exception set`。
> CI 中的 23 个失败全部发生在这一处调用；其余约 560 个测试通过，`import pyarrow` 也正常，
> 所以问题出在 duckdb 的 Arrow 桥接，而非 pyarrow。它在 Python 3.9 上不可见，因为
> duckdb 1.5.x 要求 **Python >= 3.10**，3.9 只会解析到 1.4.5 —— 这意味着**任何使用
> Python 3.10 及以上且未固定版本的人，装到的都是坏的**。
>
> 一个名为 *latest dependencies (informational)* 的 CI 任务刻意忽略所有上界并允许失败。
> **它红着，说明上界仍然必要；它变绿，就说明上界该撤掉。** 见
> [KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md) 第 15 条。

## 快速开始

```bash
git clone https://github.com/YanKaFei/LLM-Training-Debugger.git
cd LLM-Training-Debugger
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

一个完整可运行的示例：

```bash
python examples/quickstart.py
```

或者接入你现有的训练循环：

```python
from training_debugger.adapters.pytorch import PyTorchTrainingAdapter

adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="my-model")
adapter.run_start(steps=total_steps)

for step in range(1, total_steps + 1):
    with adapter.step(step) as tracker:
        loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        loss.backward()
        tracker.record_gradients()
        with tracker.phase("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tracker.record_parameters()

adapter.run_end()
```

## 你会得到什么

<div align="center">
<img src="../../assets/verdicts.svg" alt="判定" width="100%">
</div>

每个事件都携带 `reason_code`、`evidence_state`、`promotion_disposition`，以及在数据流不完整时的
`data_quality` 警告。**一个发生过事件丢弃的 run，永远不会被渲染成健康的 run。**

## 命令

<div align="center">
<img src="../../assets/cli-demo.svg" alt="命令行演示" width="100%">
</div>

```bash
training-debugger --root <store> runs                  # 列出所有 run
training-debugger --root <store> verdicts  <run>       # 所有判定，含弃权
training-debugger --root <store> incidents <run>       # 带证据的事件列表
training-debugger --root <store> incident  <run> <id>  # 单个事件的完整内容
training-debugger --root <store> metrics   <run> [m]   # 查询指标行
training-debugger --root <store> diagnose  <run>       # 确定性诊断
training-debugger --root <store> compare   <a> <b>     # 两个 run 从哪一步开始分叉
training-debugger --root <store> scan      <r1> ...    # 批次扫描：哪个 run 是离群者
```

### `compare` —— 定位分叉

对齐两个 run 的指标序列，以**它们早期一致的区域作为零假设**，报告第一个越出该区域并保持住的步。
共享噪声在差分中抵消，因此单 run 阈值做不到的定位，这里能做到。

在注入步已知的成对语料上实测：**9/9 故障对精确定位到注入步，3/3 干净对正确弃权。**

wall-clock 与内存类指标默认排除 —— *计时差异不是算法发散*。

### `scan` —— 在批次中找离群者

所有配对比较一次；一个 run 的得分是它与多少同伴发生了分叉。**它要求 like-for-like 的 run**：
固定 seed 与配置，只变一个因素。混 seed 的批次会返回 `NO_SINGLE_OUTLIER` 并解释原因，
而不是随便点一个 run 的名字。

## 能力

每一项能力都带有明确的证据等级。这里没有任何一项被标记为已验证。

| 能力 | 状态 | 证据 |
|---|---|---|
| 非有限值观测（NaN / Inf） | `EXPERIMENTAL` | 精确；无需阈值 |
| 梯度异常检测 | `EXPERIMENTAL` | 检测边界 ×20 与 ×40 → 6/6；×2 与 ×5 → 0/6 |
| 带弃权的事件融合 | `EXPERIMENTAL` | run 级 precision 0.947 / recall 0.500（42-run 语料） |
| 双 run 分叉定位 | `EXPERIMENTAL` | 9/9 精确，3/3 弃权（成对语料） |
| 批次离群扫描 | `EXPERIMENTAL` | like-for-like 批次上 5/5 同伴支持 |
| 逐模块激活健康态 | `EXPERIMENTAL` | 死模块被点名并定位到步，OBSERVED 1.0 |
| 优化器调用见证 | `EXPERIMENTAL` | 已冻结；默认关闭 |
| 关系推断（仅顺序） | `UNSUPPORTED` | 默认关闭；一次实测尝试编造了 16 条关联 |

## 已知限制

公开这些，是因为**一个失败未被记录的监控器不值得信任**：

- **run 级 recall 为 0.500。** 开发语料中的主要故障有一半被漏掉。
- **健康 run 误报率 7.7%**，而自设门限为 5%。**该门限不达标。**
- **观测器开销未达到自身 <3% 的目标**（4 层基准负载）。其成本约为每步*固定*（0.2–0.9 ms），
  因此占比随负载增大而崩塌：+11.0%（4 层）→ +5.2%（8 层）→ +0.4%（12 层）。
  步长超过约 17 ms 后即进入预算内。**这里如实报告为失败，不作辩解。**
- 没有盲测留出集；所有数字均来自开发语料。
- 仅在 4–6 层模型上验证过。

完整列表：[KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md) · 设计记录：
[DECISIONS.md](../DECISIONS.md) · 当前状态：[DEVELOPMENT_STATE.md](../DEVELOPMENT_STATE.md)

## 复现这些数字

```bash
pytest -q                                            # 584 通过，1 跳过
python tests/experiments/full_pipeline_smoke.py      # 8/8
python examples/quickstart.py
python tests/performance/audit_observer_cost.py      # 配对开销测量，含置信区间
```

本 README 中每一个关键数字都由图 `tests/experiments/` 或 `tests/performance/` 中的脚本产出，
原始输出已提交在 `outputs/` 下。

## 项目状态

**Alpha。** 接口尚不稳定，API 可能随时变更。它已经可以真正用于给训练插桩并读取可审计的判定，
但还不适合在其上构建生产流水线。

## 反馈与交流

**GitHub 是本项目唯一的讨论场所。** 没有聊天群、没有邮件列表、没有其他渠道 ——
因此在这里提出的内容所有人都能看到，并且可被检索。

| | |
|---|---|
| **发现 Bug？** | [提交 issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **有疑问，或想讨论某个设计决策？** | [发起 discussion](https://github.com/YanKaFei/LLM-Training-Debugger/discussions) |
| **想要新功能？** | [提交 issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **不同意这里的某个说法？** | [提交 issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) —— 请务必提出 |

**欢迎反驳，尤其是针对数字的反驳。** 如果你复现本 README 中的某个测量却得到不同结果，
那是一份值得提交的 bug 报告。如果这里的某个说法不被本仓库中的证据支持，那么**该说法就是错的**，
应当被更正或撤回。

提交时请附上：你运行的命令、模型形状与步数、PyTorch 版本，以及完整输出。

## 参与贡献

见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。简短版本：**让主张变弱的改动与让主张变强的改动同样受欢迎**，
并且**绝不允许为了让测试通过而削弱测试**。

## 许可证

[Apache-2.0](../../LICENSE)。

<div align="center">
<br>
<sub>使用 <a href="https://github.com/topics/deepseek-harness">DeepSeek Harness</a> 构建。</sub>
</div>
