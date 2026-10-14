# Task 1：MicroDuck 篮球平衡强化学习

## 1. 任务与系统结构

MicroDuck 在自由滚动的篮球上保持直立，并跟踪平面速度和偏航角速度指令。控制频率为50 Hz，即每20 ms计算一次动作；控制频率与视频帧率不同。

```text
机器人状态 → 观测构造 → LSTM Actor → 关节动作 → 执行器与物理仿真 → 新状态
```

Actor 不直接读取球的位置和速度，因此称为 blind actor。训练时 Critic 可以额外读取球状态等特权信息，构成非对称 Actor–Critic；部署只需要 Actor。

## 2. 61维观测与14维动作

Actor 观测形状为 `obs[1,61]`，索引采用Python左闭右开区间。

| 索引 | 内容 | 维度 |
|---|---|---:|
| `0:3` | 身体角速度 | 3 |
| `3:6` | 身体坐标系中的重力投影 | 3 |
| `6:20` | 关节位置观测 | 14 |
| `20:34` | 关节速度观测 | 14 |
| `34:48` | 上一步动作 | 14 |
| `48:51` | 前进、横移、偏航速度指令 | 3 |
| `51:55` | 头部命令槽，本任务置零 | 4 |
| `55:61` | 身体命令槽，本任务置零 | 6 |

观测必须按配置构造，包括相对默认姿态、尺度和归一化等约定，不能将原始传感器值任意拼接后直接输入模型。

输出 `actions[1,14]` 对应14个受控关节，经环境的缩放、默认姿态偏置与限幅等处理，转换为关节目标。网络输出不等同于原始角度或力矩。14个受控关节分布在左右腿与颈部/头部。索引必须与对应模型的关节排列一致。

### 数据从哪里来

真机身体角速度来自IMU陀螺仪；重力投影由姿态估计计算；关节角度由舵机反馈或编码器得到；关节速度可由反馈或角度差分得到。编码器零位、轴方向、传动比和单位需按硬件校准。上一动作和目标速度由程序提供。

仿真中，物理引擎根据力、惯性和接触计算状态变化，程序读取关节状态与虚拟传感器，再构造观测。仿真能读取球状态，但Actor不一定获准使用。

## 3. LSTM记忆和ONNX契约

```text
61维观测 → LSTM(256) → ELU MLP(512→256→128) → 14维动作
```

| 输入／输出 | 形状 | 含义 |
|---|---|---|
| `obs` | `[1,61]` | 当前观测 |
| `h_in`、`h_out` | `[1,1,256]` | 输入、输出隐藏状态 |
| `c_in`、`c_out` | `[1,1,256]` | 输入、输出记忆单元状态 |
| `actions` | `[1,14]` | 动作 |

LSTM状态维度依次表示：层数×方向数、batch大小、隐藏维度。因此h、c各有256个浮点数，前面的1不是记忆只有一个数字，也不是历史帧数。

标准LSTM通过门控更新状态：

```text
c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

f、i、o是遗忘门、输入门、输出门；g是候选记忆；⊙表示逐元素乘法。上一步状态已经压缩了更早的历史，所以“当前观测＋上一步记忆”仍能利用多步信息。它不完整保存所有历史，状态各分量也没有人为规定的物理含义。

运行时把 `h_out/c_out` 传给下一次调用。首次启动、策略切换或回合重置时清零；普通速度指令变化时保留。ONNX调用方负责维护这些状态。50 Hz不表示一次输入过去50帧。

## 4. PPO、奖励和优化器

PPO（Proximal Policy Optimization）是策略梯度强化学习算法，包含采样、优势估计、训练目标和参数更新流程；它不等同于损失函数。LSTM是网络结构，Adam是梯度优化器，lr是学习率。

```text
使用当前策略采样 → 计算回报与优势 → 多轮小批量优化Actor/Critic → 重新采样
```

训练时Actor输出动作分布并采样动作。Critic估计长期回报，优势衡量动作相对价值基线的好坏。LSTM与动作头通过策略损失联合训练，没有人为标注的“正确动作label”；Critic的目标由采样奖励和价值估计构造。

PPO-Clip的核心目标：

$$
L^{CLIP}=E[\min(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]
$$

其中概率密度比 $\rho_t=\pi_\theta(a_t|\text{观测与历史})/\pi_{old}(a_t|\text{观测与历史})$，A是估计优势。裁剪降低策略过度更新的激励，不保证所有概率比值严格处于区间内。实际训练还包括价值损失和熵项。

原论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

### 奖励设计与动作滤波

奖励同时考虑速度跟踪、直立、居中、脚球接触和根部高度，并惩罚动作突变、接近关节限位等行为。最终奖励是按任务配置加权的组合。

动作变化惩罚让策略自行权衡平滑与及时纠正。部署时额外加入训练中没有的低通滤波会改变动作响应并引入延迟，因此不能随意添加。若需要滤波，应将其纳入匹配的训练与验证流程。

## 5. 短训实验与日志解读

本次实验环境：Ubuntu、NVIDIA L40、Warp 1.12.0、CUDA Toolkit 12.9、Driver 12.8，并存在HAMI管理层。训练从公开篮球checkpoint继续，不是从随机初始化训练。

| 项目 | 结果 |
|---|---|
| 运行名 | `basketball-no-mempool-smoke` |
| 并行环境 | 64 |
| 新增PPO迭代 | 5 |
| 每轮每环境控制步 | 24 |
| 总采样 | 7680＝64×24×5 |
| 学习率 | 2e-5 |
| 训练循环耗时 | 约41秒，不含完整安装等前置耗时 |
| 末轮平均累计回报 | 6.73 |
| 末轮value/surrogate loss | 0.1435 / -0.0300 |
| 完成标记 | `FINETUNE_DONE` |
| 输出目录 | `logs/rsl_rl/basketball/2026-10-14_13-38-48_basketball-no-mempool-smoke` |

![64环境×5轮PPO续训完成截图](images/training-smoke.png)

7003/7004是沿checkpoint继续计数的迭代标签，不代表这次训练7004轮。steps/s是跨并行环境的采样吞吐量，不是50 Hz控制频率。Mean action std是动作分布标准差，不是实测关节抖动。

Mean episode length单位是控制步；训练起始回合进度随机化与短统计窗口会影响它，不能直接当作独立稳定时间评测。分项奖励与Mean reward统计方式不同，不宜直接求和对照。

`fell=0`、`nan_state=0`表示对应日志窗口未记录这类终止。它们不证明独立评测成功率100%。生存率定义为指定评测时长内未摔倒的环境比例。它衡量固定测试条件下的持续稳定性，与累计奖励是不同指标。

$$
\text{生存率}=\frac{\text{评测时长内未摔倒的环境数}}{\text{参与评测的环境总数}}\times100\%
$$

## 6. 安装与训练故障定位

### GitHub依赖下载中断

BAM来自Git依赖：`better-actuator-models @ git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967`。

出现HTTP/2 CANCEL、GnuTLS异常终止、Empty reply等网络错误。切换HTTP/1.1未解决该服务器连接问题，因此准备指定提交的Git bundle，从本地仓库获取依赖。bundle需先上传服务器；当前目录缺少文件时无法clone。

```bash
git clone --bare ./bam-locked.bundle /tmp/microduck-bam-local.git
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=url.file:///tmp/microduck-bam-local.git.insteadOf \
GIT_CONFIG_VALUE_0=https://github.com/Rhoban/bam.git \
uv sync --locked
```

这只临时替换BAM的获取地址，保持锁定提交；其他未缓存依赖仍需联网。Git依赖与PyPI依赖走不同下载链路，仅更换PyPI镜像不能解决GitHub源码下载中断。

### HAMI与CUDA Graph捕获

默认路径报 `cuMemPoolGetAttribute failed res=900`，继而出现901及100 KiB分配失败。900描述捕获期间不允许的操作；901是先前捕获错误的后续。不能据最终小额分配失败断言显存耗尽。

已验证的绕开方式：初始化前关闭Warp内存池，使mjlab 1.3.0不启用CUDA Graph，保留CUDA计算。短训随后成功。尚未分别隔离内存池、Graph与HAMI各组件，因此记录为该路径的兼容问题，不认定具体组件已被最终证明有缺陷。

可复用入口：[run_warp_compat.py](scripts/run_warp_compat.py)。它只影响当前进程，不修改上游代码或依赖版本。

## 7. 训练、推理与模型导出

训练阶段需要环境采样、奖励、Critic与优化器；推理阶段使用Actor根据观测和循环状态选择动作，不进行PPO参数更新。录制仿真视频属于推理过程，不等于重新训练。

Checkpoint保存用于恢复训练的状态，可能包含Actor、Critic、优化器和迭代信息；ONNX保存部署用的计算图。两者不是换一个文件扩展名的关系。导出时需要保留观测归一化和动作处理约定，循环策略还需要显式暴露h/c输入输出。

PyTorch/ONNX parity比较相同输入与循环状态下的动作和状态输出，判断导出是否保持数值一致性。模型接口与数值一致性通过，也不能替代机器人闭环表现评测。

视频输出目录由录像脚本管理，模型输出目录由导出参数管理，两者可能不同。视频用于检查身体姿态、接触和动作现象；生存率与速度误差用于量化表现。

## 参考资料

本文按概念理解与本次实验重新组织，不转载上游教程正文或源码。模型结构和任务配置以项目实现为准。

- [Datawhale MicroDuck专题](https://github.com/datawhalechina/every-embodied/tree/main/16-专题组队学习/05-MicroDuck运动策略与任务开发专题)
- [MicroDuck Playground](https://github.com/Vottivott/microduck-playground)
- [篮球预训练模型](https://huggingface.co/HannesVonEssen/microduck-basketball)
- [PPO原论文](https://arxiv.org/abs/1707.06347)
