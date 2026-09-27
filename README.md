# Stereo VINS-Mono

基于 [HKUST VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) 改造的紧耦合双目视觉惯性里程计。本项目保留原有单目 VIO、滑动窗口优化和回环模块，在前端、初始化和后端中加入双目观测，使系统能够直接利用双目基线恢复尺度，并在优化过程中持续使用右目约束。

> 本仓库是面向算法研究和实验复现的 ROS 1 工程，并非 VINS-Fusion 的替代品。当前双目流程主要使用 EuRoC 数据集验证。

## 主要特点

- **兼容单目模式**：通过同一套节点和配置切换单目/双目，原有单目流程仍可使用。
- **严格双目同步**：左右图像按时间戳配对，不将错位帧送入双目匹配。
- **异步双目匹配**：左目时序跟踪与右目匹配解耦，双目任务由独立工作线程处理。
- **多级匹配筛选**：包含前后向光流、边界、极线误差、视差和深度范围检查。
- **双目深度初始化**：由有效视差生成度量深度，为特征逆深度提供初值。
- **固定尺度初始化**：利用双目路标与 PnP 恢复窗口内度量位姿，再结合 IMU 求解速度、重力和陀螺仪偏置，不再依赖单目尺度估计。
- **紧耦合右目约束**：滑动窗口中加入同帧双目和跨帧双目重投影残差，右目观测同时参与非线性优化与边缘化。
- **可复现实验工具**：提供 EuRoC 单目/双目交错运行、完整性检查、资源统计、ATE/RPE 计算和报告生成。
- **IMU 噪声模型可选**：兼容原有离散标准差配置，也支持连续时间噪声密度。

## 系统流程

```text
left image ──> temporal tracking ───────────────┐
                                                ├─> feature observations ─> sliding-window VIO
right image ─> exact-time pairing ─> stereo ───┘            │
                                      matching               ├─ left reprojection factors
                                          │                  ├─ right reprojection factors
                                          └─ metric depth ───└─ stereo metric initialization

IMU ───────────────> pre-integration ────────────────────────> motion constraints
```

双目前端以左目特征 ID 为主线。右目匹配成功后，同一条观测会携带右目归一化坐标和可选的双目深度；匹配失败不会删除左目特征，因此系统可以在局部双目退化时继续依靠左目与 IMU 工作。

## 与原版 VINS-Mono 的主要差异

| 模块 | 原版 VINS-Mono | 本仓库 |
| --- | --- | --- |
| 图像输入 | 单目 | 单目或严格同步双目 |
| 前端 | 单目时序光流 | 左目时序光流 + 右目双目匹配 |
| 特征深度 | 由多帧三角化逐步恢复 | 可由双目视差直接初始化 |
| 初始化尺度 | 单目视觉惯性对齐求尺度 | 可使用双目度量初始化，无尺度状态 |
| 后端视觉残差 | 左目跨帧重投影 | 左目残差 + 同帧/跨帧右目残差 |
| 边缘化 | 单目视觉和 IMU 因子 | 同时处理右目视觉因子 |
| 评测 | 基础轨迹输出 | 自动化单目/双目 ATE、RPE、耗时和资源评测 |

## 依赖

- Linux
- ROS 1 与 catkin
- OpenCV
- Eigen3
- Ceres Solver

工程沿用原版 VINS-Mono 的 ROS 1 构建结构。原上游环境为 Ubuntu 16.04、ROS Kinetic 和 Ceres 1.14；在其他 ROS 1 发行版上可能需要调整依赖版本。

安装常见 ROS 依赖：

```bash
sudo apt install ros-${ROS_DISTRO}-cv-bridge \
                 ros-${ROS_DISTRO}-tf \
                 ros-${ROS_DISTRO}-message-filters \
                 ros-${ROS_DISTRO}-image-transport
```

## 构建

将仓库放入 catkin 工作空间：

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/XiaoWangba-Chen/VIO-stereo.git VINS-Mono
cd ..
catkin_make
source devel/setup.bash
```

## EuRoC 数据

从 [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) 下载数据。自动评测工具使用解压后的 EuRoC 目录，至少需要：

```text
<sequence>/mav0/
├── cam0/
│   ├── data.csv
│   └── data/
├── cam1/
│   ├── data.csv
│   └── data/
├── imu0/data.csv
└── state_groundtruth_estimate0/data.csv
```

仓库提供两套对照配置：

- `config/euroc/euroc_mono_eval.yaml`
- `config/euroc/euroc_stereo_eval.yaml`

双目配置使用已经校正的 EuRoC 相机模型：

- `config/euroc/euroc_rectified_cam0.yaml`
- `config/euroc/euroc_rectified_cam1.yaml`

## 手动运行

先启动 ROS：

```bash
roscore
```

单目模式：

```bash
roslaunch vins_estimator euroc.launch \
  config_path:=$(rospack find feature_tracker)/../config/euroc/euroc_mono_eval.yaml
```

双目模式：

```bash
roslaunch vins_estimator euroc.launch \
  config_path:=$(rospack find feature_tracker)/../config/euroc/euroc_stereo_eval.yaml
```

输入话题默认为：

| 数据 | 话题 |
| --- | --- |
| 左目图像 | `/cam0/image_raw` |
| 右目图像 | `/cam1/image_raw` |
| IMU | `/imu0` |

可以直接播放带有这些话题的 EuRoC ROS bag，也可以使用仓库中的数据播放器：

```bash
python3 tools/play_euroc_ros1.py /path/to/MH_03_medium --stereo --rectify-euroc
```

启动 RViz：

```bash
roslaunch vins_estimator vins_rviz.launch
```

## 双目配置

核心参数位于 YAML 配置中：

| 参数 | 含义 |
| --- | --- |
| `num_of_cam` | `1` 为单目，`2` 为双目 |
| `image_topic` / `image1_topic` | 左、右目图像话题 |
| `cam0_calib` / `cam1_calib` | 左、右目标定文件 |
| `extrinsicRotation1` / `extrinsicTranslation1` | 右相机到 IMU 的外参 |
| `stereo_track` | 启用双目匹配 |
| `stereo_fb_threshold` | 前后向光流一致性阈值，单位为像素 |
| `stereo_epipolar_threshold` | 前端极线误差阈值，单位为像素 |
| `stereo_backend_reprojection_threshold` | 后端接收右目观测的重投影阈值 |
| `stereo_min_depth` / `stereo_max_depth` | 可接受的双目深度范围，单位为米 |
| `stereo_use_depth_init` | 使用双目深度初始化特征逆深度 |
| `stereo_use_metric_init` | 使用固定尺度双目初始化 |
| `stereo_use_right_factors` | 在优化与边缘化中加入右目因子 |
| `imu_noise_mode` | `discrete_std` 或 `continuous_density` |

推荐完整双目模式：

```yaml
num_of_cam: 2
stereo_track: 1
stereo_use_depth_init: 1
stereo_use_metric_init: 1
stereo_use_right_factors: 1
```

这些选项具有依赖关系。例如，度量初始化依赖双目深度初始化，而深度初始化和右目因子都要求启用双目跟踪。程序启动时会检查不合法的组合并直接报错。

## 自动化基准测试

确认 ROS 1 环境、工作空间和数据完整：

```bash
python3 tools/run_euroc_benchmark.py \
  --dataset /path/to/MH_03_medium \
  --validate-only
```

执行单目/双目对照实验：

```bash
python3 tools/run_euroc_benchmark.py \
  --dataset /path/to/MH_03_medium \
  --runs 3
```

默认按 `mono, stereo, stereo, mono, mono, stereo` 的顺序交错运行，降低运行顺序对资源和精度比较的影响。结果默认写入 `outputs/euroc_eval/MH_03_medium`，包括：

- 每次运行使用的配置、轨迹和日志
- 前端、双目匹配、后端优化和边缘化耗时
- CPU 与内存时间序列
- SE(3) 对齐 ATE
- 一秒间隔 RPE
- Sim(3) ATE 与尺度诊断
- 成功率、轨迹覆盖率和双目配对完整性

仅重新生成报告：

```bash
python3 tools/evaluate_euroc_benchmark.py
```

更完整的实验说明见 [EuRoC 单目/双目评测文档](docs/euroc_mono_stereo_benchmark.md)。

## 测试

C++ 单元测试已接入 catkin：

```bash
cd ~/catkin_ws
catkin_make run_tests
catkin_test_results
```

覆盖内容包括：

- 双目光流、极线、视差和深度筛选
- 双目观测在特征生命周期中的传播
- 同帧和跨帧双目重投影因子的解析 Jacobian
- 双目度量路标构建与 PnP

评测指标的纯 Python 测试：

```bash
python3 -m unittest tools/test_euroc_benchmark_metrics.py
```

## 输出与诊断

正常运行时，估计轨迹写入配置中的 `output_path`。设置环境变量 `VINS_BENCHMARK_DIR` 后，还会生成：

- `feature_timing.csv`：时序跟踪、双目匹配、队列深度和发布时间
- `backend_timing.csv`：后端处理、优化、边缘化和状态信息

`datasets/`、`outputs/`、ROS bag、ROS 2 bag 和 Python 缓存均已加入 `.gitignore`，不会随代码提交。

## 当前限制

- 双目输入必须具有可靠时间戳；当前实现采用严格时间戳配对。
- 双目度量初始化依赖正确的左右目标定和相机—IMU 外参。
- 当前主要覆盖水平校正双目；非校正相机需要确保相机模型和极线阈值设置正确。
- 回环模块继承自原版 VINS-Mono，双目改动主要集中于 VIO 前端、初始化和滑动窗口后端。
- 自动评测脚本依赖 ROS 1 命令行工具以及 EuRoC 目录格式。

## 上游项目与引用

本项目建立在 HKUST Aerial Robotics Group 的 VINS-Mono 上：

- Tong Qin, Peiliang Li, Zhenfei Yang, Shaojie Shen, **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**, IEEE Transactions on Robotics, 2018.
- Tong Qin, Shaojie Shen, **Online Temporal Calibration for Monocular Visual-Inertial Systems**, IROS 2018.

如果在研究中使用本仓库，请引用原始 VINS-Mono 论文；原项目 BibTeX 位于 [support_files/paper_bib.txt](support_files/paper_bib.txt)。

本项目使用 Ceres Solver、DBoW2 和 camodocal 相机模型代码。感谢原作者及相关开源项目。

## 许可证

本仓库继承原项目的 [GPLv3](LICENCE) 许可证。
