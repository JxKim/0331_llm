# SFTConfig 参数整理

## 1. 了解SFTConfig所涉及的几个部分

一套 `SFTConfig` 参数配置，主要包含如下几个方面：

- 数据规模相关，包含：per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, max_steps, num_train_epochs等
- 训练可视化相关，包含：logging_strategy, logging_steps, report_to等
- 优化器和学习率策略，包含：learning_rate, lr_scheduler_type, warmup_step，optim等
- 评估和保存相关，包含：eval_strategy, eval_steps, save_strategy, save_steps,load_best_model_at_end, greater_is_better,metric_for_best_model, save_total_limit,output_dir等
- 优化相关，包含：bf16, gradient_checkpointing, activation_offloading, max_length, use_liger_kernel, model_init_kwargs
- 其他参数，此处重点介绍：assistant_only_loss, chat_template_path

## 2. 数据规模相关

这组参数决定“训练跑多大、跑多久”。

### `per_device_train_batch_size`

- **类型**：`int`
- 每张卡每一步喂多少条训练样本
- 如果是单卡，就是单步 batch size
- 如果是多卡，总 batch 还要乘设备数

### `per_device_eval_batch_size`

- **类型**：`int`
- 每次验证时，每次前向传播的样本数量
- 如果是单卡，就是单步 batch size
- 如果是多卡，总 batch 还要乘设备数

### `gradient_accumulation_steps`

- **类型**：`int`
- 梯度累积步数
- 显存不够时非常常用
- 比如每步只跑 1 条，但累积 8 步再更新一次，相当于有效 batch 变大了

### `max_steps`

- **类型**：`int`
- 最大训练步数
- 如果设置为正数，通常会优先生效
- 常见理解：`num_train_epochs` 控制“按轮数训练”，`max_steps` 控制“按步数训练”

### `num_train_epochs`

- **类型**：`float`
- 训练多少轮
- 一轮表示把整个训练集完整跑一遍

### 有效 batch size 的理解

常见公式：

`有效 batch size = per_device_train_batch_size × 设备数 × gradient_accumulation_steps`

例如：

- 单卡
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`

那么有效 batch size 可以理解为 `8`。

实际执行过程中，总的 step 数计算公式为：

`total_steps = （total_trainable_data_nums + 有效 batch size - 1） // 有效 batch size`

## 3. 训练可视化相关

这组参数决定“训练过程怎么观察”。

### `logging_strategy`

- **类型**：`IntervalStrategy | str`
- 什么时候记录日志
- 常见是 `steps`

### `logging_steps`

- **类型**：`float`
- 每多少步记录一次日志

### `report_to`

- **类型**：`None | str | list[str]`
- 日志汇报到哪里
- 常见有：
  - `none`
  - `tensorboard`
  - `wandb`

## 4. 学习率和优化器

这组参数决定“模型参数怎么更新”。

### `learning_rate`

- **类型**：`float`
- 初始学习率
- 是最核心的超参数之一

### `lr_scheduler_type`

- **类型**：`SchedulerType | str`
- 学习率调度器类型
- 常见如：
  - `linear`
  - `cosine`
  - `constant`

### `warmup_steps`

- **类型**：`float | int`
- 预热比例/预热步数
- 训练刚开始时让学习率从较小值逐渐升上去

### `optim`

- **类型**：`str`
- 优化器类型，默认是AdamW

## 5. 评估和保存相关

这组参数决定“什么时候验证模型效果、多久保存一次、是否加载最佳模型”。

### `eval_strategy`

- **类型**：`IntervalStrategy | str`
- 是否评估，以及按什么节奏评估
- 常见：
  - `no`
  - `steps`
  - `epoch`

### `eval_steps`

- **类型**：`float | None`
- 每多少步评估一次

### `metric_for_best_model`

- **类型**：`str | None`
- 用哪个指标判断“最佳模型”

### `greater_is_better`

- **类型**：`bool | None`
- 指标越大越好还是越小越好
- 比如 accuracy 通常越大越好，loss 通常越小越好

### `load_best_model_at_end`

- **类型**：`bool`
- 训练结束后是否自动加载表现最好的 checkpoint

### `save_strategy`

- **类型**：`SaveStrategy | str`
- 保存策略
- 常见：
  - `steps`
  - `epoch`

### `save_steps`

- **类型**：`float`
- 每多少步保存一次

### `save_total_limit`

- **类型**：`int | None`
- 最多保留多少个 checkpoint
- 超过后会删除更早的 checkpoint

### `output_dir`

- **类型**：`str | None`
- 输出目录
- 用来保存模型、checkpoint、日志等内容

## 6. 优化相关

这一组主要决定“显存、速度和输入长度怎么平衡”，在单卡 SFT 里尤其重要。

### `bf16`

- **类型**：`bool | None`
- 是否启用 `bfloat16`
- 如果显卡支持，通常优先使用

### `gradient_checkpointing`

- **类型**：`bool`
- 梯度检查点
- 通过“多算一点”来“少占一些显存”
- 大模型训练里很常见

### `activation_offloading`

- **类型**：`bool`
- 将部分激活转移到别的位置，例如 CPU
- 本质是“用时间换显存”

### `max_length`

- **类型**：`int | None`
- 每条样本允许的最大 token 长度
- 超过就需要截断
- `max_length` 越大，单条样本能保留的信息越多，但显存压力也越大

### `use_liger_kernel`

- **类型**：`bool`
- 是否启用 liger kernel
- 可以提升速度或降低显存占用


### `model_init_kwargs`

- **类型**：`dict[str, Any] | str | None`
- 模型初始化时传入的额外参数
- 可以在此处传入 `{"attn_implementation": "kernels-community/flash-attn2"}`，表示训练过程中使用 flash attention 来加快训练过程，减少显存占用

## 7. 其他参数

这一组放一些不完全属于前面几类，但在 SFT 中经常需要单独理解的参数。

### `assistant_only_loss`

- **类型**：`bool`
- 是否只对 assistant 回复部分计算 loss
- 很适合多轮对话 SFT

### `chat_template_path`

- **类型**：`str | None`
- 聊天模板路径
- 用于把多轮对话格式化成模型输入文本

## 8. 最值得优先理解的一小撮参数

如果是做单卡 SFT，不要试图一次记住全部参数。先抓下面这些最关键：

- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs` 或 `max_steps`
- `learning_rate`
- `lr_scheduler_type`
- `warmup_ratio`
- `bf16`
- `gradient_checkpointing`
- `logging_steps`
- `save_steps`
- `save_total_limit`
- `eval_strategy`
- `max_length`
- `assistant_only_loss`
- `chat_template_path`

如果这几个已经理解得比较清楚，基本就能开始配置一次正常的 SFT 训练了。

## 9. 记忆总框架

### 第一步：训练怎么跑

- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `max_steps` / `num_train_epochs`

### 第二步：过程怎么看

- `logging_strategy`
- `logging_steps`
- `report_to`

### 第三步：参数怎么更新

- `learning_rate`
- `lr_scheduler_type`
- `warmup_ratio`

### 第四步：评估和保存怎么管理

- `eval_strategy`
- `eval_steps`
- `save_strategy`
- `save_steps`
- `load_best_model_at_end`
- `greater_is_better`
- `metric_for_best_model`
- `save_total_limit`
- `output_dir`

### 第五步：显存和速度怎么平衡

- `bf16`
- `gradient_checkpointing`
- `activation_offloading`
- `max_length`
- `use_liger_kernel`
- `padding_free`
- `model_init_kwargs`

### 第六步：其他 SFT 参数怎么配

- `assistant_only_loss`
- `chat_template_path`