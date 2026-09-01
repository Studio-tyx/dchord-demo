# DChord v2.2 训练与Torch推理复现包

## 1. 复现范围

本包面向标准 TorchSpec 源码，复现以下五项：

1. 1,000条CelebA自蒸馏与冻结Schema预校准；
2. 基于开源DFlash初始化的DChord-K3/Anchor40在线训练；
3. 普通自回归生成（AR）与 Schema 条件化 AR；
4. DChord-K3 的 Torch 推理路径；
5. 开源 DFlash-B16 与 DChord-K3 的接受长度对比。

本包**包含训练代码，但不包含训练检查点或自蒸馏结果**。训练者需要从开源DFlash重新训练。包内没有DChord自定义服务后端代码；在线训练只复用标准TorchSpec已经提供的target隐藏状态抽取和Mooncake传输机制。Torch推理用于检查正确性和接受长度，其墙钟时间不是生产系统速度。

## 2. 固定的软件与数据条件

- 标准 TorchSpec revision：`43dec00d39a919309fb9b531b3fab66bdadbc397`
- conda 环境：`torchspec`
- Target：`/root/autodl-tmp/models/Qwen3.5-4B`
- 开源草稿：`/root/autodl-tmp/models/Qwen3.5-4B-DFlash`
- CelebA parquet：`/root/autodl-tmp/datasets/celeba/img_align+identity+attr`
- 精度：BF16
- 注意力：PyTorch SDPA
- 自蒸馏：greedy、`enable_thinking=False`、冻结Schema约束
- 训练：一张GPU运行target隐藏状态抽取，另一张GPU训练DChord草稿头
- 推理：greedy、`enable_thinking=False`、batch=1

包内已经提供同款one-shot prompt、冻结Schema表面格式、1,000条训练清单、DFlash配置以及512条测试清单。正式接受长度只取测试清单前128条。

如果目录不同，可以设置：

```bash
export DCHORD_REPRO_ROOT=/root/autodl-tmp
export DCHORD_TARGET=/path/to/Qwen3.5-4B
export DCHORD_OFFICIAL_DFLASH=/path/to/Qwen3.5-4B-DFlash
export DCHORD_CELEBA_PARQUET=/path/to/celeba/img_align+identity+attr
export DCHORD_CELEBA_ROOT=/path/to/celeba
```

## 3. 安装补丁

```bash
cd /path/to/repro_torch_backend_v1
bash apply.sh /root/autodl-tmp/TorchSpec
```

`apply.sh`会先检查标准TorchSpec的commit和目标文件是否干净，然后：

- 对`torchspec/models/draft/dflash.py`应用两处最小修改：支持稀疏绝对位置的RoPE长度，以及K3小型布尔注意力掩码的SDPA路径；
- 新增`torchspec/models/dchord.py`；
- 新增`torchspec/training/dchord_trainer.py`；
- 在标准训练配置、trainer分发、Qwen3-VL模板、DFlash初始化和最终导出位置增加最小DChord接入；
- 安装独立的`experiments/dchord/repro/`自蒸馏、训练准备、推理和汇总脚本；
- 执行Python语法检查。

它不会安装我们另外开发的服务推理patch。`torchspec/inference/engine/vllm_engine.py`只有一行训练期边界修正：隐藏状态抽取会额外产生一个token，因此`max_model_len=max_seq_length+1`。这属于标准TorchSpec在线训练路径，不是DChord服务后端适配。

验收：终端必须出现：

```text
[PASS] DChord training and Torch inference code installed
No custom serving-backend patch was installed.
```

## 4. DChord到底训练什么

训练样本的输入是CelebA图像和普通one-shot Schema prompt；监督输出是target自蒸馏得到的完整40字段JSON。程序利用冻结Schema模板找到40个value位置，但只在这些value位置计算训练目标。

每次训练随机选择40个起点。每个起点最多建立一个K3块：

- 三个查询只对应三个连续value；
- 查询能看到当前块第一个value之前的target上下文；
- 块内三个查询可以互相注意；
- 查询不能看到未来target value，避免答案泄漏；
- key、引号、冒号、逗号和空格不计算loss。

草稿仍使用完整词表输出头。损失为三个value位置的交叉熵，并使用DFlash的D-PACE位置权重；不是把任务简化成一个专用二分类器。推理时再把预测的value通过Schema编译器展开成完整`key+value`候选。

## 5. 自蒸馏与预校准

这一步生成1,000条target teacher数据，不需要DChord检查点：

```bash
conda activate torchspec
cd /root/autodl-tmp/dchord_v22_torch_repro_code_v2
bash scripts/run_self_distill_1k.sh /root/autodl-tmp/TorchSpec
```

脚本依次执行：冻结1k清单、生成0–499、生成500–999、重复生成前32条、最终合并与校验。输出：

```text
/root/autodl-tmp/reports/dchord_torch_repro/self_distill_1k/dchord_teacher_1000.jsonl
```

验收点：1,000/1,000合法Schema、40,000个value具有正确上下文词元映射、32条重复生成的decision一致率不低于99.5%。该任务耗时较长，日志保存在输出目录的`logs/`。

## 6. 准备TorchSpec在线训练数据

```bash
bash scripts/prepare_training_data.sh \
  /root/autodl-tmp/TorchSpec cuda:0
```

该步骤做两件事：

1. 验证Qwen3.5 target、开源DFlash权重和TorchSpec DFlash结构兼容；
2. 把teacher JSONL转换成TorchSpec多模态conversation格式，并把开源DFlash权重转换为TorchSpec内部初始检查点。

验收输出：

```text
/root/autodl-tmp/reports/dchord_torch_repro/training/data/train_1000.jsonl
/root/autodl-tmp/models/Qwen3.5-4B-DFlash-torchspec-dchord/model.safetensors
```

## 7. 正式训练DChord-K3/Anchor40

训练需要两张GPU，默认由TorchSpec分配一张给target在线前向、一张给DChord训练：

```bash
bash scripts/run_train_anchor40_1k.sh /root/autodl-tmp/TorchSpec
```

冻结配置为：K=3、40个随机起点、1,000条、1 epoch、梯度累积4、250次优化、BF16、学习率`1e-5`、D-PACE。日志位置：

```text
/root/autodl-tmp/reports/dchord_torch_repro/training/logs/03_train_anchor40_1k.log
```

最终检查点：

```text
/root/autodl-tmp/reports/dchord_torch_repro/training/dchord_k3_anchor40_export/pytorch_model.bin
```

## 8. Schema条件化AR是什么

普通AR逐词元生成整个JSON，包括key、引号、冒号、空格和value。

Schema条件化AR预先冻结完整JSON模板。每到一个字段时，target只在该字段合法value中做一次选择；选择完成后，程序直接写入下一个字段之前的固定结构。CelebA的合法value是`true/false`。固定结构仍会经过target前向并更新上下文，但不再由模型自由生成。

先跑8条smoke：

```bash
conda activate torchspec
cd /root/autodl-tmp/dchord_v22_torch_repro_code_v2
bash scripts/run_schema_ar.sh /root/autodl-tmp/TorchSpec 8 cuda:0
```

验收点：

- `ar`与`schema`均完成8条；
- `exact_40_bool_records=8`；
- 每条Schema输出均包含顺序固定的40个key，且value都是布尔值；
- 正确性比较只看40个解析后的value，不要求空格、换行等表面文本与AR相同。

结果默认写入：

```text
/root/autodl-tmp/reports/dchord_torch_repro/schema_ar/
```

## 9. DChord-K3检查点接口

代码包不附带训练检查点。可以使用第7节重新训练得到的`pytorch_model.bin`，也可以把兼容的外部DChord-K3/Anchor40权重作为第二个参数传入。

```bash
export DCHORD_K3_CHECKPOINT=/path/to/pytorch_model.bin
```

加载采用`strict=True`。如果权重结构不匹配，程序会立即失败，不会静默忽略参数。

## 10. 正式接受长度对比

```bash
conda activate torchspec
cd /root/autodl-tmp/dchord_v22_torch_repro_code_v2
bash scripts/run_acceptance_128.sh \
  /root/autodl-tmp/TorchSpec \
  /path/to/pytorch_model.bin \
  cuda:0
```

脚本按顺序运行开源DFlash-B16和DChord-K3，避免两个方法同时占用GPU。每种方法先预热1条，再测试同一批128条，最后生成：

```text
/root/autodl-tmp/reports/dchord_torch_repro/acceptance_128/acceptance_summary.json
```

正式比较量不是“DFlash接受了多少raw token”与“DChord接受了多少value”的直接比值，因为二者单位不同。统一指标是：

> 每次target验证后，完整Schema输出实际推进了多少个target词元。

我们的参考结果为：

| 方法 | 草稿原始接受量/轮 | 完整输出推进/轮 |
|---|---:|---:|
| 开源DFlash-B16 | 7.084个raw token | 8.060个target token |
| DChord-K3 | 2.394个value decision | 27.979个target token |

DChord/DFlash完整输出推进比为`3.472×`。复现时BF16近边界可能产生少量差异，但必须满足：

- 两路均为128/128合法40布尔值Schema；
- 接受长度量级与参考值一致；
- 若value不同，单独列出字段和target margin，不能把格式差异算成value错误。

## 11. 文件说明

```text
apply.sh
patches/0001-dchord-torch-sparse-positions.patch
patches/0002-dchord-training-integration.patch
overlay/torchspec/models/dchord.py
overlay/torchspec/training/dchord_trainer.py
overlay/experiments/dchord/repro/self_distill_1k.py
overlay/experiments/dchord/repro/prepare_online_training.py
overlay/experiments/dchord/repro/training_compat.py
overlay/experiments/dchord/repro/configs/dchord_q35_anchor40_1k.yaml
overlay/experiments/dchord/repro/common.py
overlay/experiments/dchord/repro/cache_transaction.py
overlay/experiments/dchord/repro/torch_fourway_cached.py
overlay/experiments/dchord/repro/torch_fourway_nocache.py
overlay/experiments/dchord/repro/summarize_acceptance.py
overlay/experiments/dchord/repro/assets/*
scripts/run_schema_ar.sh
scripts/run_acceptance_128.sh
scripts/run_self_distill_1k.sh
scripts/prepare_training_data.sh
scripts/run_train_anchor40_1k.sh
expected_torch_results.json
```

## 12. 常见问题

### 为什么不比较Torch墙钟速度？

Qwen3.5包含循环GDN状态。Torch runner为了保证拒绝后状态正确，会保存、恢复并重放已接受前缀；接受位置是可信的，但时间包含适配器额外开销。生产速度应由集成推理后端测量，本包不包含该部分。

### Schema条件化AR是否需要DChord检查点？

不需要。它只使用target模型、冻结模板和合法value集合。

### DFlash-B16是否需要额外训练？

不需要。本包直接加载开源`Qwen3.5-4B-DFlash`。

### DChord-K3是否一次预测40个value？

不是。每轮最多提出3个连续value，target按首错规则验证；全块接受时同时提交标准bonus value。

### 训练时是否也注射Schema？

训练时用冻结Schema定位value监督位置，并给查询加入字段结构向量；监督文本本身仍是完整Schema。真正的固定key注射发生在推理阶段：草稿只提出value，编译器把value放回正确字段并补齐固定结构，再交给target验证。
