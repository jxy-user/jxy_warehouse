# 任务模块（active_module）

主配置仅一条：`default.yaml` 里的 `active_module`，指向本目录下同名 `<id>.yaml`。

## 现有模块

| ID | 说明 |
|----|------|
| `bone` | 骨骼 X 线/DR 五维多标签（默认） |
| `chest_mvp` | 胸部影像五维多标签（原版联调示例） |
| `posture_stub` | **坐姿检测占位**：`pipeline: posture_stub`，不接 MMCANet；训练/评测被拒绝；推理走占位或专用 `/posture/*` |

## 新增模块（不改核心代码）

1. 复制 `bone.yaml` 为 `my_task.yaml`（文件名即 **模块 ID**）。
2. 修改 `data.task`、`infer.class_names`、`data.num_classes`、`model.num_classes`（须一致）。
3. 指定独立 `csv_path`、`train.save_path`、`infer.checkpoint_path`，避免与别的任务互相覆盖权重。
4. 在 `default.yaml` 中设置 `active_module: my_task`。
5. 新增文档可放在 `workspace/01_文档/docs/`，并在 `问题修复总结.md` 记一条。

推理、训练、评测入口自动读取合并后的配置，无需改 Python 业务逻辑。

## 坐姿占位与主流程并行

- 服务启动后，无论 `active_module` 是否为 `posture_stub`，均可调用 **`GET /posture/info`**、**`POST /posture/analyze`**（配置固定来自 `posture_stub.yaml`）。
- 若将 `active_module` 设为 `posture_stub`，则 `/infer`、`/infer_with_image` 亦返回占位 JSON，且不加载 MMCANet。
