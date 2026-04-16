# jxy_warehouse

## AI医学影像大创方案文档

本仓库已补充完整的大创方案与实施文档，入口如下：

- 总方案：`docs/AI_medical_imaging_innovation_proposal.md`
- 数据治理：`docs/02_data_governance_and_labeling_spec.md`
- 模型评测：`docs/03_multimodal_baseline_and_offline_eval.md`
- 小样本与部署：`docs/04_fewshot_and_lightweight_deployment.md`
- 结题交付：`docs/05_final_deliverables_package.md`

云开发必备文档：

- 数据库文档：`docs/cloud_database_design.md`
- 云函数权限文档：`docs/cloud_function_permissions.md`

问题修复沉淀：

- 问题修改总结：`docs/problem_fix_summary.md`


## 第一版代码骨架（可直接运行）

- 训练入口：`src/training/train.py`
- 评测入口：`src/eval/evaluate.py`
- 推理API：`src/api/infer_api.py`
- 默认配置：`src/config/default.yaml`

### 快速开始

1. 安装依赖  
   `pip install -r requirements.txt`
2. 训练mock基线  
   `powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1`
3. 评测mock基线  
   `powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1`
4. 启动API服务  
   `powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1`

## 真实数据接入（CSV模式）

默认是 `mock` 模式。切换真实数据时，在 `src/config/default.yaml` 将 `data.mode` 改为 `csv`，并配置：

- `data.csv_path`：CSV标注路径（示例：`data/samples/train.csv`）
- `data.image_root`：影像根目录（示例：`data/images`）

CSV列规范：

- 基础列：`image_path`,`text`
- 结构化特征：`struct_0` 到 `struct_5`
- 多标签：`label_0` 到 `label_4`

## 中文推理接口

- 健康检查：`GET /health`
- 文本+结构化推理：`POST /infer`
- 图片上传推理：`POST /infer_with_image`（返回风险分数与热图路径）