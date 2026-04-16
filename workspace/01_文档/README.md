# jxy_warehouse

## AI医学影像大创方案文档（已整理版）

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

- 训练入口：`../02_工程/src/training/train.py`
- 评测入口：`../02_工程/src/eval/evaluate.py`
- 推理API：`../02_工程/src/api/infer_api.py`
- 默认配置：`../02_工程/src/config/default.yaml`

### 快速开始

1. 安装依赖  
   `cd ../02_工程 && pip install -r requirements.txt`
2. 训练mock基线  
   `cd ../02_工程 && powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1`
3. 评测mock基线  
   `cd ../02_工程 && powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1`
4. 启动API服务  
   `cd ../02_工程 && powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1`

## 真实数据接入（CSV模式）

默认是 `mock` 模式。切换真实数据时，在 `../02_工程/src/config/default.yaml` 将 `data.mode` 改为 `csv`，并配置：

- `data.csv_path`：CSV标注路径（示例：`../03_数据/data/samples/train.csv`）
- `data.image_root`：影像根目录（示例：`../03_数据/data/images`）

## 仓库目录规范

- 主文件夹：`workspace/`
- 文档目录：`workspace/01_文档/`
- 工程目录：`workspace/02_工程/`
- 数据目录：`workspace/03_数据/`

CSV列规范：

- 基础列：`image_path`,`text`
- 结构化特征：`struct_0` 到 `struct_5`
- 多标签：`label_0` 到 `label_4`

## 中文推理接口

- 健康检查：`GET /health`
- 文本+结构化推理：`POST /infer`
- 图片上传推理：`POST /infer_with_image`（返回风险分数与Grad-CAM热图路径）

### 接口安全（鉴权+限流）

- 已启用API Key鉴权，请在请求头中传：`X-API-Key`
- 服务端环境变量：`MEDFUSE_API_KEYS=dev-key,demo-key`
- 推理接口默认限流：每个`API Key + IP`每分钟60次（可在`default.yaml`调节）
- 超限返回：`429`；缺失Key返回：`401`；无效Key返回：`403`

### 安全验收清单（实测）

1. 缺失API Key请求`/infer` -> 返回`401`  
2. 错误API Key请求`/infer` -> 返回`403`  
3. 正确API Key请求`/infer` -> 返回`200`并输出风险分数  
4. 连续高频请求`/infer`超过阈值 -> 返回`429`  
5. `GET /health`在公开配置下可正常返回`200`

## 深色前端页面

- 演示页面：`frontend_dark.html`
- 用途：患者友好简洁版操作页（上传图片 -> 一键分析 -> 查看结果）
- 特点：仅保留必要输入，降低操作复杂度，普通患者也能快速使用