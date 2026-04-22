# jxy_warehouse

## 灵枢AI 与任务模块

- 在 `../02_工程/src/config/default.yaml` 中设置 **`active_module`**，即可在**不改训练/推理代码**的情况下切换任务（胸片示例、骨骼、或你自建的模块）。
- 模块文件目录：`../02_工程/src/config/modules/`，说明见同目录 `README.md`。
- 当前默认 **`active_module: bone`**（骨骼 X 线/DR 多标签，见 `docs/09_骨骼影像任务与标签说明.md`）。
- 恢复**原版胸片联调五类**时，将 `active_module` 改为 **`chest_mvp`**（CSV：`data/samples/train_chest_mvp.csv`，权重：`checkpoints/mmca_net_chest_mvp.pt`，需先训练或自备权重）。
- `docs/01_病种范围与数据源.md` 等仍以胸片数据为方法参考时可继续查阅。

## AI医学影像大创方案文档（已整理版）

本仓库已补充完整的大创方案与实施文档，入口如下：

- 总方案：`docs/00_项目总方案.md`
- 数据治理：`docs/02_数据治理与标注规范.md`
- 模型评测：`docs/03_多模态基线与离线评测.md`
- 小样本与部署：`docs/04_小样本与轻量化部署.md`
- 结题交付：`docs/05_结题交付包.md`

云开发必备文档：

- 数据库文档：`docs/云数据库设计.md`
- 云函数权限文档：`docs/云函数权限说明.md`

问题修复沉淀：

- 问题修改总结：`docs/问题修复总结.md`
- 骨骼任务与标签：`docs/09_骨骼影像任务与标签说明.md`
- 坐姿占位模块：`docs/10_坐姿检测模块占位说明.md`


## 第一版代码骨架（可直接运行）

- 训练入口：`../02_工程/src/training/train.py`
- 评测入口：`../02_工程/src/eval/evaluate.py`
- 推理API：`../02_工程/src/api/infer_api.py`
- 主配置：`../02_工程/src/config/default.yaml`
- 任务模块：`../02_工程/src/config/modules/*.yaml`

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

默认是 `mock` 模式。切换真实数据时，在 `default.yaml` 将 `data.mode` 改为 `csv`。**各任务 CSV 路径在对应模块 YAML 里**（如 `bone` → `train_bone.csv`，`chest_mvp` → `train_chest_mvp.csv`）；影像根目录由主配置的 `data.image_root` 指定（示例：`../03_数据/data/images`）。

## 仓库目录规范

- 主文件夹：`workspace/`
- 文档目录：`workspace/01_文档/`
- 工程目录：`workspace/02_工程/`
- 数据目录：`workspace/03_数据/`

CSV列规范：

- 基础列：`image_path`,`text`
- 结构化特征：`struct_0` 到 `struct_5`
- 多标签：`label_0` 到 `label_4`（语义随 `active_module` 变化：**骨骼模块**见 `docs/09_骨骼影像任务与标签说明.md`，**胸片模块**见 `modules/chest_mvp.yaml` 内 `infer.class_names`）

## 中文推理接口

- 健康检查：`GET /health`（返回 `pipeline`、`任务模块`、`类别` 等）
- 文本+结构化推理：`POST /infer`
- 图片上传推理：`POST /infer_with_image`（上传说明随模块变化；`pipeline=mmc_net` 时含 Grad-CAM）
- 自动路由推理：`POST /route_infer`（根据问题文本和是否上传影像自动分流到骨骼影像或坐姿占位模块，并返回路由原因与置信度）
- **坐姿占位（与当前 `active_module` 无关，始终挂载）**：`GET /posture/info`、`POST /posture/analyze`（详见 `docs/10_坐姿检测模块占位说明.md`）

### 接口安全（鉴权+限流）

- 已启用API Key鉴权，请在请求头中传：`X-API-Key`
- 服务端环境变量：`MEDFUSE_API_KEYS=dev-key,demo-key`
- 推理接口默认限流：每个`API Key + IP`每分钟60次（可在`default.yaml`调节）
- 超限返回：`429`；缺失Key返回：`401`；无效Key返回：`403`

### 跨域与前端地址策略

- 后端已启用CORS白名单，默认允许：`http://127.0.0.1:8000`、`http://localhost:8000`、`https://jxy-user.github.io`
- 前端默认请求：`http://127.0.0.1:8000`
- 前端支持通过URL参数覆盖接口地址：`frontend_dark.html?api=https://你的后端域名`
- 参数中的`api`会自动写入`localStorage`（键名：`medfuse_api_base_url`），后续刷新可持续生效
- 注意：GitHub Pages（HTTPS）不能直接调用本地HTTP接口，需改为可公网访问的HTTPS后端地址

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