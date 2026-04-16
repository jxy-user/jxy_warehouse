# 云开发数据库文档（首版）

## 1. 目的

该文档用于约束云开发阶段的数据表设计。后续每新增云函数，必须先检查并更新本文件。

## 2. 集合设计（建议）

### 2.1 `patients_anonymous`
- 用途：匿名患者基础信息
- 主键：`patient_uuid`
- 字段：
  - `patient_uuid` (string, unique)
  - `gender` (string)
  - `age_range` (string)
  - `risk_factors` (array<string>)
  - `created_at` (datetime)

### 2.2 `imaging_studies`
- 用途：影像检查元数据
- 主键：`study_id`
- 字段：
  - `study_id` (string, unique)
  - `patient_uuid` (string, index)
  - `modality` (string, e.g. CXR/CT)
  - `study_time` (datetime)
  - `image_uri` (string)
  - `source_domain` (string)

### 2.3 `ai_inference_results`
- 用途：模型推理结果
- 主键：`inference_id`
- 字段：
  - `inference_id` (string, unique)
  - `study_id` (string, index)
  - `model_version` (string)
  - `risk_scores` (object)
  - `top_labels` (array<string>)
  - `explainability_uri` (string)
  - `created_at` (datetime)

### 2.4 `annotation_tasks`
- 用途：主动学习标注任务管理
- 主键：`task_id`
- 字段：
  - `task_id` (string, unique)
  - `study_id` (string, index)
  - `priority_score` (number)
  - `status` (string: pending/reviewed/closed)
  - `assignee` (string)
  - `updated_at` (datetime)

### 2.5 `inference_artifacts`
- 用途：推理中间产物（Grad-CAM热图、解释文本、缩略图）
- 主键：`artifact_id`
- 字段：
  - `artifact_id` (string, unique)
  - `inference_id` (string, index)
  - `artifact_type` (string: gradcam/explanation/thumbnail)
  - `artifact_uri` (string)
  - `created_at` (datetime)

### 2.6 `security_access_logs`
- 用途：记录接口鉴权与限流事件（审计与风控）
- 主键：`log_id`
- 字段：
  - `log_id` (string, unique)
  - `endpoint` (string)
  - `client_ip` (string)
  - `api_key_hash` (string)
  - `result` (string: pass/unauthorized/forbidden/rate_limited)
  - `created_at` (datetime)

## 3. 索引与性能建议

- `patient_uuid`、`study_id`、`created_at` 建立组合索引
- 推理结果查询优先按 `study_id + created_at` 排序
- 高并发读取场景建议按日期分区归档

## 4. 安全与合规

- 禁止存储直接身份信息（姓名、证件号）
- 所有导出接口需审计日志
- 最小权限访问，区分开发、标注、运维角色

## 5. 变更记录

- 2026-04-16：初始化数据库文档（v1.0）
- 2026-04-16：新增`inference_artifacts`集合，支持热图等可解释产物存储（v1.1）
- 2026-04-16：将热图类型明确为`gradcam`，对齐当前推理接口实现（v1.2）
- 2026-04-16：新增`security_access_logs`集合，支持鉴权与限流审计（v1.3）
