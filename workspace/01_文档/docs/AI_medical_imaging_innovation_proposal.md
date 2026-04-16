# 基于AI医学影像的大创完整方案（创新且可落地）

## 项目背景

医学影像检查需求快速增长，但优质放射科资源分布不均。基层与教学场景需要“高可用、可解释、可部署”的辅助系统，提升早筛效率并降低误漏诊风险。

## 创新点（四选二以上已满足）

- 多模态融合：联合影像、报告文本、结构化临床信息进行患者级判定。
- 可解释性增强：输出热图定位+证据文本，减少黑箱问题。
- 小样本学习：自监督预训练+主动学习，降低标注成本。
- 轻量化部署：蒸馏+量化，支持边缘设备快速推理。

## 技术路线

1. 数据治理与标签标准化  
2. 多模态基线建模与离线评测  
3. 小样本学习迭代  
4. 轻量化压缩与部署验证  
5. 场景化试点与结题交付

## 核心算法

- MMCA-Net：跨注意力多模态融合网络  
- Self-Supervised + Few-shot：自监督预训练与少样本微调  
- Explainability Constraint：解释一致性约束训练  
- Distill-Deploy：教师学生蒸馏与ONNX量化部署

## 应用场景

- 基层初筛与分诊排序
- 慢病随访与变化趋势提示
- 医学院教学实训案例平台
- 医工交叉科研验证平台

## 预期成果与社会价值

- 形成可运行的多模态医学影像原型系统
- 形成可解释输出与轻量部署能力
- 支撑大创结题、竞赛、论文/技术报告
- 提升基层医疗可及性与医疗效率

## 详细实施文档索引

- 病种与数据源：`01_topic_scope_and_datasets.md`
- 数据治理与标注：`02_data_governance_and_labeling_spec.md`
- 模型与离线评测：`03_multimodal_baseline_and_offline_eval.md`
- 小样本与轻量化：`04_fewshot_and_lightweight_deployment.md`
- 结题交付包：`05_final_deliverables_package.md`
- 云数据库文档：`cloud_database_design.md`
- 云函数权限文档：`cloud_function_permissions.md`
- 问题修改总结：`problem_fix_summary.md`
