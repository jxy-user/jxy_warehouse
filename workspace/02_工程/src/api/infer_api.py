import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.config.utils import load_module_overlay, load_resolved_config
from src.api.security import require_auth_and_rate_limit
from src.data.dataset import tokenize_text_to_ids
from src.models.mmca_net import MMCANet


class InferenceRequest(BaseModel):
    text: str
    struct_features: List[float]


def _build_app_title(main_cfg: dict) -> str:
    mod = main_cfg.get("metadata", {}).get("display_name")
    if mod:
        return f"灵枢AI — {mod}"
    return "灵枢AI 多模态医学影像推理服务"


BASE_CFG_PATH = "src/config/default.yaml"
cfg = load_resolved_config(BASE_CFG_PATH)
# 坐姿占位：与 active_module 无关，始终可读固定模块配置
posture_cfg = load_module_overlay(BASE_CFG_PATH, "posture_stub")

PIPELINE = cfg.get("pipeline", "mmc_net")

app = FastAPI(
    title=_build_app_title(cfg),
    version="0.5.0",
    description="active_module 切换影像任务；/posture/* 为坐姿占位接口（见 modules/posture_stub.yaml）",
)
heatmap_dir = Path(cfg["infer"]["heatmap_dir"])
heatmap_dir.mkdir(parents=True, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=str(heatmap_dir.resolve())), name="heatmaps")
cors_cfg = cfg.get("cors", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_cfg.get("allow_origins", ["*"]),
    allow_credentials=cors_cfg.get("allow_credentials", True),
    allow_methods=cors_cfg.get("allow_methods", ["*"]),
    allow_headers=cors_cfg.get("allow_headers", ["*"]),
)

model: Optional[MMCANet] = None
if PIPELINE == "mmc_net":
    model_cfg = dict(cfg["model"])
    model_cfg["num_struct_features"] = cfg["data"]["num_struct_features"]
    model_cfg["num_classes"] = cfg["data"]["num_classes"]
    model = MMCANet(model_cfg)
    ckpt = Path(cfg["infer"]["checkpoint_path"])
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

labels = cfg["infer"]["class_names"]
posture_labels = posture_cfg["infer"]["class_names"]

security_enabled = cfg.get("security", {}).get("enabled", True)
rate_limit_per_minute = cfg.get("security", {}).get("rate_limit_per_minute", 60)
public_health = cfg.get("security", {}).get("public_health", True)

UPLOAD_DESC = cfg.get("api", {}).get("upload_image_description", "医学影像文件（PNG/JPG）")
BONE_KEYWORDS = {
    "骨", "骨折", "骨裂", "骨龄", "关节", "骺线", "生长板", "脊柱", "侧弯", "x光", "dr", "片子", "外伤", "肿痛",
}
POSTURE_KEYWORDS = {
    "坐姿", "低头", "驼背", "歪", "歪斜", "久坐", "肩颈", "颈椎", "写作业", "屏幕", "视距", "坐太久", "姿势",
}
RED_FLAG_KEYWORDS = {
    "剧痛", "持续加重", "夜间痛", "麻木", "无力", "无法站立", "无法行走", "活动受限", "外伤后畸形", "发热",
}
TCM_DISCLAIMER = "本系统仅提供中医健康管理辅助建议，不构成医疗诊断或处方依据，不能替代执业医师面诊。"
DEFAULT_TCM_ADVICE = [
    "起居调养：避免久坐久站与受寒，保持规律作息，注意局部保暖。",
    "饮食建议：清淡均衡，少辛辣油腻，适当补充蛋白与富钙食物。",
    "运动建议：在疼痛可耐受范围内进行轻柔拉伸与关节活动，避免剧烈负重。",
]
TCM_TEMPLATE_CONFIG = {
    "tcm_lowback": {
        "keywords": ["腰疼", "腰痛", "腰酸", "久坐腰酸", "闪腰", "腰背痛"],
        "category": "腰部不适倾向",
        "syndrome": "气血不畅、筋脉失和倾向",
        "advice": [
            "可在中医师辨证后选用推拿或针灸进行调理，缓解腰部紧张与不适。",
            "每30-40分钟起身活动2-3分钟，避免久坐与突然负重。",
            "疼痛缓解后逐步进行腰背轻柔牵伸与核心稳定训练。",
        ],
    },
    "tcm_knee": {
        "keywords": ["膝盖", "膝关节", "膝痛", "上下楼痛", "蹲起痛"],
        "category": "膝关节不适倾向",
        "syndrome": "肝肾不足、筋骨失养倾向",
        "advice": [
            "以减负与温养为主，避免频繁上下楼和深蹲动作。",
            "可在中医师指导下采用针灸或推拿，改善膝关节不适。",
            "进行低冲击康复练习，循序渐进恢复关节活动度。",
        ],
    },
    "tcm_neck": {
        "keywords": ["颈椎", "颈部", "脖子", "落枕", "颈项僵硬"],
        "category": "颈项不适倾向",
        "syndrome": "气血不畅、经络不舒倾向",
        "advice": [
            "可在中医师辨证下配合针灸或推拿缓解颈项僵硬。",
            "减少长时间低头，保持屏幕与视线平齐，避免颈部受寒。",
            "每日进行颈肩轻柔拉伸，改善局部气血运行。",
        ],
    },
    "tcm_shoulder_neck": {
        "keywords": ["肩颈", "肩膀僵硬", "肩背紧", "肩酸", "斜方肌酸痛"],
        "category": "肩颈紧张倾向",
        "syndrome": "气滞血瘀、筋脉拘急倾向",
        "advice": [
            "可进行温和推拿或艾灸，帮助缓解肩颈肌群紧张。",
            "避免久坐耸肩与含胸姿势，注意肩背保暖。",
            "配合肩胛稳定训练与胸椎伸展，改善体态负担。",
        ],
    },
    "tcm_posture": {
        "keywords": ["驼背", "体态", "含胸", "圆肩", "坐姿", "姿势不好"],
        "category": "体态失衡倾向",
        "syndrome": "筋脉失和倾向",
        "advice": [
            "建议以姿势矫正为主，保持脊柱中立位，减少含胸低头。",
            "每30-40分钟起身活动，进行颈肩腰背放松训练。",
            "可配合八段锦等中医导引练习，逐步改善体态。",
        ],
    },
}


def _auth_dep(request: Request) -> str:
    return require_auth_and_rate_limit(
        request=request,
        enabled=security_enabled,
        rate_limit_per_minute=rate_limit_per_minute,
    )


def _stub_infer_response(kind: str) -> dict:
    zero_scores = {name: 0.0 for name in labels}
    return {
        "风险分数": zero_scores,
        "说明": "坐姿占位模块（pipeline=posture_stub）：关键点估计与规则引擎未接入，分数固定为占位。请改用 bone/chest_mvp 进行影像推理，或使用 /posture/analyze。",
        "pipeline": PIPELINE,
        "占位类型": kind,
        "辨识结果": {
            "类别倾向": "筋骨失养倾向（占位）",
            "中医证候倾向": "待结合面诊进一步辨证",
            "风险等级": "待评估",
        },
        "调养建议": list(DEFAULT_TCM_ADVICE),
        "就医提醒": "如症状持续加重或出现剧痛、麻木、活动受限，请及时线下就医。",
        "免责声明": TCM_DISCLAIMER,
        "红旗预警": False,
    }


def _route_query(text: str, has_image: bool) -> Tuple[str, str, float]:
    content = (text or "").lower()
    bone_hits = [k for k in BONE_KEYWORDS if k in content]
    posture_hits = [k for k in POSTURE_KEYWORDS if k in content]
    bone_score = len(bone_hits)
    posture_score = len(posture_hits)

    # 若上传影像，优先走影像模块
    if has_image:
        bone_score += 2

    if posture_score > bone_score:
        total = posture_score + bone_score + 1e-6
        conf = min(0.95, max(0.55, posture_score / total + 0.3))
        reason = "命中坐姿关键词: " + (",".join(posture_hits[:3]) if posture_hits else "无")
        return "posture_stub", reason, round(conf, 2)

    total = posture_score + bone_score + 1e-6
    conf = min(0.95, max(0.55, bone_score / total + 0.3))
    reason = "命中骨骼关键词: " + (",".join(bone_hits[:3]) if bone_hits else "默认骨骼路由")
    return "bone", reason, round(conf, 2)


def _normalize_route_mode(route_mode: str) -> str:
    mapping = {
        "auto": "auto",
        "force_bone": "force_bone",
        "force_posture": "force_posture",
        # 兼容前端可能传入的别名
        "bone_first": "force_bone",
        "posture_first": "force_posture",
    }
    return mapping.get((route_mode or "").strip().lower(), "auto")


def _prepare_image_tensor(content: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(io.BytesIO(content)).convert("L")
    image = image.resize((image_size, image_size))
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image_size, image_size) / 255.0
    return arr.unsqueeze(0).unsqueeze(0)


def _contains_red_flag(text: str) -> bool:
    content = (text or "").lower()
    return any(k in content for k in RED_FLAG_KEYWORDS)


def _select_tcm_template(text: str) -> str:
    content = (text or "").lower()
    priority = ["tcm_lowback", "tcm_knee", "tcm_neck", "tcm_shoulder_neck", "tcm_posture"]
    for key in priority:
        cfg_item = TCM_TEMPLATE_CONFIG[key]
        if any(k in content for k in cfg_item["keywords"]):
            return key
    return "tcm_general"


def _build_tcm_guidance(top_label: str, score: float, text: str) -> dict:
    red_flag = _contains_red_flag(text)
    template_id = _select_tcm_template(text)
    level = "较高" if score >= 0.5 else "较低"
    cfg_item = TCM_TEMPLATE_CONFIG.get(template_id, {})
    syndrome = cfg_item.get("syndrome", "肝肾不足、筋骨失养倾向")
    category = cfg_item.get("category", top_label)
    advice = list(cfg_item.get("advice", DEFAULT_TCM_ADVICE))
    if score >= 0.5:
        advice.append("建议尽快至正规医疗机构复核影像与体征，再由医生制定个体化方案。")

    remind = "建议结合原片、症状变化和线下体格检查综合判断。"
    if red_flag:
        remind = "检测到红旗症状关键词，请立即线下就医，不建议仅依赖线上建议。"

    return {
        "辨识结果": {
            "类别倾向": category,
            "中医证候倾向": syndrome,
            "风险等级": level,
        },
        "调养建议": advice,
        "就医提醒": remind,
        "免责声明": TCM_DISCLAIMER,
        "红旗预警": red_flag,
        "模板ID": template_id,
    }


def _run_mmc_json_infer(text: str, struct_values: List[float]) -> dict:
    assert model is not None
    text_ids = tokenize_text_to_ids(text, cfg["model"]["text_max_len"], cfg["model"]["text_vocab_size"])
    text_ids = torch.tensor([text_ids], dtype=torch.long)
    struct = torch.tensor([struct_values], dtype=torch.float32)
    image = torch.zeros((1, 1, cfg["data"]["image_size"], cfg["data"]["image_size"]), dtype=torch.float32)
    with torch.no_grad():
        logits = model(image, text_ids, struct)
        probs = torch.sigmoid(logits).squeeze(0).tolist()
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    out = {"风险分数": dict(zip(labels, probs))}
    out.update(_build_tcm_guidance(top_label=top_label, score=float(probs[top_idx]), text=text))
    return out


def _run_mmc_image_infer(content: bytes, filename: str, text: str, struct_values: List[float]) -> dict:
    assert model is not None
    struct = torch.tensor([struct_values], dtype=torch.float32)
    text_ids = tokenize_text_to_ids(text, cfg["model"]["text_max_len"], cfg["model"]["text_vocab_size"])
    text_ids = torch.tensor([text_ids], dtype=torch.long)

    image = _prepare_image_tensor(content, cfg["data"]["image_size"])
    logits = model(image, text_ids, struct)
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()
    target_idx = int(np.argmax(probs))
    cam = _generate_gradcam(image, text_ids, struct, target_idx)
    heatmap_path = _save_gradcam_heatmap(image, cam, Path(filename).stem or "sample")

    suffix = cfg.get("api", {}).get("infer_description_suffix", "")
    tip = f"已生成Grad-CAM热图，当前倾向类别: {labels[target_idx]}"
    if suffix:
        tip += f"（{suffix}）"
    out = {
        "文件名": filename,
        "风险分数": dict(zip(labels, probs)),
        "热图路径": heatmap_path,
        "说明": tip,
    }
    out.update(_build_tcm_guidance(top_label=labels[target_idx], score=float(probs[target_idx]), text=text))
    return out


def _generate_gradcam(
    image: torch.Tensor,
    text_ids: torch.Tensor,
    struct: torch.Tensor,
    target_idx: int,
) -> np.ndarray:
    assert model is not None
    activations = []
    gradients = []

    def _forward_hook(_module, _inputs, output):
        activations.append(output)

    def _backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.image_encoder.backbone[2]
    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(image, text_ids, struct)
    score = logits[0, target_idx]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(cfg["data"]["image_size"], cfg["data"]["image_size"]), mode="bilinear")
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _save_gradcam_heatmap(image_tensor: torch.Tensor, cam: np.ndarray, file_stem: str) -> str:
    gray = (image_tensor.squeeze().detach().cpu().numpy() * 255).astype("uint8")
    red = (cam * 255).astype("uint8")
    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    heat_color = np.stack([red, green, blue], axis=-1)
    base_rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = (0.6 * base_rgb + 0.4 * heat_color).astype("uint8")
    heatmap = Image.fromarray(overlay, mode="RGB")
    filename = f"{file_stem}_gradcam.png"
    out_path = heatmap_dir / filename
    heatmap.save(out_path)
    # 返回可被浏览器直接访问的静态 URL 路径，而不是本地磁盘路径。
    return f"/heatmaps/{filename}"


@app.get("/health")
def health(request: Request) -> dict:
    if not public_health:
        require_auth_and_rate_limit(
            request=request,
            enabled=security_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
        )
    out = {
        "状态": "正常",
        "服务": "灵枢AI",
        "pipeline": PIPELINE,
        "任务模块": cfg.get("active_module"),
        "任务标识": cfg["data"].get("task"),
        "模块名称": cfg.get("metadata", {}).get("display_name"),
        "类别": labels,
        "坐姿占位路由": "/posture/info（无需切换 active_module）",
    }
    return out


@app.post("/infer")
def infer(
    req: InferenceRequest,
    _api_key: str = Depends(_auth_dep),
) -> dict:
    if PIPELINE != "mmc_net":
        return _stub_infer_response("infer_json")

    if len(req.struct_features) != cfg["data"]["num_struct_features"]:
        return {"错误": f"struct_features长度必须为 {cfg['data']['num_struct_features']}"}
    return _run_mmc_json_infer(req.text, req.struct_features)


@app.post("/infer_with_image")
async def infer_with_image(
    image_file: UploadFile = File(..., description=UPLOAD_DESC),
    text: str = Form(default="", description="报告文本或主诉"),
    struct_features: str = Form(..., description="结构化特征，逗号分隔"),
    _api_key: str = Depends(_auth_dep),
) -> dict:
    if PIPELINE != "mmc_net":
        await image_file.read()
        return _stub_infer_response("infer_image")

    parts = [x.strip() for x in struct_features.split(",") if x.strip()]
    if len(parts) != cfg["data"]["num_struct_features"]:
        return {"错误": f"struct_features长度必须为 {cfg['data']['num_struct_features']}"}
    struct_values = [float(x) for x in parts]
    content = await image_file.read()
    return _run_mmc_image_infer(content, image_file.filename, text, struct_values)


@app.get("/posture/info")
def posture_info(request: Request, _api_key: str = Depends(_auth_dep)) -> dict:
    """坐姿占位模块说明（配置来自 modules/posture_stub.yaml，与当前 active_module 独立）。"""
    return {
        "模块": "posture_stub",
        "pipeline": posture_cfg.get("pipeline"),
        "任务标识": posture_cfg["data"].get("task"),
        "模块名称": posture_cfg.get("metadata", {}).get("display_name"),
        "行为维度": posture_labels,
        "stub": posture_cfg.get("stub", {}),
        "说明": "POST /posture/analyze 上传单帧图像将返回占位分数（全0）；算法接入后替换为关键点+规则引擎。",
    }


@app.post("/posture/analyze")
async def posture_analyze(
    image_file: Optional[UploadFile] = File(None),
    _api_key: str = Depends(_auth_dep),
) -> dict:
    """坐姿分析占位：接收可选单帧图像，返回固定占位结构。"""
    if image_file is not None:
        await image_file.read()
    return {
        "模块": "posture_stub",
        "行为风险分数": {name: 0.0 for name in posture_labels},
        "说明": "占位响应：未运行姿态估计。后续将接入关键点检测与久坐/低头等规则评分。",
        "pipeline": posture_cfg.get("pipeline"),
        "辨识结果": {
            "类别倾向": "姿势失衡倾向（占位）",
            "中医证候倾向": "筋脉失和倾向",
            "风险等级": "待评估",
        },
        "调养建议": [
            "保持脊柱中立位，减少含胸低头姿势，学习时注意桌椅高度匹配。",
            "每30-40分钟起身活动2-3分钟，避免久坐不动。",
            "可进行八段锦或轻柔牵伸练习，循序渐进，不可急于求成。",
        ],
        "就医提醒": "本结果为辅助建议，若症状持续或加重，请及时线下就医。",
        "免责声明": TCM_DISCLAIMER,
        "红旗预警": False,
    }


@app.post("/route_infer")
async def route_infer(
    text: str = Form(default="", description="用户问题文本"),
    struct_features: str = Form(default="", description="结构化特征，逗号分隔，可空"),
    route_mode: str = Form(default="auto", description="路由模式：auto/force_bone/force_posture"),
    image_file: Optional[UploadFile] = File(None, description="可选影像文件"),
    _api_key: str = Depends(_auth_dep),
) -> dict:
    route_mode_norm = _normalize_route_mode(route_mode)
    has_image = image_file is not None
    if route_mode_norm == "force_bone":
        route_module, reason, confidence = "bone", "前端强制骨骼模块", 1.0
    elif route_mode_norm == "force_posture":
        route_module, reason, confidence = "posture_stub", "前端强制坐姿模块", 1.0
    else:
        route_module, reason, confidence = _route_query(text, has_image)

    parts = [x.strip() for x in struct_features.split(",") if x.strip()]
    if parts and len(parts) != cfg["data"]["num_struct_features"]:
        return {"错误": f"struct_features长度必须为 {cfg['data']['num_struct_features']}"}
    if parts:
        struct_values = [float(x) for x in parts]
    else:
        struct_values = [0.0] * cfg["data"]["num_struct_features"]

    if route_module == "posture_stub":
        if image_file is not None:
            await image_file.read()
        result = {
            "模块": "posture_stub",
            "行为风险分数": {name: 0.0 for name in posture_labels},
            "说明": "自动路由到坐姿占位模块：当前返回占位分数（全0）。",
            "pipeline": posture_cfg.get("pipeline"),
            "辨识结果": {
                "类别倾向": "姿势失衡倾向（占位）",
                "中医证候倾向": "筋脉失和倾向",
                "风险等级": "待评估",
            },
            "调养建议": [
                "调整坐姿：头颈中正、双肩放松，屏幕与视线平齐。",
                "每30-40分钟起身活动，配合颈肩腰背轻柔拉伸。",
                "日常调养宜循序渐进，避免长期低头与单侧受力。",
            ],
            "就医提醒": "如出现持续疼痛、麻木或活动受限，请及时线下就医。",
            "免责声明": TCM_DISCLAIMER,
            "红旗预警": _contains_red_flag(text),
        }
        return {
            "route_module": route_module,
            "route_mode_used": route_mode_norm,
            "route_reason": reason,
            "route_confidence": confidence,
            "result": result,
        }

    # 路由到骨骼/影像模块，要求当前 active_module 是 mmc_net 管线
    if PIPELINE != "mmc_net" or model is None:
        return {
            "route_module": route_module,
            "route_mode_used": route_mode_norm,
            "route_reason": reason,
            "route_confidence": confidence,
            "错误": "当前 active_module 不是影像推理模块（mmc_net），无法执行骨骼路由。请切换到 bone 或 chest_mvp。",
        }

    if image_file is None:
        result = _run_mmc_json_infer(text, struct_values)
    else:
        content = await image_file.read()
        result = _run_mmc_image_infer(content, image_file.filename, text, struct_values)
    return {
        "route_module": route_module,
        "route_mode_used": route_mode_norm,
        "route_reason": reason,
        "route_confidence": confidence,
        "result": result,
    }
