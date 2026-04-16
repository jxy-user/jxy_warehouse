import io
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.config.utils import load_config
from src.data.dataset import tokenize_text_to_ids
from src.models.mmca_net import MMCANet


class InferenceRequest(BaseModel):
    text: str
    struct_features: List[float]


app = FastAPI(title="MedFuse-X 中文推理接口", version="0.2.0")
cfg = load_config("src/config/default.yaml")
heatmap_dir = Path(cfg["infer"]["heatmap_dir"])
heatmap_dir.mkdir(parents=True, exist_ok=True)

model_cfg = dict(cfg["model"])
model_cfg["num_struct_features"] = cfg["data"]["num_struct_features"]
model_cfg["num_classes"] = cfg["data"]["num_classes"]
model = MMCANet(model_cfg)
ckpt = Path(cfg["infer"]["checkpoint_path"])
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
labels = cfg["infer"]["class_names"]


def _prepare_image_tensor(content: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(io.BytesIO(content)).convert("L")
    image = image.resize((image_size, image_size))
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image_size, image_size) / 255.0
    return arr.unsqueeze(0).unsqueeze(0)


def _save_heatmap_placeholder(image_tensor: torch.Tensor, file_stem: str) -> str:
    # 轻量占位热图: 直接复用灰度图，后续可替换为Grad-CAM。
    arr = (image_tensor.squeeze().clamp(0, 1).numpy() * 255).astype("uint8")
    heatmap = Image.fromarray(arr, mode="L")
    out_path = heatmap_dir / f"{file_stem}_heatmap.png"
    heatmap.save(out_path)
    return str(out_path).replace("\\", "/")


@app.get("/health")
def health() -> dict:
    return {"状态": "正常", "服务": "MedFuse-X"}


@app.post("/infer")
def infer(req: InferenceRequest) -> dict:
    text_ids = tokenize_text_to_ids(req.text, cfg["model"]["text_max_len"], cfg["model"]["text_vocab_size"])
    text_ids = torch.tensor([text_ids], dtype=torch.long)

    struct = torch.tensor([req.struct_features], dtype=torch.float32)
    if struct.shape[1] != cfg["data"]["num_struct_features"]:
        return {"错误": f"struct_features长度必须为 {cfg['data']['num_struct_features']}"}

    # JSON接口兜底使用空白图，适合联调。
    image = torch.zeros((1, 1, cfg["data"]["image_size"], cfg["data"]["image_size"]), dtype=torch.float32)

    with torch.no_grad():
        logits = model(image, text_ids, struct)
        probs = torch.sigmoid(logits).squeeze(0).tolist()
    return {"风险分数": dict(zip(labels, probs))}


@app.post("/infer_with_image")
async def infer_with_image(
    image_file: UploadFile = File(..., description="胸片图像文件"),
    text: str = Form(default="", description="报告文本或主诉"),
    struct_features: str = Form(..., description="结构化特征，逗号分隔"),
) -> dict:
    parts = [x.strip() for x in struct_features.split(",") if x.strip()]
    if len(parts) != cfg["data"]["num_struct_features"]:
        return {"错误": f"struct_features长度必须为 {cfg['data']['num_struct_features']}"}

    struct = torch.tensor([[float(x) for x in parts]], dtype=torch.float32)
    text_ids = tokenize_text_to_ids(text, cfg["model"]["text_max_len"], cfg["model"]["text_vocab_size"])
    text_ids = torch.tensor([text_ids], dtype=torch.long)

    content = await image_file.read()
    image = _prepare_image_tensor(content, cfg["data"]["image_size"])
    heatmap_path = _save_heatmap_placeholder(image, Path(image_file.filename).stem or "sample")

    with torch.no_grad():
        logits = model(image, text_ids, struct)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    return {
        "文件名": image_file.filename,
        "风险分数": dict(zip(labels, probs)),
        "热图路径": heatmap_path,
        "说明": "当前为占位热图，后续可替换为Grad-CAM结果。",
    }
