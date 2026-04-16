import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config.utils import load_config
from src.data.dataset import CSVMultiModalDataset, MockMultiModalDataset
from src.models.mmca_net import MMCANet


def evaluate(cfg_path: str, checkpoint: str) -> None:
    cfg = load_config(cfg_path)
    device = torch.device(cfg["device"])

    if cfg["data"]["mode"] == "csv":
        dataset = CSVMultiModalDataset(
            csv_path=cfg["data"]["csv_path"],
            image_root=cfg["data"]["image_root"],
            image_size=cfg["data"]["image_size"],
            text_max_len=cfg["model"]["text_max_len"],
            num_struct_features=cfg["data"]["num_struct_features"],
            num_classes=cfg["data"]["num_classes"],
            text_vocab_size=cfg["model"]["text_vocab_size"],
        )
        print("使用真实CSV数据集评测。")
    else:
        dataset = MockMultiModalDataset(
            num_samples=cfg["data"]["num_samples"],
            image_size=cfg["data"]["image_size"],
            text_max_len=cfg["model"]["text_max_len"],
            num_struct_features=cfg["data"]["num_struct_features"],
            num_classes=cfg["data"]["num_classes"],
            seed=cfg["seed"] + 1,
        )
        print("使用Mock数据集评测。")
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"])

    model_cfg = dict(cfg["model"])
    model_cfg["num_struct_features"] = cfg["data"]["num_struct_features"]
    model_cfg["num_classes"] = cfg["data"]["num_classes"]
    model = MMCANet(model_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["image"].to(device),
                batch["text_ids"].to(device),
                batch["struct_features"].to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(batch["label"].numpy())
            ps.append(probs)

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    y_pred = (y_prob > 0.5).astype(np.int32)

    auc = roc_auc_score(y_true, y_prob, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"评测AUC(macro): {auc:.4f}")
    print(f"评测F1(macro): {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/mmca_net.pt")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
