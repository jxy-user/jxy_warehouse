import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config.utils import load_resolved_config
from src.data.dataset import CSVMultiModalDataset, MockMultiModalDataset, split_indices
from src.models.mmca_net import MMCANet


def train(cfg_path: str) -> None:
    cfg = load_resolved_config(cfg_path)
    if cfg.get("pipeline", "mmc_net") != "mmc_net":
        print("当前 active_module 为坐姿占位（pipeline=posture_stub），不支持 MMCANet 训练。请改用 bone 或 chest_mvp。")
        sys.exit(1)
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
        print(f"使用真实CSV数据集训练。（模块={cfg.get('active_module')}）")
    else:
        dataset = MockMultiModalDataset(
            num_samples=cfg["data"]["num_samples"],
            image_size=cfg["data"]["image_size"],
            text_max_len=cfg["model"]["text_max_len"],
            num_struct_features=cfg["data"]["num_struct_features"],
            num_classes=cfg["data"]["num_classes"],
            seed=cfg["seed"],
        )
        print("使用Mock数据集训练。")
    train_idx, val_idx = split_indices(len(dataset), cfg["data"]["train_split"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["train"]["batch_size"])

    model_cfg = dict(cfg["model"])
    model_cfg["num_struct_features"] = cfg["data"]["num_struct_features"]
    model_cfg["num_classes"] = cfg["data"]["num_classes"]
    model = MMCANet(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            struct_features = batch["struct_features"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image, text_ids, struct_features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["image"].to(device),
                    batch["text_ids"].to(device),
                    batch["struct_features"].to(device),
                )
                loss = criterion(logits, batch["label"].to(device))
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} "
            f"train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_loss={val_loss / max(len(val_loader), 1):.4f}"
        )

    save_path = Path(cfg["train"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    args = parser.parse_args()
    train(args.config)
