from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.utils import AverageMeter, ensure_dir, get_device, save_csv, save_json, set_seed


class WindowedArrayDataset(Dataset):
    def __init__(self, data: np.ndarray, input_width: int, horizon: int, target_index: int) -> None:
        self.data = data.astype(np.float32)
        self.input_width = input_width
        self.horizon = horizon
        self.target_index = target_index
        self.total = input_width + horizon

    def __len__(self) -> int:
        return max(0, len(self.data) - self.total + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.data[index : index + self.total]
        x = window[: self.input_width]
        y = window[self.input_width :, self.target_index]
        return torch.from_numpy(x), torch.from_numpy(y)


class SyntheticWeatherDataset(Dataset):
    def __init__(self, size: int, input_width: int = 48, horizon: int = 24, features: int = 8, offset: int = 0) -> None:
        self.size = size
        self.input_width = input_width
        self.horizon = horizon
        self.features = features
        self.offset = offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = np.arange(self.input_width + self.horizon, dtype=np.float32) + self.offset + index
        temp = 10 + 8 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / (24 * 30))
        pressure = 1000 + 4 * np.cos(2 * np.pi * t / (24 * 7))
        rho = 1.2 + 0.05 * np.sin(2 * np.pi * t / 48)
        wx = np.sin(2 * np.pi * t / 18)
        wy = np.cos(2 * np.pi * t / 18)
        day_sin = np.sin(2 * np.pi * t / 24)
        day_cos = np.cos(2 * np.pi * t / 24)
        year_sin = np.sin(2 * np.pi * t / (24 * 365))
        arr = np.stack([temp, pressure, rho, wx, wy, day_sin, day_cos, year_sin], axis=-1).astype(np.float32)
        mean = arr[: self.input_width].mean(axis=0, keepdims=True)
        std = arr[: self.input_width].std(axis=0, keepdims=True) + 1e-6
        arr = (arr - mean) / std
        return torch.from_numpy(arr[: self.input_width]), torch.from_numpy(arr[self.input_width :, 0])


def preprocess_jena(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    df = pd.read_csv(csv_path)
    df = df[5::6].copy()
    date_time = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")

    for column in ["wv (m/s)", "max. wv (m/s)"]:
        bad = df[column] == -9999.0
        df.loc[bad, column] = 0.0

    wv = df.pop("wv (m/s)")
    max_wv = df.pop("max. wv (m/s)")
    wd_rad = df.pop("wd (deg)") * np.pi / 180
    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)
    df["max Wx"] = max_wv * np.cos(wd_rad)
    df["max Wy"] = max_wv * np.sin(wd_rad)

    timestamp_s = date_time.map(pd.Timestamp.timestamp).to_numpy()
    day = 24 * 60 * 60
    year = 365.2425 * day
    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    columns = list(df.columns)
    target_index = columns.index("T (degC)")
    n = len(df)
    train_df = df.iloc[: int(n * 0.7)]
    val_df = df.iloc[int(n * 0.7) : int(n * 0.9)]
    test_df = df.iloc[int(n * 0.9) :]
    mean = train_df.mean()
    std = train_df.std().replace(0, 1.0)
    return (
        ((train_df - mean) / std).to_numpy(np.float32),
        ((val_df - mean) / std).to_numpy(np.float32),
        ((test_df - mean) / std).to_numpy(np.float32),
        columns,
        target_index,
    )


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        kind: str,
        input_dim: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kind = kind.lower()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def train_weather_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_mse = float("inf")
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_mse, val_mae = evaluate_weather_model(model, val_loader, device)
        if val_mse < best_mse:
            best_mse = val_mse
            best_mae = val_mae
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == args.epochs or epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                {
                    "model": getattr(args, "_current_model", "weather"),
                    "epoch": epoch,
                    "val_mse": round(val_mse, 6),
                    "val_mae": round(val_mae, 6),
                    "best_epoch": best_epoch,
                    "best_val_mse": round(best_mse, 6),
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "val_mse": best_mse, "val_mae": best_mae}


@torch.no_grad()
def evaluate_weather_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.MSELoss()
    mse_meter = AverageMeter()
    mae_meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse_meter.update(criterion(pred, y).item(), x.size(0))
        mae_meter.update(torch.mean(torch.abs(pred - y)).item(), x.size(0))
    return mse_meter.avg, mae_meter.avg


@torch.no_grad()
def save_weather_demo(model: nn.Module, loader: DataLoader, device: torch.device, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[float, np.ndarray, np.ndarray]] = []
    model.eval()
    for x, y in loader:
        pred = model(x.to(device)).detach().cpu()
        mse = torch.mean((pred - y) ** 2, dim=1)
        for score, pred_item, true_item in zip(mse.tolist(), pred.numpy(), y.numpy()):
            cases.append((float(score), pred_item, true_item))
    if not cases:
        return
    cases.sort(key=lambda item: item[0])
    _, pred, true = cases[len(cases) // 2]
    plt.figure(figsize=(8, 3.4), dpi=140)
    plt.plot(true, label="true", linewidth=2)
    plt.plot(pred, label="prediction", linewidth=2)
    plt.title("Next-week temperature forecast case")
    plt.xlabel("hours ahead")
    plt.ylabel("normalized T (degC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_weather(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    if args.dataset == "synthetic":
        train_ds = SyntheticWeatherDataset(args.synthetic_train_size, args.input_width, args.horizon, offset=0)
        val_ds = SyntheticWeatherDataset(args.synthetic_val_size, args.input_width, args.horizon, offset=1000)
        test_ds = SyntheticWeatherDataset(args.synthetic_val_size, args.input_width, args.horizon, offset=2000)
        input_dim = train_ds.features
    else:
        train, val, test, columns, target_index = preprocess_jena(args.csv_path)
        train_ds = WindowedArrayDataset(train, args.input_width, args.horizon, target_index)
        val_ds = WindowedArrayDataset(val, args.input_width, args.horizon, target_index)
        test_ds = WindowedArrayDataset(test, args.input_width, args.horizon, target_index)
        input_dim = len(columns)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    eval_ds = val_ds if args.eval_split == "val" else test_ds
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    rows: list[dict] = []
    for kind in args.models:
        model = SequenceRegressor(kind, input_dim, args.horizon, args.hidden_size, args.num_layers, args.dropout).to(device)
        args._current_model = kind
        train_info = train_weather_model(model, train_loader, val_loader, args, device)
        mse, mae = evaluate_weather_model(model, eval_loader, device)
        checkpoint_path = output_dir / "checkpoints" / f"weather_{kind}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "task": "weather",
                "model": kind,
                "input_dim": input_dim,
                "input_width": args.input_width,
                "horizon": args.horizon,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "best_epoch": train_info["best_epoch"],
                "val_mse": train_info["val_mse"],
                "val_mae": train_info["val_mae"],
                "eval_split": args.eval_split,
            },
            checkpoint_path,
        )
        demo_path = output_dir / "demo_cases" / f"weather_{kind}.png"
        save_weather_demo(model, eval_loader, device, demo_path)
        row = {
            "task": "weather",
            "dataset": args.dataset,
            "model": kind,
            "input_width_hours": args.input_width,
            "forecast_horizon_hours": args.horizon,
            "epochs": args.epochs,
            "best_epoch": train_info["best_epoch"],
            "val_mse": round(train_info["val_mse"], 6),
            "val_mae": round(train_info["val_mae"], 6),
            "eval_split": args.eval_split,
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "checkpoint": str(checkpoint_path),
            "demo_image": str(demo_path),
        }
        print(row)
        rows.append(row)
    save_csv(rows, output_dir / "weather_results.csv")
    save_json({"results": rows}, output_dir / "weather_results.json")
    return rows


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 80) -> None:
        if len(text) < seq_len + 2:
            repeats = (seq_len + 2) // max(1, len(text)) + 2
            text = text * repeats
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.encoded = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded) - self.seq_len - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded[index : index + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, kind: str = "lstm", embed_dim: int = 64, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind.lower()]
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = rnn_cls(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.rnn(self.embed(x), hidden)
        return self.head(out), hidden


def load_shakespeare_text(args: argparse.Namespace) -> str:
    path = Path(args.text_file)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if args.dataset == "synthetic":
        return (
            "ROMEO: But soft, what light through yonder window breaks?\n"
            "JULIET: It is the east, and Juliet is the sun.\n"
            "HAMLET: To be, or not to be, that is the question.\n"
            "KING: Speak on, my lord, and let the court attend.\n"
        )
    raise FileNotFoundError(
        f"Missing Shakespeare corpus: {path}. Put the text file there or run with --dataset synthetic for a smoke test."
    )


def train_char_model(model: CharRNN, dataset: CharDataset, args: argparse.Namespace, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        meter = AverageMeter()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            meter.update(loss.item(), x.size(0))
        last_loss = meter.avg
    return last_loss


@torch.no_grad()
def generate_text(model: CharRNN, dataset: CharDataset, seed: str, length: int, temperature: float, device: torch.device) -> str:
    model.eval()
    hidden = None
    output = seed
    idx = torch.tensor([[dataset.stoi.get(seed[0], 0)]], dtype=torch.long, device=device)
    for ch in seed:
        idx = torch.tensor([[dataset.stoi.get(ch, 0)]], dtype=torch.long, device=device)
        _, hidden = model(idx, hidden)
    for _ in range(length):
        logits, hidden = model(idx, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        ch = dataset.itos[int(idx.item())]
        output += ch
    return output


def run_shakespeare(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    text = load_shakespeare_text(args)
    dataset = CharDataset(text, args.seq_len)
    model = CharRNN(len(dataset.chars), args.rnn_type, args.embed_dim, args.hidden_size, args.num_layers).to(device)
    loss = train_char_model(model, dataset, args, device)
    generated = generate_text(model, dataset, args.seed_text, args.generate_length, args.temperature, device)
    generated_path = output_dir / "shakespeare_generated.txt"
    generated_path.write_text(generated, encoding="utf-8")
    checkpoint_path = output_dir / "checkpoints" / "shakespeare_lstm.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": "shakespeare",
            "model": args.rnn_type,
            "vocab": dataset.chars,
            "seq_len": args.seq_len,
            "embed_dim": args.embed_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        },
        checkpoint_path,
    )
    row = {
        "task": "shakespeare",
        "dataset": args.dataset,
        "model": args.rnn_type,
        "epochs": args.epochs,
        "vocab_size": len(dataset.chars),
        "train_loss": round(loss, 6),
        "generated_path": str(generated_path),
        "checkpoint": str(checkpoint_path),
    }
    save_json(row, output_dir / "shakespeare_results.json")
    print(row)
    print(generated)
    return row


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--output-dir", default="outputs/exp3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: recurrent neural networks.")
    sub = parser.add_subparsers(dest="task", required=True)

    weather = sub.add_parser("weather")
    add_common(weather)
    weather.add_argument("--csv-path", default="datasets/jena_climate_2009_2016.csv")
    weather.add_argument("--models", nargs="+", default=["rnn", "gru", "lstm"])
    weather.add_argument("--input-width", type=int, default=168)
    weather.add_argument("--horizon", type=int, default=168)
    weather.add_argument("--hidden-size", type=int, default=64)
    weather.add_argument("--num-layers", type=int, default=1)
    weather.add_argument("--dropout", type=float, default=0.0)
    weather.add_argument("--eval-split", choices=["val", "test"], default="test")
    weather.add_argument("--log-every", type=int, default=5)
    weather.add_argument("--synthetic-train-size", type=int, default=256)
    weather.add_argument("--synthetic-val-size", type=int, default=64)

    shakes = sub.add_parser("shakespeare")
    add_common(shakes)
    shakes.add_argument("--text-file", default="datasets/shakespeare.txt")
    shakes.add_argument("--rnn-type", default="lstm", choices=["rnn", "gru", "lstm"])
    shakes.add_argument("--seq-len", type=int, default=80)
    shakes.add_argument("--embed-dim", type=int, default=64)
    shakes.add_argument("--hidden-size", type=int, default=128)
    shakes.add_argument("--num-layers", type=int, default=2)
    shakes.add_argument("--grad-clip", type=float, default=1.0)
    shakes.add_argument("--seed-text", default="ROMEO:")
    shakes.add_argument("--generate-length", type=int, default=400)
    shakes.add_argument("--temperature", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.task == "weather":
        run_weather(args)
    elif args.task == "shakespeare":
        run_shakespeare(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
