# Training Artifacts

Model checkpoints, metrics, demo images, and audit outputs are generated under `outputs/` and are not committed to this code repository.

Run the full experiment workflow:

```powershell
python scripts\run_full_real_experiments.py --profile full
```

Run the shorter real-data acceptance workflow:

```powershell
python scripts\run_full_real_experiments.py --profile acceptance
```

Recheck the generated checkpoints:

```powershell
python scripts\audit_model_quality.py
```

The GUI demo loads checkpoints from `outputs\full_real` when they exist:

```powershell
python -m dl_labs.demo_gui
```

Expected generated artifact families:

```text
outputs/full_real/exp1/checkpoints/*.pt
outputs/full_real/exp2_seg/checkpoints/unet_msrc.pt
outputs/full_real/exp2_sr/checkpoints/srcnn_bsds500.pt
outputs/full_real/exp3_weather/checkpoints/weather_*.pt
outputs/full_real/exp3_shakespeare/checkpoints/shakespeare_lstm.pt
```
