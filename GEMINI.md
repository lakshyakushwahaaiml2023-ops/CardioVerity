# Project Rules & Guidelines

## PROJECT
Web app for the "Cardiovascular Risk Visualization & Prediction" hackathon track. Predicts overall CAD and stenosis of LAD, LCX, RCA from the Z-Alizadeh Sani extension dataset (303 rows) and shows the result on an interactive 3D heart.

## STACK
- **Runtime & Environment**: Python 3.11
- **ML & Compute**: PyTorch (CUDA), XGBoost (`device="cuda"`), scikit-learn (strictly for preprocessing and metrics only)
- **Backend API**: FastAPI
- **Frontend**: React + Vite + TypeScript + React Three Fiber + @react-three/drei
- **Styling**: Plain CSS with CSS variables (no Tailwind, no UI kit)

## HARD RULES
1. **GPU Heavy Compute**: All heavy compute runs on GPU: XGBoost with `device="cuda"`, PyTorch models on CUDA, bootstrap/ensemble members trained on CUDA. Every training script calls a shared `utils/device.py` that logs `torch.cuda.get_device_name` and VRAM, and RAISES if CUDA is unavailable unless `--allow-cpu` is passed. Sklearn is used only for preprocessing and metrics.
2. **CPU Inference at Serve Time**: Inference at serve time must run on CPU, with no GPU required.
3. **Leakage Prevention**: Never use `LAD`, `LCX`, `RCA`, or `Cath` as input features. Log every feature dropped and why.
4. **Reproducibility**: Fixed seeds, `requirements.txt` with pinned versions, a Makefile or scripts for each step, config in YAML.
5. **No Hard-coded Numbers**: Every number shown in the UI or docs must come from a saved metrics file, never hard-coded.
6. **Clinical UI Tone**: Clinical decision-support software. No emojis, no gradients, no decorative animation, no marketing language, no "AI-powered" wording. Plain clinical labels with units and reference ranges.
7. **Mandatory Disclaimer**: Disclaimer for decision support / educational use must be visible on every screen.
8. **Code Quality**: Typed, small modules, docstrings only where non-obvious, no placeholder or fake data, no TODO left behind.
