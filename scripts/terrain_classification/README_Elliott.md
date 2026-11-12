# Terrain classification

I used these scripts to predict terrain from robot sensor recordings. I was able to run them without Isaac Sim.

- `prepare_recordings.py`: checked recording folders and made features for each walking cycle.
- `train_recordings.py`: prepared the data and trained the random forest version.
- `terrain_network.py`: defined the small PyTorch network.
- `train_pytorch.py`: trained that network using the prepared features.
- `predict_recording.py`: predicted terrain using either saved model.  
- `prepare.py`: handled the older single-CSV format. Its features were different from the folder version.

I followed the [PyTorch Quickstart tutorial](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) for the neural network version.

## How I ran it

I ran these commands from the project folder. I only needed to do this setup once:

```powershell
python3.15 -m venv .venv-terrain
.\.venv-terrain\Scripts\python.exe -m pip install -r scripts/terrain_classification/requirements.txt
```

First, I prepared the recordings and trained the first model:

```powershell
.\.venv-terrain\Scripts\python.exe -B scripts/terrain_classification/train_recordings.py C:\Users\ellio\Downloads\terrain_data artifacts/terrain_classifier
```

Then I trained the PyTorch model using the same features and data splits:

```powershell
.\.venv-terrain\Scripts\python.exe -B scripts/terrain_classification/train_pytorch.py artifacts/terrain_classifier/features.csv artifacts/terrain_classifier
```

To predict one recording, I replaced the example path below with a folder containing `meta.json`, `imu.csv`, and `joints.csv`:

```powershell
.\.venv-terrain\Scripts\python.exe -B scripts/terrain_classification/predict_recording.py artifacts/terrain_classifier/terrain_model.pt "C:\path\to\recording"
```

I used `terrain_model.joblib` instead for the random forest. I only loaded model files I trusted.

## Notes

I saved the models, `features.csv`, and reports in `artifacts/terrain_classifier/`. I left the original recordings unchanged. The JSON reports contained results and skipped recordings.

I kept matching gait programs in the same split and kept terrain labels separate. I left out labels with fewer than three usable gait programs. I had not yet tested either model on new collection days or on the live robot. Unknown terrain was assigned one of the known labels, and the score was only an estimate.

I saved both trained models locally in artifacts/terrain_classifier/: terrain_model.pt for PyTorch and terrain_model.joblib for the random forest. They are excluded from Git, so running predictions on another computer requires copying a saved model or training one first.