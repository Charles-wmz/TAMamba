# TAMamba: Full Model Training

This repository contains the training implementation of the Full four-expert
TAMamba system for visible-infrared person re-identification on RegDB and
SYSU-MM01. A single command trains all four experts in sequence.

The architecture and supervision recipes are fixed. There are no module,
ablation, expert-selection, evaluation, or visualization switches. Dataset files,
pretrained weights, experiment results, and trained checkpoints are not bundled.
The `data/` and `pretrained/` directories contain only Git placeholder files.

## 1. Install the environment

Use Python 3.10-3.12. The pinned dependencies match the original experiment
environment; local verification used Python 3.12, PyTorch 2.7.1, and an RTX 4090.
A CUDA-capable GPU is recommended for training. CPU execution is supported but
will be slow.

From this directory:

```bash
python -m venv .venv
```

Activate the environment on Linux:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For GPU training, the installed PyTorch build must support CUDA and the local
NVIDIA driver. If CUDA is unavailable, install a compatible CUDA build of
PyTorch 2.7.1 and torchvision 0.22.1 before training. The selective state-space
module is implemented in PyTorch; `mamba-ssm` and custom CUDA extensions are not
required.

## 2. Prepare the training data

Obtain the datasets separately and preserve their original image paths. You can
place them under `data/` or pass an existing location using `--data-root`.
Training requires real training images; the repository does not create or
download datasets.

### RegDB

For trial 1, the required layout is:

```text
data/RegDB/
  idx/
    train_visible_1.txt
    train_thermal_1.txt
  Visible/
    ...
  Thermal/
    ...
```

Each index line contains an image path relative to the dataset root, followed
by its integer identity label:

```text
Visible/1/0001_v_1.bmp 0
```

The actual filenames and directory capitalization must match the index files.
Both modalities must contain the same training identities. Labels are remapped
to consecutive class indices. For another trial, provide its corresponding
`train_visible_<trial>.txt` and `train_thermal_<trial>.txt` files. Test index files
are not read.

### SYSU-MM01

```text
data/SYSU-MM01/
  exp/
    train_id.txt
    val_id.txt
  cam1/<identity>/<image>
  cam2/<identity>/<image>
  cam3/<identity>/<image>
  cam4/<identity>/<image>
  cam5/<identity>/<image>
  cam6/<identity>/<image>
```

Identity folders can use four-digit names such as `0001`. Training combines the
official training and validation identities, following the original training
code. Cameras 1, 2, 4, and 5 provide visible images; cameras 3 and 6 provide
infrared images. Each training identity needs at least one image in each
modality. `test_id.txt` and `rand_perm_cam.mat` are not required or read.

## 3. Prepare CLIP RN101 weights

By default, OpenCLIP loads the OpenAI RN101 pretrained model, using the cache
when available and downloading the weights when necessary.

For an offline run, obtain the OpenCLIP RN101 checkpoint separately, place it at
`pretrained/open_clip_model.safetensors`, and add:

```text
--pretrained pretrained/open_clip_model.safetensors
```

An absolute path to an existing compatible checkpoint also works. The file must
contain RN101 weights; a different CLIP backbone cannot be substituted.

## 4. Train the Full model

RegDB trial 1, with data under `data/RegDB`:

```bash
python train.py --dataset regdb --trial 1
```

SYSU-MM01, with data under `data/SYSU-MM01`:

```bash
python train.py --dataset sysu-mm01
```

Use existing data and local pretrained weights without copying them into this
repository:

```bash
python train.py --dataset regdb --trial 1 --data-root /path/to/RegDB --pretrained /path/to/open_clip_model.safetensors
python train.py --dataset sysu-mm01 --data-root /path/to/SYSU-MM01 --pretrained /path/to/open_clip_model.safetensors
```

For example, when this directory is still inside the original `code/` folder,
the following works in both Bash and PowerShell if those local assets exist:

```bash
python train.py --dataset regdb --data-root ../data/RegDB --pretrained ../pretrained/open_clip_model.safetensors
```

The four fixed experts are:

| Expert | TAMamba + M3F + WMDR + IC-DEE | Spatial alignment | CM-MPPR |
| --- | --- | --- | --- |
| `global` | Included | Not part of this expert | Not part of this expert |
| `spatial` | Included | Weight 0.1 | Not part of this expert |
| `cmmppr_global` | Included | Not part of this expert | Included |
| `cmmppr_spatial` | Included | Weight 0.1 | Included |

These are the four constituent experts of Full, not selectable ablation
variants. Each is initialized and trained independently. The two CM-MPPR
experts use six reconstruction parts, mask ratio 0.5, and a loss weight that
increases linearly to 0.2 over the first ten epochs.

Default training settings:

| Setting | Value |
| --- | --- |
| Backbone | CLIP RN101 with separate visible/infrared stems |
| Image size | 288 x 144 |
| Epochs | 120 per expert |
| Batch | 8 identities x 2 images per modality = 32 images |
| Seed | 42 for each expert |
| Optimizer | AdamW, weight decay 0.0005 |
| Learning rates | Backbone 0.00002; task modules 0.0003 |
| Scheduler | Cosine annealing |
| Precision | FP32 |
| Representation | Fixed 8192-dimensional fused descriptor per expert |

Paired geometric augmentation, cross-spectral blending, CLIP normalization,
cross-batch memory, and composite identity/metric supervision are always used.
RegDB sampling matches visible/thermal filenames when possible; SYSU sampling
draws the two modalities independently for each identity.

Only dataset paths, trial, pretrained path, epoch budget, batch size, worker
count, and device are exposed as command-line arguments:

```bash
python train.py --help
```

For a smaller batch, for example:

```bash
python train.py --dataset regdb --identities-per-batch 4 --instances-per-modality 1
```

This still trains all four experts. Changing the batch size or epoch budget
changes the training recipe. At least two identities per batch are required.
Workers default to zero for portability; increase `--workers` if appropriate.
Use `--device cuda:0` to select a GPU or `--device cpu` for CPU execution.

Train all ten RegDB trials in Bash:

```bash
for trial in {1..10}; do
  python train.py --dataset regdb --trial "$trial" || exit 1
done
```

Or in PowerShell:

```powershell
1..10 | ForEach-Object {
    python train.py --dataset regdb --trial $_
    if ($LASTEXITCODE -ne 0) { throw "Training failed for trial $_" }
}
```

## 5. Training progress and interruptions

Training displays progress and losses in the terminal. It does not create an
output directory or save trained weights, results, configuration files, or
log files. The default budget is 120 epochs per expert.

Every invocation initializes all four experts from the pretrained RN101 weights
and trains them in sequence. After an interruption, rerunning the command starts
training again from the first expert; there is no checkpoint or resume step.

Datasets and input pretrained weights are excluded by `.gitignore`. Retrieval
evaluation, distance fusion, re-ranking, metrics, and visualization scripts are
outside the scope of this training-only release.

## Source layout

```text
train.py              Full four-expert training entry point
models/network.py     Fixed TAMamba networks and auxiliary heads
models/ssm.py         Bidirectional selective state-space module
datasets/             Training loaders, transforms, and identity sampler
losses/reid.py        Composite Full-model supervision
engine/trainer.py     Paired augmentation and one-epoch training
utils/runtime.py      Optimizer and reproducibility helpers
data/.gitkeep         Empty dataset directory placeholder
pretrained/.gitkeep   Empty pretrained-weight directory placeholder
requirements.txt      Training dependencies
```
