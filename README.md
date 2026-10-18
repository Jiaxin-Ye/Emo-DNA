<div align="center">
  <h1>Emo-DNA</h1>
  <p>
    Official PyTorch implementation for training and inference in
    <br>
    <b><em>Emotion Decoupling and Alignment Learning for Cross-Corpus Speech Emotion Recognition</em></b>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2308.02190">
      <img src="https://img.shields.io/badge/arXiv-2308.02190-b31b1b.svg" alt="arXiv">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-orange.svg" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.1%2B-brightgreen.svg" alt="PyTorch">
  </p>
</div>

## 📕 Introduction

Emo-DNA is a cross-corpus speech emotion recognition framework that decouples emotion-related and corpus-related representations and applies dual-level alignment between speech emotion domains. The released implementation covers binary arousal and valence classification across CASIA, EMODB, EMOVO, IEMOCAP, RAVDE, and SAVEE.

## 📖 Usage

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The main configuration is [`configs/emodna.yaml`](configs/emodna.yaml).

The complete release archive includes the source code, checkpoints, and [reference .npy features](https://github.com/Jiaxin-Ye/Emo-DNA/releases/download/MFCC/reference_features.zip).

For the reproduction of the released results, use the reference .npy features with the provided checkpoint and evaluation setting. But numerical differences may occur because MFCC extraction depends on the audio files, sampling rates, resampling, random seed, and library versions. 

### 2. Prepare WAV Data

Raw speech corpora are not included. Organize the audio files as follows:

```text
SER_WAV_DATA/
  CASIA/<emotion>/*.wav
  EMODB/<emotion>/*.wav
  EMOVO/<emotion>/*.wav
  IEMOCAP/<emotion>/*.wav
  RAVDE/<emotion>/*.wav
  SAVEE/<emotion>/*.wav
```

The model uses 40-dimensional MFCC features. The recommended training path extracts MFCCs directly from WAV files. IEMOCAP uses 48 kHz extraction to match the reference configuration; the other corpora use their native WAV sampling rates.

To save a corpus as an NPY feature bundle:

```bash
python -m tools.wav_to_mfcc_npy \
  --audio-root /path/to/SER_WAV_DATA \
  --corpus IEMOCAP \
  --out-dir reference_features_audio/IEMOCAP
```

The generated bundle contains `features.npy`, `labels.npy`, and `metadata.json`.

### 3. Training

Train one cross-corpus scenario with online WAV feature extraction:

```bash
python train.py \
  --source CASIA \
  --target EMODB \
  --task arousal \
  --input-format audio \
  --audio-root /path/to/SER_WAV_DATA
```

To train from NPY bundles instead:

```bash
python train.py \
  --source CASIA \
  --target EMODB \
  --task arousal \
  --input-format npy \
  --features-root reference_features
```

Use `--task valence` for valence classification. The label mappings are:

```text
arousal:  0 = low,      1 = high
valence:  0 = negative, 1 = positive
```

### 4. Inference

Run a released checkpoint on a WAV file:

```bash
python infer.py \
  --checkpoint checkpoints/arousal/Arousal-Emo-DNA-S_CASIA-T_EMODB.pt \
  --task arousal \
  --input-format audio \
  sample.wav
```

Use `--input-format npy` for a single `[time, 40]` NPY file or a batch with shape `[N, time, 40]`.

### 5. Evaluation

The unified evaluation script supports either packaged NPY features or direct WAV input.

Evaluate the released checkpoints on NPY features:

```bash
python evaluate.py \
  --input-format npy \
  --features-root reference_features \
  --checkpoint-root checkpoints \
  --output evaluation_npy.csv
```

Evaluate the released checkpoints directly from WAV files:

```bash
python evaluate.py \
  --input-format audio \
  --audio-root /path/to/SER_WAV_DATA \
  --checkpoint-root checkpoints \
  --output evaluation_audio.csv
```

The release contains 30 checkpoints for each task under
`checkpoints/arousal/` and `checkpoints/valence/`. They use the original `E`/`EMO_E` checkpoint structure and can be loaded by the release loader.

Verify the released model and feature files after downloading:

```bash
sha256sum -c SHA256SUMS
```

## 📖 Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{emodna,
  author       = {Jiaxin Ye and
                  Yujie Wei and
                  Xin{-}Cheng Wen and
                  Chenglong Ma and
                  Zhizhong Huang and
                  Kunhong Liu and
                  Hongming Shan},
  title        = {Emo-DNA: Emotion Decoupling and Alignment Learning for Cross-Corpus
                  Speech Emotion Recognition},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Multimedia,
                  {MM} 2023},
  pages        = {5956--5965},
  year         = {2023},
  doi          = {10.1145/3581783.3611704}
}
```
