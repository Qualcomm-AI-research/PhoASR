# Vietnamese Automatic Speech Recognition - A Revisit

<p  align="center">
<a  href="https://github.com/qualcomm-ai-research/PhoASR"><img  src="https://img.shields.io/badge/Project%20Page-GitHub-blue?logo=github"  alt="Project Page"></a>&nbsp; <a  href="https://arxiv.org/abs/2603.14779"><img  src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"  alt="Paper"></a>&nbsp; <a  href="https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small"><img  src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow"  alt="Hugging Face Model"></a>
</p>

## 📖 Introduction

> Automatic Speech Recognition (ASR) performance is heavily dependent on the availability of large-scale, high-quality datasets. For low-resource languages, existing open-source ASR datasets often suffer from insufficient quality and inconsistent annotation, hindering the development of robust models. To address these challenges, we propose a novel and generalizable data aggregation and preprocessing pipeline designed to construct high-quality ASR datasets from diverse, potentially noisy, open-source sources. Our pipeline incorporates rigorous processing steps to ensure data diversity, balance, and the inclusion of crucial features like word-level timestamps. We demonstrate the effectiveness of our methodology by applying it to Vietnamese, resulting in a unified, high-quality dataset that provides a foundation for training and evaluating state-of-the-art Vietnamese ASR systems.

Details can be found in our [following paper](https://aclanthology.org/2026.findings-eacl.345/):

```bibtex
@InProceedings{Vu_2026_PhoASR,
    title     = {{Vietnamese Automatic Speech Recognition: A Revisit}},
    author    = {Thi Vu and Linh The Nguyen and Dat Quoc Nguyen},
    booktitle = {Findings of the Association for Computational Linguistics: EACL 2026},
    year      = {2026}
}
```

We release **[PhoASR-whisper-small](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small)**, a `whisper-small` checkpoint fine-tuned on the 3,000-hour high-quality Vietnamese dataset produced by our pipeline.

**Please CITE** our paper when our code or model is used to help produce published results or is incorporated into other software.


## 🛠️ Data Creation Pipeline

This repository contains the PhoASR data creation pipeline used to build the high-quality Vietnamese ASR dataset introduced in the paper. The default Vietnamese configuration follows the model stack described in our work.

### Installation

From the repository root, create the environment and install the pipeline:

```bash
conda create -n phoasr python=3.11 -y
conda activate phoasr
uv pip install -r requirements.txt
uv pip install -e .
conda install -c anaconda ffmpeg -y
```

Convert PhoWhisper into CTranslate2 format before running the verifier stage:

```bash
ct2-transformers-converter --model vinai/PhoWhisper-large --output_dir checkpoints/phowhisper-large-ctranslate
```

Before running the pipeline, download the fine-tuned punctuation/capitalization checkpoint: a [BARTpho](https://github.com/vinairesearch/bartpho) model fine-tuned on Vietnamese data for punctuation restoration and capitalization. This step is required for any full run:

```bash
python download_checkpoints.py
```

This downloads and extracts the checkpoint into `checkpoints/`. It is idempotent and skips a checkpoint that is already present. Alternatively, download it manually from the [GitHub releases page](https://github.com/qualcomm-ai-research/PhoASR/releases) and place it under `checkpoints/bartpho-punc-cap`. The remaining models in the default stack (Parakeet ASR/alignment and the num2word normalizer) are pulled automatically from Hugging Face on first use.

The BSD-3 Clear License of this repository also applies to the model weights. See [LICENSE](LICENSE) for the full license text.

### Prepare Data

Prepare your data in the same layout as `examples/demo_vi/`:

```text
examples/your_dataset/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.ogg
│   └── ...
└── input_manifest.csv
```

The manifest must contain at least:

- `sample_id`
- `audio_path`

Optional columns include:

- `transcript`
- `dataset_name`
- `speaker`
- `region`
- `province`
- `duration`

Use paths in `audio_path` relative to the manifest location, following `examples/demo_vi/input_manifest.csv`. For example:

```csv
sample_id,audio_path,transcript,dataset_name
sample_001,audio/sample_001.wav,Tôi đọc những cái tư liệu đầu tiên...,source_a
sample_002,audio/sample_002.ogg,,source_b
```

Samples with a non-empty `transcript` are treated as already transcribed. Samples with an empty `transcript` are sent to the transcription stage.

### Run the Pipeline

The Vietnamese configurations are:

- `configs/default.vi.yaml` — the default pipeline (all used models are commercial friendly).
- `configs/paper.vi.yaml` — use this to reproduce the exact model stack described in our paper.
- `configs/minimal.vi.yaml` for the minimally processed variant used for paper Steps 1, 2, and 4 only.

The default configuration uses the following model stack:

- primary ASR: `parakeet_nemo` with `nvidia/parakeet-ctc-0.6b-Vietnamese`
- verifier ASR: `PhoWhisper-large` in CTranslate2 format
- punctuation/capitalization: `BARTpho` model fine-tuned on Vietnamese data (See the Installation section)
- num2word: Vietnamese text-normalization model [PhoTextNormalization](https://huggingface.co/thivux/PhoTextNormalization)
- alignment: `parakeet_nemo_align` (forced CTC alignment) with the same Parakeet model

The `configs/paper.vi.yaml` configuration swaps the primary ASR and alignment
backends for the models described in our paper, while keeping the same verifier,
punctuation/capitalization, and num2word models.

The pipeline code defines the following stages, in order:

- `validate`: validate the input manifest, normalize audio, and apply basic transcript and duration checks
- `transcribe`: generate transcripts for samples whose `transcript` field is empty
- `score_wer`: score transcript quality with the verifier ASR model
- `filter`: drop low-quality samples based on configured filtering rules
- `punc_cap`: restore punctuation and capitalization
- `num2word`: normalize spoken-form text using the Vietnamese text-normalization model
- `align`: generate word-level timestamp artifacts
- `finalize`: assemble the final output manifest and timestamp projections

Run the full pipeline:

```bash
phoasr-data-pipeline run \
  --config configs/default.vi.yaml \
  --input examples/demo_vi/input_manifest.csv \
  --output runs/demo_full
```

Run a specific stage range:

```bash
phoasr-data-pipeline run \
  --config configs/default.vi.yaml \
  --input examples/demo_vi/input_manifest.csv \
  --output runs/demo_full \
  --start-stage score_wer \
  --end-stage finalize
```

## 📁 Repository Structure

```
phoasr/
├── download_checkpoints.py       # Script to download the punctuation/capitalization checkpoint
├── requirements.txt              # Full Python dependency list
├── pyproject.toml                # Package build configuration
├── configs/
│   ├── default.vi.yaml           # Full 8-stage Vietnamese pipeline (all used models are commercial friendly)
│   ├── paper.vi.yaml             # Full 8-stage variant using the paper's stack
│   └── minimal.vi.yaml           # Minimal 3-stage configuration (Steps 1, 2, 4 of the paper)
├── examples/
│   └── demo_vi/
│       ├── audio/                # Sample audio files (.m4a)
│       └── input_manifest.csv    # Sample input manifest
└── src/
    └── phoasr_data_pipeline/
        ├── cli.py                # CLI entry point (phoasr-data-pipeline)
        ├── pipeline.py           # Pipeline orchestration and stage dispatch
        ├── config.py             # YAML configuration loader
        ├── constants.py          # Stage names and directory mappings
        ├── io.py                 # Manifest I/O and validation
        ├── audio.py              # Audio normalization and duration helpers
        ├── transcription.py      # Transcription stage logic
        ├── quality.py            # WER scoring and filtering
        ├── formatting.py         # Punctuation and capitalization restoration
        ├── normalization.py      # Number-to-word normalization
        ├── alignment.py          # Word-level timestamp generation
        ├── postprocess.py        # Final manifest assembly
        ├── text.py               # Text normalization utilities
        ├── mapping_utils.py      # Token mapping for timestamp projection
        └── backends/
            ├── asr.py            # ASR backends
            ├── alignment.py      # Alignment backends
            └── seq2seq.py        # BARTpho / mBART seq2seq backend
```
