# Edge-Synergy

## Quick Start

```bash
git clone https://github.com/ssjjrrr/Edge-Synergy.git
cd Edge-Synergy

# install dependency. In addition you need to download the appropriate version of Pytorch.
pip install -r requirements.txt

# initialize submodules 'ultralytics'
git submodule update --init --recursive

# run agent_test.py for RL partition results
python3 src/agent_test.py
```

## Dataset

In our project, we use PANDA Dataset for training the YOLOv8 models. We provide our yolo format dataset in [dataset in YOLO format](https://drive.google.com/file/d/1fYKes3c8CeWrupWZ-gJHummlX8D4a69H/view?usp=sharing). You can also visit the [official website](https://gigavision.cn/data/news/?nav=DataSet%20Panda&type=nav%2Findex.html) to get the entire dataset.

Or you can download the dataset by running the following commands:

```bash
cd data
# download the dataset, make sure you have downloaded curl
bash download_dataset.sh

# unzip the dataset
unzip PANDA.zip
```

## Checkpoints

We also provide the trained YOLOv8 checkpoints and RL network checkpoint for you to download. You can download the checkpoints by running the following commands:
```bash
cd checkpoints
# download the checkpoints, make sure you have downloaded curl
bash download_checkpoints.sh
```

Or you can select the checkpoints you want to download from the following table:

|Model Name|Image Size|Epoch|
|:---:|:---:|:---:|
|[yolov8n_640_ep300.pt](https://drive.google.com/file/d/155hpRllOb63Wg62-qNT1mJrl8tsE1Ul3/view?usp=drive_link)|640|300|
|[yolov8s_768_ep300.pt](https://drive.google.com/file/d/1nNEKdLZfwVavbfPsi_44zzGs9JLG-YID/view?usp=drive_link)|768|300|
|[yolov8m_896_ep300.pt](https://drive.google.com/file/d/1PPt7Z5q1qoZs6jGIW1u6gvgyRlsTStYn/view?usp=drive_link)|896|300|
|[yolov8l_1024_ep300.pt](https://drive.google.com/file/d/1EGOEhiX9H2pmmuz32ubFvPnLGBn6snJ3/view?usp=drive_link)|1024|300|
|[yolov8x_1024_ep300.pt](https://drive.google.com/file/d/1AZcSeba0JKFfeIZ-JxBiOK_TDlhz9yqT/view?usp=drive_link)|1024|300|
|[yolov8x_1280_ep300.pt](https://drive.google.com/file/d/1M4Y6PM6bupm-1d2UPKYqFNTiaiS_Exbu/view?usp=drive_link)|1280|300|
|[ppo_rl_clustering.zip](https://drive.google.com/file/d/14s7VwU7w7wYUqI37TOJFCoQC2BHOTJ8E/view?usp=sharing)|-|-|

## Data Format

The file structure is as follows:

```
project_root/
├── data/
│   └── PANDA/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── checkpoints/
│   ├── yolov8n_640_ep300.pt
│   ├── yolov8s_768_ep300.pt
│   ├── yolov8m_896_ep300.pt
│   ├── yolov8l_1024_ep300.pt
│   ├── yolov8x_1024_ep300.pt
│   ├── yolov8x_1280_ep300.pt
│   └── ppo_rl_clustering.zip
```
