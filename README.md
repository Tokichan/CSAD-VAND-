# Environment installation:
python==3.10

pytorch==2.2.0 with cuda==12.1
`pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`

Grounding DINO
`pip install git+https://github.com/IDEA-Research/GroundingDINO.git`

SAM
`pip install git+https://github.com/facebookresearch/segment-anything.git`

other dependencies
`pip install -r requirements.txt`

# To run the evaluation
`python evaluation.py --module_path vand_model --class_name CSAD --dataset_path DATASET_PATH  --category CATEGORY`

replace the DATASET_PATH and CATEGORY with your config

