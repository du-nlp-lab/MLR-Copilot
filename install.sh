#/bin/bash

# auto-gpt
# pip install -r Auto-GPT/requirements.txt

# crfm api
# pip install crfm-helm

# ML dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install typing-inspect==0.8.0 typing_extensions==4.5.0
pip install pydantic -U
pip install numpy==1.26.4
pip install --force-reinstall charset-normalizer==3.1.0
