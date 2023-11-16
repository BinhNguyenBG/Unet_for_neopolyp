# BKAI-IGH NeoPolyp

Student Name: Nguyễn Thanh Bình
Student ID: 20210106

# Inferencing guideline

Step 1:
Add data "bkai-igh-neopolyp" to /kaggle/input/

Step 2:
First, we need to download the "model.pth" from Google Drive and put it in "/kaggle/working/"

```python
import requests
import os

drive_url = f'https://drive.google.com/uc?id=1nSuknDJRVcgU59N6w9K40B6RJ_ujJk6K&export=download&confirm=t&uuid=90412a97-6456-4d5d-bdfb-49fa13245942'

save_dir = '/kaggle/working/'

response = requests.get(drive_url)

with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

print('Save "model.pth" successfully!')
```

Inferring

```python
!git clone https://github.com/BinhNguyenBG/Unet_for_neopolyp
!cp /kaggle/working/model.pth /kaggle/working/Unet_for_neopolyp
!pip install git+https://github.com/mberkay0/pretrained-backbones-unet
!python /kaggle/working/Unet_for_neopolyp/infer.py
```
