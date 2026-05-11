from glob import glob
from tqdm import tqdm
import os

data_list = sorted(glob(f"experiments/behave_test/*"))

for dir_path in tqdm(data_list):
    source_dir = dir_path.replace("/behave_test", "/behave")

    source_h = f"{source_dir}/human/human_params.npz"
    source_o = f"{source_dir}/object/object_params.npz"
    
    target_h = f"{dir_path}/human/human_params_cari4d.npz"
    target_o = f"{dir_path}/object/object_params_cari4d.npz"

    ret_h = 0
    ret_o = 0

    if not os.path.isfile(target_h):
        ret_h = os.system(f"cp {source_h} {target_h}")
    if not os.path.isfile(target_o):
        ret_o = os.system(f"cp {source_o} {target_o}")

    if ret_h != 0 or ret_o != 0:
        print(os.path.basename(dir_path))