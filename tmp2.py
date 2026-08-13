from glob import glob
from tqdm import tqdm
import os


data_list = sorted(glob(f"experiments/behave/*"))

for dir_path in tqdm(data_list):
    os.system(f"rm -rf {dir_path}/cari4d")