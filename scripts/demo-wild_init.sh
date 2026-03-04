video="data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4"
video_prefix=$(basename "$video" | cut -d. -f1)
echo $video_prefix

set -e

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --wild_video --video ${video} -o data/cari4d-demo/wild/videos/

# Step 2: run GENMO
# see: https://github.com/NVlabs/GENMO

# Step 3: align Unidepth to GENMO human
python prep/align_monod2hum.py --wild_video --nlf_path data/cari4d-demo/wild/genmo \
--masks_root data/cari4d-demo/wild/masks/ \
--video ${video}

# Step 4: run FP in tracking mode
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root data/cari4d-demo/wild/masks/ --hy3d_root=data/cari4d-demo/meshes \
--video ${video} -o data/cari4d-demo/wild/fp-hy3d3-track
