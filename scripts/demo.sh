video=data/cari4d-demo/behave/videos/Date03_Sub03_chairblack_debug.2.color.mp4

set -e

# # Step 1: run Unidepth estimation
python prep/unidepth_behave.py --cameras 2 --data_source behave -o data/cari4d-demo/behave/videos/  --video ${video}

# Step 2: run NLF to estimate human pose
python prep/run_nlf_sepK.py --data_source behave -o data/cari4d-demo/behave/nlf-smplh-gender-sepK --masks_root data/cari4d-demo/behave/masks/ --video ${video} -tstart=0.0 --redo

# # Step 3: align NLF to unidepth prediction
python prep/align_nlf2unidepth.py --data_source behave -o data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth --masks_root data/cari4d-demo/behave/masks \
--nlf_path data/cari4d-demo/behave/nlf-smplh-gender-sepK --video ${video} -tstart 0.0
