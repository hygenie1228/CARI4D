# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
sys.path.append(os.getcwd())
from prep.fp_behave import merge_pickles
from glob import glob
import time 
from prep.fp_filter_2dir import FPFilterTwoDirProcessor
import json
import os.path as osp


class BehaveHy3D2DirFPRunner(FPFilterTwoDirProcessor):
    def get_template_file(self):
        "load from hy3d"
        root = osp.normpath(self.args.hy3d_root)
        vp = self.video_prefix
        # Exact layout from scripts/stage_exp_behave.py / stage_exp_intercap.py (glob can miss on some trees).
        direct = osp.join(root, f"{vp}_export", f"{vp}_align.obj")
        if osp.isfile(direct) and not osp.islink(direct):
            print("using object template from file:", direct)
            return direct
        # Standard wild layout: <exp>/cari4d/hy3d_staged — build centered align from object/model.obj
        if (
            osp.basename(root) == "hy3d_staged"
            and osp.basename(osp.dirname(root)) == "cari4d"
        ):
            exp_dir = osp.dirname(osp.dirname(root))
            raw_obj = osp.join(exp_dir, "object", "model.obj")
            if osp.isfile(raw_obj):
                os.makedirs(osp.dirname(direct), exist_ok=True)
                if osp.lexists(direct):
                    os.remove(direct)
                import importlib.util

                repo = osp.abspath(osp.join(osp.dirname(__file__), ".."))
                stage_script = (
                    "stage_exp_open4dhoi.py"
                    if self.args.data_source == "open4dhoi"
                    else "stage_exp_behave.py"
                )
                sp = osp.join(repo, "scripts", stage_script)
                spec = importlib.util.spec_from_file_location("_stage_mesh", sp)
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                mod.write_obj_aabb_center_at_origin(raw_obj, direct)
                print(f"[fp_hy3d_2dir] staged centered mesh {raw_obj} -> {direct}")
                return direct
        pat = osp.join(root, f"{vp}*", f"*{vp}*_align.obj")
        files = sorted(glob(pat))
        if len(files) == 0:
            hint = ""
            if (
                osp.basename(root) == "hy3d_staged"
                and osp.basename(osp.dirname(root)) == "cari4d"
            ):
                ed = osp.dirname(osp.dirname(root))
                hint = f"; expected source OBJ at {osp.join(ed, 'object', 'model.obj')}"
            raise ValueError(
                f"no aligned hy3d template for {vp!r} under {root!r} "
                f"(tried {direct!r}, glob {pat!r}{hint})"
            )
        mesh_file = files[0]
        print("using object template from file:", mesh_file)
        return mesh_file

def _child_run_kid(kid, args):
    """Run one view in an isolated child process.
    Creates CUDA/context-heavy objects inside the child to avoid pickling issues.
    """
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    processor_child = BehaveHy3D2DirFPRunner(args) # this holds some thread lock 
    processor_child.process_video(kid)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def process_video(args):
    # use multiple processes to process different views of this video 
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    procs = []  # run 4 processes in parallel, around 7s/frame

    # run only on the selected views 
    selected_views = json.load(open('splits/selected-views-map.json'))
    video_prefix = osp.basename(args.video).split('.')[0]
    kids = [int(selected_views[video_prefix][1])] if video_prefix in selected_views else args.cameras
    args.cameras = kids 
    print(f"running kids {kids} for seq {video_prefix}")
    for k in kids:
        p = ctx.Process(target=_child_run_kid, args=(k, args))
        p.start()
        time.sleep(5) # to avoid race condition
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    parser = FPFilterTwoDirProcessor.get_parser()
    args = parser.parse_args()
    import traceback

    try:
        if osp.isfile(args.video):
            videos = [args.video]
        else:
            videos = sorted(glob(args.video))
        print(f"In total {len(videos)} video files")
        selected_views = json.load(open('splits/selected-views-map.json'))
        video_prefix = osp.basename(args.video).split('.')[0]
        if args.index is not None:
            chunk_size = 1  # for easy paralle 
            videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
        print(f"Processing {len(videos)} video files, first video: {videos[0]}, last video: {videos[-1]}")
        for video in videos:
            args.video = video
            processor = BehaveHy3D2DirFPRunner(args)
            if args.wild_video:
                kid_to_run = args.kid
            else:
                kid_to_run = int(selected_views[video_prefix][1]) if video_prefix in selected_views else args.kid
            processor.process_video(kid_to_run)

        # now collect results from different cameras into one file
        merge_pickles(videos, args)
    except Exception as e:
        print(args.video, 'failed')
        traceback.print_exc()
    


        
