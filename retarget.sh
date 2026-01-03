# Genesis viewer
export PYTHONPATH=$PYTHONPATH:/Users/huangxiansheng/Project
python scripts/smplx_to_robot_gs.py \
    --smplx_file /Users/huangxiansheng/Downloads/Stefanos_1os_antrikos_karsilamas_C3D_stageii.npz \
    --loop 
    # --smplx_file /Users/huangxiansheng/Downloads/Stefanos_1os_antrikos_karsilamas_C3D_stageii.npz
    

python scripts/vis_robot_motion_gs.py \
    --robot_motion_path /Users/huangxiansheng/Downloads/dance_db_gmr/01_01_stageii.pkl

# python scripts/retarget_to_redis.py \
#   --smplx_file /Users/huangxiansheng/Downloads/01_01_stageii.npz \
#   --redis_url redis://localhost:6379/0 \
#   --redis_key motion:ref:latest \
#   --loop

# Single motion
# --smplx_file /Users/huangxiansheng/Dataset/AMASS/DanceDB_smplx/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_stageii.npz
mjpython scripts/smplx_to_robot.py \
    --smplx_file /Users/huangxiansheng/Downloads/Stefanos_1os_antrikos_karsilamas_C3D_stageii.npz \
    --save_path /Users/huangxiansheng/Downloads/Stefanos_1os_antrikos_karsilamas_C3D_stageii.pkl \
    --rate_limit

# Motion directory
mjpython scripts/smplx_to_robot_dataset.py \
    --src_folder /Users/huangxiansheng/Downloads/dance_db \
    --tgt_folder /Users/huangxiansheng/Downloads/dance_db_gmr \
    --num_cpus 1

# Visualize retargeted motion
mjpython scripts/vis_robot_motion.py \
    --robot_motion_path /Users/huangxiansheng/Downloads/Subject_75_F_MoSh_GMR/Subject_75_F_4_stageii.pkl \
    --record_video \
    --video_path /Users/huangxiansheng/Downloads/Subject_75_F_MoSh_GMR/Subject_75_F_4_stageii.mp4