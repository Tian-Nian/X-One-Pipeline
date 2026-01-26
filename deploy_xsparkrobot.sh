python example/deploy/deploy_origin.py \
    --base_model_name "openpi"\
    --base_model_class "PI0_DUAL"\
    --base_model_path "/home/xspark-ai/project/openpi_ckpts/fudai_0124_30000/"\
    --base_task_name "redbao"\
    --base_robot_name "xspark_robot"\
    --base_robot_class "XsparkRobot"\
    --video "cam_head"\
    --overrides \
    --train_config_name "pi05_full_base"
    # --eef
    # pi05_base_aloha_robotwin_lora
    # pi05_full_base pi05_full_base_eef pi05_full_base_rtc
    # /home/xspark-ai/project/control_your_robot/policy/openpi/checkpoint/pi05/new/30000/
    # /home/xspark-ai/project/control_your_robot/policy/openpi/checkpoint/airpods_14000/
    # /home/xspark-ai/project/openpi/checkpoint/38000_tmp/
    # /home/xspark-ai/project/control_your_robot/policy/openpi/checkpoint/putbox_color_mix/
