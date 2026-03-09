PYTHONWARNINGS=ignore::UserWarning \
python policy_lab/setup_policy_server.py \
    --port 10001 \
    --config_path policy_lab/replay_policy/deploy.yml \
    --overrides \
    --task_name demo_task \
    --policy_name replay_policy \
    --data_path "data/mertuan_demo_2/x-one/5.hdf5" \