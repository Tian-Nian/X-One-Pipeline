# 环境安装
``` bash
# 基础环境安装
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# 额外环境安装
uv pip install -r requirements.txt
# 如果有额外的库, 有关pip的变成uv pip就行
```

# 模型部署
为了方便设置语言指令, task_instructions放置在根目录, 需要link到src中:
``` bash
ln -s task_instructions/ src/robot/
```
## 整体介绍

`--base_model_name`: 对应`src/robot/policy/`下对应模型的文件夹名

`--base_model_class`: 对应`src/robot/policy/${base_model_name}/inference_model.py`中的类名

`--base_model_path`: 对应模型权重路径下

`--base_task_name`: 对应`task_instructions/`中的语言指令文件名

`--base_robot_name`: 对应`my_robot/`下对应机器人文件名

`--base_robot_class`: 对应`my_robot/${base_robot_name}.py`中的类名

`--is_robotwin`: 是否使用的`RoboTwin`部署的流程

`--video`: 表示你希望录制哪个视角的视频(可选)

`--orverrides`: 额外参数, ROboTwin模型初始化, 和基础模型中的openpi用于初始化模型的参数

## 非RoboTwin模型
根据介绍设置好对应参数即可使用, 注意`ENTER`用来控制开启与结束.

## RoboTwin支持模型
1. 在`control_your_robot`根目录下执行:
`ln -s /path/to/RoboTwin/policy/${policy} src/robot/policy/`

2. 修改`deploy.sh`
**对于RoboTwin的模型, 需要加上`--robotwin`参数!!!**
以RoboTwin部署PI0的脚本为例子:
1. 将robotwin对应的`--override`参数写到`deploy.sh`中.
2. base_model_name改为ln -s后在`src/robot/policy/`下的对应文件夹名称
3. 选择自己的机器人封装
```bash
# pi0 eval.sh
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 

# your deploy.sh
python example/deploy/deploy.py \
    --base_model_name "openpi"\
    --base_model_class "None"\
    --base_model_path "None"\
    --base_task_name "test"\
    --base_robot_name "test_robot"\
    --base_robot_class "TestRobot"\
    --robotwin \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
```

# X one格式转化
1. 采集阶段直接保存为X one格式
在`my_robot/`对应的机器人文件中, 先引入函数:
`from robot.utils.base.data_transform_pipeline import X_one_format_pipeline`,然后在`__init__()`最下方加上`self.collection._add_data_transform_pipeline(X_one_format_pipeline)`

2. 将已经保存的数据转化为X one格式
``` bash
# data_dir:数据存放的跟路径, 目标数据集路径为: data_dir/task_name/*.hdf5
python scripts/convert2x_one.py ${data_dir} ${output_dir} ${task_name}
```