#!/usr/bin/env bash
# /bin/bash

backbone='pointnet'
test_dir="/home/mark/NAS/Airplane/real_airplane/Data"
begin=700
end=900
ground_high=-4

for dir in `ls $test_dir`
do
  echo "Running, The current direction is ${dir}"
  dir=${test_dir}/${dir}
  gt_type=`echo $dir | grep -Eo "[a-bA-B]+[0-9]{3}+\-+[0-9]{3}"` #正则表达式提取飞机标签
  gt_type=`echo ${gt_type^}`  #首字母大写
  echo "The real airplane type is ${gt_type}"
  python main.py --backbone $backbone --seq_path $dir --gt_type $gt_type --begin=$begin --end=$end --ground_height=$ground_high
done

