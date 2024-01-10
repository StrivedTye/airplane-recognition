#!/usr/bin/env bash

test_dir='/home/mark/NAS/Airplane/real_airplane/Data'
dir=('20201017_121903_DEFAULT-A320-200' '20201017_140356_DEFAULT-A320-200' '20201018_202159_DEFAULT-A320-200' \
'20201019_121603_DEFAULT-a320-200' '20201021_201930_DEFAULT-a320-200' '20201022_083935_DEFAULT-b737-800' '20201022_141003_DEFAULT-A320-200' \
'20201103_085730_DEFAULT-B737-800' '20201107_213255_DEFAULT-a320-200' '20201108_131556_DEFAULT-a320-200' '20201108_213056_DEFAULT-a320-200' \
'20201109_084451_DEFAULT-B737-800' '20201109_121617_DEFAULT-a320-200' '20201109_173247_DEFAULT-a320-200' '20201110_084149_DEFAULT-b737-800' '20201110_175912_DEFAULT-A320-200' \
'20201111_121752_DEFAULT-A320-200' '20201112_082925_DEFAULT-a320-200' '20201112_213106_DEFAULT-A320-200')

begin=(700 700 900 0 400 400 400 0 100 300 100 200 300 200 0 300 400 300 200)

for i in $(seq 0 ${#dir[@]})
do
  idx=${dir[i]}
  curbegin=${begin[i]}
  gt_type=`echo $idx | grep -Eo "[a-bA-B]+[0-9]{3}+\-+[0-9]{3}"` #正则表达式提取飞机标签
  gt_type=`echo ${gt_type^}`  #首字母大写
  str1='/seq_path/s/Data\/.*/Data\/'${idx}'/'
  str2='/begin/s/: .*/: '${curbegin}'/'
  str3='/gt_type/s/: .*/: '${gt_type}'/'
  order1="sed -i '"${str1}"' config/test_simulation.yaml"
  order2="sed -i '"${str2}"' config/test_simulation.yaml"
  order3="sed -i '"${str3}"' config/test_simulation.yaml"
#  echo $order1
#  echo $order2
#  echo $order3
  eval ${order1}
  eval ${order2}
  eval ${order3}
  python main.py
done
