export GLOG_v=2
export RANK_TABLE_FILE=$1
ROOT_PATH=`pwd`
LOCAL_DEVICE_NUM=$2
for((i=0;i<${LOCAL_DEVICE_NUM};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export DEVICE_ID=$[i]
    export RANK_ID=$[i]
    python ${ROOT_PATH}/tools/train.py --config ${ROOT_PATH}/configs/rec/svtr_tiny.yaml > run_train.log 2>&1 &
done
