script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# PSRT
mkdir "$script_dir/rethinkvsralignment/experiments/PSRT_Reccurrent/"
gdown 1UpGdtDlVRBGpTfF2G7SZReNlmH0ZjuWf -O "$script_dir/rethinkvsralignment/experiments/PSRT_Reccurrent/PSRT_Vimeo.pth"

cd "$script_dir/practical_rife"

# SAFA
mkdir train_log_SAFA
cd train_log_SAFA
gdown 1OLO9hLV97ZQ4uRV2-aQqgnwhbKMMt6TX -O 'SAFA_trained_model.zip'
pwd
unzip -o SAFA_trained_model.zip

cd ..

# RIFE
mkdir train_log_RIFE
cd train_log_RIFE
gdown 1mj9lH6Be7ztYtHAr1xUUGT3hRtWJBy_5 -O 'RIFE_trained_model.zip'
pwd
unzip -o RIFE_trained_model.zip

cd ..