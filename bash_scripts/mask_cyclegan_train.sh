# Sample training script to convert between VCC2SF3 and VCC2TF1
# Continues training from epoch 500

# --- [0, 25] ---

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TM1 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TM1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TM1 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TM1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 1

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TF1 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 1

# --- temporal mask [0, 25] with frequency mask [0, 20]  ---

# --- [0, 25] ---

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1_FreqMask20 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --spectrum_max_mask_len 20 \
    --gpu_ids 2

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TM1_FreqMask20 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TM1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --spectrum_max_mask_len 20 \
    --gpu_ids 2

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TM1_FreqMask20 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TM1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --spectrum_max_mask_len 20 \
    --gpu_ids 3

python3.10 -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TF1_FreqMask20 \
    --seed 0 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/ \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/training \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --spectrum_max_mask_len 20 \
    --gpu_ids 3