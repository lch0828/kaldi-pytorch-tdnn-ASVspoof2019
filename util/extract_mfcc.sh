#!/bin/sh

HOME_PATH=$HOME
dataset_dir="../data/LA"
output_dir="../data/mfcc"
s1_to_util="$HOME/util"
stage=2

if [ $stage -le 0 ]; then
  # Loop over train, eval, and dev sets
  for subset in "ASVspoof2019_LA_train" "ASVspoof2019_LA_eval" "ASVspoof2019_LA_dev"; do
    # Convert .flac to .wav
    for file in $s1_to_util/$dataset_dir/${subset}/flac/*.flac; do
      echo "Converting $file"
      sox "$file" -r 16000 -c 1 -b 16 -t wav "$s1_to_util/$dataset_dir/wav/${subset}/$(basename "$file" .flac).wav"
    done
  done
fi

if [ $stage -le 1 ]; then
  for subset in "ASVspoof2019_LA_train" "ASVspoof2019_LA_eval" "ASVspoof2019_LA_dev"; do
    # Create wav.scp file for the subset
    find "$s1_to_util/$dataset_dir/wav/${subset}" -name "*.wav" -type f | while read -r wav_file; do
    utterance_id="${wav_file#$dataset_dir/}"
    echo "${subset}_${utterance_id%.wav} $wav_file"
  done > $s1_to_util/$dataset_dir/wav/${subset}/wav.scp
done
fi

if [ $stage -le 2 ]; then
  for subset in "ASVspoof2019_LA_train" "ASVspoof2019_LA_eval" "ASVspoof2019_LA_dev"; do
    # Extract MFCC features using Kaldi for the subset
    steps/make_mfcc.sh --mfcc-config $s1_to_util/conf/mfcc.conf --nj 1 $s1_to_util/$dataset_dir/wav/${subset} $s1_to_util/$output_dir/exp/make_mfcc_${subset} $s1_to_util/$output_dir/feature/${subset}
  done
fi
