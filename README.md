# Deepfake Audio Detection With Time-Delay Neural Network
### 


## Tools

* https://github.com/cvqluu/Factorized-TDNN

* https://www.asvspoof.org/index2019.html

## Run
1. Download Kaldi tool
    ```
    ./kaldi_prerequisites.sh
    ```

2. Download ASVspoof2019 dataset from above link to ```./data``` folder

3. Create corresponding path, copy ```extract_mfcc.sh``` and extract MFCC feture with Kaldi
    ```
    ./util/kaldi/egs/6998/s1/extract_mfcc.sh
    ```

4. Train and test
    ```
    ./main/run.sh
    ```
    default parameter is set fro best performing configuration.

5. Scoring: Metric function is obtained from official site
    ```
    ./main/tDCF_python_v1/score.sh
    ```


