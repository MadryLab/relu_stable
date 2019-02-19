Code for **Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability**
_Kai Xiao, Vincent Tjeng, Nur Muhammad Shafiullah, Aleksander Madry_
https://arxiv.org/abs/1809.03008
International Conference on Learning Representations (ICLR), 2019

### Workflow

See `example.ipynb` for an interactive version of the following instructions.

**Model Training**
1. `python train.py`: trains a model using parameters in `config.json`.

      Description of the defaults in `config.json`:
      
            1. It trains a 3-hidden layer convolutional architecture on MNIST.
            
            2. It uses adversarial training, L1 regularization, and ReLU stability regularization.
            
            3. The model is saved in `trained_models/relu_stable`

2. `python post_process_model.py --model_dir $MODELDIR --output $MATNAME`: apply post-processing, converting the model from $MODELDIR to a .mat file and saving it as `model_mats/$MATNAME.mat`.

      Command-line flags are available to choose post-processing options. Type `python post_process_model.py -h` to see all options.
      
**Verification**

In theory, you can use any verification procedure here; this code repo is set up to use the same verifier as in the paper.

**Requires installation of the Julia package of https://github.com/vtjeng/MIPVerify.jl, as well as the Gurobi solver. Details are in the linked github repo.**

3. `./verification/verify.sh $MATNAME $EPS ($START_INDEX) ($END_INDEX)` to verify robustness of the saved .mat file against L_infinity perturbations with norm-bound `$EPS`. The script automatically does the following two things:
      
      A. Runs `julia verification/verify_MNIST.jl` - outputs a `summary.csv` to the folder `./verification/results/$MATNAME__linf-norm-bounded-$EPS__Inf__0.0/` for step 4
      
      B. Writes the console output to the log file `./verification/logs/$MATNAME.log`  for step 5

      You most likely want to set `$EPS` to the epsilon specified in `config.json` during training. Setting `$START_INDEX=1` and `$END_INDEX=10000` verifies the entire MNIST test set.

4. `python parser/parse_csv.py --csv_name $CSVPATH`: parses the .csv file to get provable adversarial accuracy and solve times
5. **(OPTIONAL)** `./parser/convert_log.sh $MATNAME` follwed by `python parser/parse_log.py --log_name $MATNAME`: parse the logs to get exact ReLU stability numbers



## Citing this Code
```
@article{xiao2019training,
  title={Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability},
  author={Xiao, Kai and Tjeng, Vincent and Shafiullah, Nur Muhammad and Madry, Aleksander},
  journal={ICLR},
  year={2019}
}
```
