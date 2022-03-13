# SecureCovid

Implementation of secure covid project described in <a href="https://drive.google.com/file/d/1Odv8z8mtWyKuNxhsem-cj6CA3QCb_QnS/view?usp=sharing">propal</a>.

## Shadow Model

To train, first prepare for the image training data and run

<code>python shadow_train.py --data_path path/to/file</code>

```
usage: shadow_train.py [-h] [--data_path DATA_PATH] [--out_path OUT_PATH]
                       [--weight_path WEIGHT_PATH] [--mode MODE]
                       [--model MODEL] [--valid_size VALID_SIZE]
                       [--learning_rate LEARNING_RATE] [--step_size STEP_SIZE]
                       [--gamma GAMMA] [--epoch EPOCH] [--name NAME]

Secure Covid Shadow Train

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to store the data
  --out_path OUT_PATH   Path to store the trained model
  --weight_path WEIGHT_PATH
                        Path to load the trained model
  --mode MODE           Select whether to train, evaluate, inference the model
  --model MODEL         Select which model to use
  --valid_size VALID_SIZE
                        Proportion of data used as validation set
  --learning_rate LEARNING_RATE
                        Default learning rate
  --step_size STEP_SIZE
                        Default step size
  --gamma GAMMA         Default gamma
  --epoch EPOCH         epoch number
  --name NAME           Name of the model
```



## Attack Model

To train, first prepare for the input data and output data and run

<code>python attack_train.py</code>

```
sage: attack_train.py [-h] [--input_path INPUT_PATH]
                       [--target_path TARGET_PATH] [--out_path OUT_PATH]
                       [--weight_path WEIGHT_PATH] [--mode MODE]
                       [--label LABEL] [--valid_size VALID_SIZE]
                       [--learning_rate LEARNING_RATE] [--epoch EPOCH]
                       [--name NAME]

Secure Covid

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to store the data
  --target_path TARGET_PATH
                        Path to store the data
  --out_path OUT_PATH   Path to store the trained model
  --weight_path WEIGHT_PATH
                        Path to load the trained model
  --mode MODE           Select whether to train, evaluate, inference the model
  --label LABEL         Select the label for the attack model
  --valid_size VALID_SIZE
                        Proportion of data used as validation set
  --learning_rate LEARNING_RATE
                        Default learning rate
  --epoch EPOCH         epoch number
  --name NAME           Name of the model
```

