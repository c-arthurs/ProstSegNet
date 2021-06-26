# ProstSegNet

Keras unet and then chromogen analysis for prostate histopathology images.

## Repo layout 

``` bash
.
├── predict_calculate_dab # scripts for the prediction of histo images and chromogen analysis
│   ├── new_output_dir.sh # bash script to create the output directories
│   └── run_all_unet_predictions.py # runs precictions on each TA image and calculates DAB measurements
└── train_model # scripts for training the 3 unet models that are used
    ├── checkfiles.py # for checking the dataset for corrupt files
    └── unet.py # for trianing vanilla unet model 
```

