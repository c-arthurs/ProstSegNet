#!/usr/bin/env bash

# This is used to detect if there is an output file present and change the name to OUTPUT_PROCESSED_TIME
# it will then create a new OUTPUT directory with all of the correct filenames

TIME=$(date +"%T")
echo "$TIME"

if [ -d "./OUTPUT" ] 
then
  echo "Directory OUTPUT exists."
  echo "moving ./OUTPUT to ./OUTPUT_PROCESSED"
  mv ./OUTPUT "./OUTPUT_PROCESSED_$TIME"
fi

mkdir ./OUTPUT
cd ./OUTPUT
mkdir acinar_nuclei acini acini_predictions nuclei nuclei_predictions result_spreadsheets stroma stromal_nuclei tissue_predictions
echo "OUTPUT dir created containing all subdirs"
