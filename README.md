# Towards Robust Evidence-Aware Fake News Detection via Improving Semantic Perception

## How to set up
* Python3.7 
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```


## Train models
You can simply run the bash script.
```shell script
sh train-cl.sh
```




## Test the performance of the trained models
* To test the performance of a trained model, run the command below:
```shell script
sh test.sh
```
You need to modify the filename path to the location where the trained model is stored.
By default, it is the datasetName_DAType. For example, "PolitiFact_gpt"





