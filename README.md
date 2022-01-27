# Deep-Neyman-Scott-Processes
## Reference
[Deep Neyman-Scott Processes](https://arxiv.org/pdf/2111.03949.pdf) by Chengkuan Hong and Christian R. Shelton, AISTATS 2022.
## Instructions
### Dependency
- Anaconda (Python >= 3.8)
### Data
The data for retweets can be downloaded from [Google Drive Link](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing)

The data for earthquakes and homicides can be downloaded from [Google Drive Link](https://drive.google.com/drive/folders/1ELuYM9qIj2hoSzJcYAs9UklcwO2WdTNu?usp=sharing)
### Train

Training of 1-hidden for earthquakes
```
python earthquake_train_1_hidden.py
```
Traning of 2-hidden for earthquakes
```
python earthquake_train_2_hidden.py
```
You can do the same things for retweets and homicides.

### Test and prediction

For example, you can calculate the log-likelihood and do the prediction for 1-hidden for the first sequence with the following code. 0 represents the first sequence in the test dataset. 
You can replace 0 with another integer.
```
python earthquake_prediction_1_hidden.py -e 0
```
For 2-hidden,
```
python earthquake_prediction_2_hidden.py -e 0
```

After collecting the results for all the sequences, you can run the following code to get the results reported in the paper.
```
python prediction_result.py
```
## Disclaimer
This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago.  The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site.  The data provided at this site is subject to change at any time.  It is understood that the data provided at this site is being used at one’s own risk.
