# Rukaya-lab.github.io
Portfolio

# Welcome to my GitHub Page

Here you can find some of the projects I have worked on. You can also check my [LinkedIn](https://www.linkedin.com/in/rukayat-amzat-889839173/) profile for further information.

## Machine Learning Projects


## Natural Language Processing

### [Detecting Hate Speech in Tweets]()

- Project: Classification of Tweets into Hateful speech or Normal Speech
- Context: Hate speech is defined as public speech that expresses hate, disparages, or incites or encourages violence against another individual or group based on some characteristic such as race, gender, religious belief, sexual orientation, etc. 
- Dataset: The dataset contains 5,232 total rows of archived tweets with 3000 as non-hateful speech and 2242 hateful speech.
- Steps I took for the task
  - Tokenization
  - Normalization(Stemming or lemmatization)
  - Removing stop words
  - Vectorization
- Result
  - Classification using the Guassian Naives Bayes Model with an Accuaracy of 60% and 86% when calssified after vectorization.

![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/3d469d62-5828-48a7-906f-f69384b179b2) ![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/75c2deb1-8c7d-488a-a9c0-f2671e01ebb2)


## Deep Learning- Tensorflow
## Classification

## Data Analaysis and Exploration Projects

### [Investigate a Dataset: No Show Appointment](https://github.com/Rukaya-lab/Project-/blob/main/Investigate_a_Dataset.ipynb)

- Project: Medical Appointment No Shows. Why do 30% of patients miss their scheduled appointments?
- Context: A person makes a doctor appointment, receives all the instructions and no-show. Who to blame?
- Dataset: This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
- Exploration:
  - People who do not have Diabetes have a average show up rate of 80% while people with Diabetes have a average show up rate of 82%.
  - There are significant number of female records as compared to the male records.
  - In teenagers, less than 70% of those with hypertension tend to show up, which is less than average of all across the ages.
  - Possible Limitations could include the geography of the hospitals and residences of the patients and how there could be possible commuting problems that could affect show up rate.

![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/c5add26d-8aa8-4134-9d36-9f692b5e20af)


### [Wrangle and Analyze Data: We Rate Dogs](https://github.com/Rukaya-lab/Project-/blob/main/WeRateDogs/wrangle_act.ipynb)

- Project: Wrangle data from different sources and Analyze.
  WeRateDogs is a Twitter account that posts and rates pictures of dogs. 
- Dataset:
Gathring the dataset from different data sources including:
  - A csv data file directly from archived data 
  - Using the Requests library to download the tweet image prediction
  - Using the Tweepy library to query additional data via the Twitter API
- Task:
  - detecting and cleaning quality issues and tidiness issue with the data.
    - There are significant number of missing values in the twitter archive dataset
    - The name column has a lot of duplicated values and some names are None
    - Tweet ids are currently in integer but they should be strings
    - The timestamp is in integer format rather than datetime
    - The dog stages are in separate columns but they should be in one
    - The three dataset have tweet id in common, hence could be combined
 - Exploration:
    - The most common dog breed is the Labrador Retriever.
    - The highest number of retweet is recorded in 2016
    - The most common name is Charlie followed by Cooper, Lucy and Oliver with the same count

 
 ![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/9f20079b-ea18-4dd2-873d-c30509e85d32)
