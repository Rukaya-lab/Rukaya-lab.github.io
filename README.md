# Rukayat AMZAT


#### Portfolio


## Welcome to my GitHub Page

Here you can find some of the projects I have worked on. You can also check my [LinkedIn](https://www.linkedin.com/in/rukayat-amzat-889839173/) profile for further information.

## Machine Learning Projects


## [Querying and reading Multiple PDF files using OpenAI and Langchain](https://github.com/Rukaya-lab/OpenAI_ex/blob/main/Quering_Multiple_PDF_from_Vector_Store_using_Langchain.ipynb)

Background: With LLM prevalent and are trained on various human data, it poses the question of hallucination. What if we want to leverage the benefit of large language models but have our own specific knowledge base e.g PDF files.
Solution: Integrate the data sources into the OpenAI querying pipeline.
Approach: 
  - Using the Langchain library.
    - LangChain is an open-source framework designed for developing applications powered by a language model.
    - It provides the integration with OpenAI as the Large Language Model (LLM) and a PDF explorer library called UnstructuredPDFLoader.
  - The files are then converted to a documnet langchain object.
  - A vectorstore Index is used to convert the data in the documents to indexes that the LLM can understand.
    **VectorstoreIndexCreator:**
     There are three main steps going on in the background when the vectorstoreindex is used after the documents are loaded:
      - Splitting documents into chunks

      - Creating embeddings for each document

      - Storing documents and embeddings in a vectorstore
  - The created index can then be queried to return answers as found in the documents.

![](/images/query.png)




## [Smart Conversation Reply](https://github.com/Rukaya-lab/Smart-Reply-suggest)
- Problem: In today's fast-paced world, it can be difficult to keep up with all of the messages we receive. This can lead to missed opportunities, misunderstandings, and even stress.
- Solution: A smart reply system could help people to save time and improve their communication. By automatically suggesting relevant responses to incoming messages, a smart reply system could free up people to focus on more important tasks.
- Approach: A smart reply system could be implemented using a variety of techniques, including natural language processing (NLP) and long short-term memory (LSTM) models. NLP techniques could be used to identify the key words and phrases in a message, while LSTM models could be used to predict the most likely response.
  - Building Similarity indexes with the ANNOY library.
  - Empolying Hdbscan clustering to cluster similar replies.
  - LSTM since they have memory and able to keep context.


#### The Data

[Data can be found here](https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)

This is a Topical Chat dataset from Amazon! It consists of over 8000 conversations and over 184000 messages.

#### Steps

1. Wrangling
  - Since the dataset was originally compiled for some other text calssification task, I had to re process and collect only the information that is necessary for the project.

2. Tokenization of the input and target texts to convert the input and target texts to sequences of integers. 

3. Creating Annoy Index for both the Input and Target Texts using Annoy algorithm is to find the nearest neighbors of a given text.

4. Clustering using the similarity matrix for both input and target texts.
  - Tested both the hdbscan algorithm and dbscan algorithm using different epsilon value while finding the optimal value for the epsilon parameter.

5. Generate padded sequences from a list of input sequences.  
  - The padding ensures that all input sequences are the same length, which can improve the performance of the model.

6. Modeling with LSTM
  The model is a sequential model with the following layers:

    - An embedding layer that converts the input text into a sequence of dense vectors.
    - A long short-term memory (LSTM) layer that learns long-range dependencies in the input text.
    - A dropout layer that randomly drops out some of the neurons in the model, which helps to prevent overfitting.
    - A dense layer that outputs the predicted probability distribution over the possible target labels.

#### Result
  The model achieved a loss of 0.1177 and  accuracy of 97%.

Example usage
   Input - bye

    Response 1 -> bye  
    Response 2 -> have a good one  
    Response 3 -> same to you  
    Response 4 -> have a good weekend  

#### Potential Stonewall
  - The similarity index building is quite large and takes time to create hence you need GPU access.
  - Since the texts have been stripped of punctuations in the preprocessing, one needs to find a way to add them back.

#### Next Steps
  - A dataset can be built speciafically for this project that will be robust enough to capture conversation complexities. The current dataset contains some converstaions that are topic specific.
  - I tried to add back the punctation using the rpunct library i found on Hugging face but the libraray hasn't been updated for a while and hence didn't captuer all punctions. So a work could be done focused on that.
  - The clustering method can be improved.
  - Deploying the model.




## [Detecting Hate Speech in Tweets](https://github.com/Rukaya-lab/NLP-notebooks/blob/main/Detecting%20Hate%20speech%20in%20tweet.ipynb)

- Project: Classification of Tweets into Hateful speech or Normal Speech
- Context: Hate speech is defined as public speech that expresses hate, disparages, or incites or encourages violence against another individual or group based on some characteristic such as race, gender, religious belief, sexual orientation, etc. 
- Dataset: The dataset contains 5,232 total rows of archived tweets with 3000 as non-hateful speech and 2242 hateful speech.
- Steps I took for the task
  - Removing stop words and special characters
  - Vectorization

![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/3d469d62-5828-48a7-906f-f69384b179b2)
- Result
  - Classification using the Guassian Naives Bayes Model with an Accuaracy of 60% and,

  - Classification after feature engineering with Count Vectorizer with Multinomial Naive Bayes gave an accuracy 86%.
  
 ![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/75c2deb1-8c7d-488a-a9c0-f2671e01ebb2)



## [Face Recognition with Siamese Network](https://github.com/Rukaya-lab/Face-Verification-with-Siamese-Network-and-Kivy-App)
Project: Detect and recognize human faces in images or via video streams.
Model Architecture: Siamese Convolutation Neural Network.
  - A Siamese network is a type of neural network (Convolutionall Ne) that consists of two or more identical networks that are trained together. The networks are trained to learn a shared representation of the input data, and they are then used to compare pairs of inputs and determine whether they are the same or different.
#### The Data

For this project there are two sets of Dataset.
One is a public data set of different labelled faces [Labelled Faces](http://vis-www.cs.umass.edu/lfw/), 
and the other dataset is that of my faces at different angles and augmentations.

- The public dataset id labelled as the Negative dataset and my own images are bothe the positive data and the Anchor images.
    - open Cv libarary is used to collect the images from video capture.
    - The images collected were augmneted to create more samples of different properties to enable for a larger dataset.
    - All the collected images were resized to have a unifirm dimension of 100 by 100 
- The anchor images are then used with the positive and negative dataset to craete a paired and labeled dataset.
    - match a positive with an anchor and takes value of 1 since it the same person.
    - match negative and anchor and that will take a 0 since the two people wouldnt match.
    - That generated 6000 samples of paired data. 70% was for training and 30% for testing.      

#### The Model

##### Embedding Layer
- There are two embedding layer built. The Input embedding layer for the input images nad the Validation embedding layer for the validation images.
- Each model layer consist of:
    - The model consists of four convolution blocks, each followed by a max pooling layer. 
    - The convolution blocks use 64, 128, 128, and 256 filters, respectively. 
    - The max pooling layers use a pool size of 2x2 and a stride of 2. 
    - The final layer of the model is a dense layer with 4096 neurons. 
    - The activation function for all layers is ReLU, except for the final layer, which uses a sigmoid activation function.

### Distance Layer
A distance layer is built to calculate the L1 distance between two embeddings. 
- The L1 distance is a measure of the difference between two vectors. It is calculated by taking the absolute difference between the corresponding elements of the vectors.
- The distance is the the difference between the inpu embeddinga and the validation embedding.

** The Siamese network is then the amalgamtion of the Distance layers, embeddings and images. **

- The loss function is the Binary Cross Entropy.
- The optimizer is Adam.
- The metric is Accuracy, Precison, Recall
- The model achived a precision and recall of 1.0.

![](/images/verified.jpg)

### UI with Kivy
- Created a base UI app to interact with the model.
![](/images/kivy.png)




## [Segmentation and Classification of fishes using Images](https://www.kaggle.com/rukayaamzat/91-accuracy-cnn-for-fish-classification)
- Project: Augmentation and Classification of the fishes based on images to their correct class using Sequential model.
- The dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey for a university-industry collaboration project at Izmir University of Economics.   
    The dataset contains 9000 images. For each class, there are 1000 augmented images and their pair-wise augmented ground truths.
- Approach:
  - Keras Image Data generation was used for Image preprocessing and all images are resized to a target size of (224, 224).
  - A convolution layer with filter size of 32 and a Max pool layer with a pool_size of (2, 2) was included in the sequential model.
  - Drop out of 30% was included to avoid overfitting.
  - The trained model achieved an accuracy of 92% and a recall of 98%.
- Conclusion: The model really performed well on this partcular dataset and has achieved a low loss value for both the train and validation set. The recall and precsion is also quite high and there doesn't seem to be any over fitting.


![](/images/fish_loss.png)

##### The model performed quite well with predicting the fish classes.

![](images/fish_class.png)




## [Heart Disease Prediction](https://github.com/Rukaya-lab/Tensorflow-practice/blob/main/Dp_Classification%2C_heart_disease.ipynb)
- Project: Prediction whether a patient has heart disease based on their medical vitals.
- Dataset: The dataset contains the age, sex, blood pressure, heart rate, blood sugar etc values. It has 14 columns and about 300 data samples
- Approach: For this project, I decided to explore feature selection and behaviour of the model when couple of the features are left out.
  - Trained first a base sequential model that achieved a training loss of 0.37 and accuracy of 82% and validation loss of  0.54 and accuracy of 79%.

![](/images/loss_without.png)

  - Use the Recursive Feature Elimination approach with random forest model as the base model to slecet the top features that affect the models performance.

![](/images/feature_selected.png)

  - Trained another sequential model that achieved a training loss of 0.48 and accuracy of 78% and validation loss of 0.45 and accuracy of 84%.

![](/images/loss_with.png)

- Findings: Even though the accuracy of the train set didn't experience any significant positive change, it is worthy noting that with a few of the features the model was still   able to learn enough of the operations of the dataset and perform fairly while and the loss difference is set to low. On the flip side the validation dataset perfromed better when the features were selected which can then translate to the fact that any unseen dataset tested with the second model is going to be close to its ground truth.




## [Credit Card Lead Prediction](https://github.com/Rukaya-lab/Project-/blob/main/Credit_card_interest_classification.ipynb)

- Project: Happy Customer Bank is a mid-sized private bank that deals in all kinds of banking products, like Savings accounts, Current accounts, investment products, credit products, among other offerings.
- Task: Identifying customers that could show higher intent towards a recommended credit card
- Dataset description: 
  - The training dataset contains 245725 rows, 11 feature columns
  - The dataset is an unbalanced dataset,  the traget column contains 187437 Customer is not interested and 58288 Customer is interested.
  
 ![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/46454917-682a-4d91-a7b3-ae8096d20c6f)

 - Approach
   - Label Encoder for ordinal categorical columns and one hot encoder for nominal variables.
   - The dataset is an unbalanced dataset, the traget column contains 187437 Customer is not interested and 58288 Customer is interested.
      Hence, I used the SMOTE from imblearn.over_sampling library to upsample the class
   - Preprocessed the datset over a standard scaler to bring the distribution closer
   - Built the ML model using voting classifier which uses logistic regression and random forest as the two base sub model.
      The voting classifier gave an accuracy of 0.77, the logistic model gave an accuracy of 0.75 and the random forest model with an accuracy of 0.76
   - Built a Xgboost Model that gave 0.76 accuaracy and F1 score of 0.84
   - Built a basic tensorflow Sequential model and avoided overfitting by calling early stopping.
     The model gave and accuracy of: 0.8580 - mean squared error(mse) of : 0.0968 - and validation accuracy of : 0.7826 - validation_mse: 0.1465
     
   ![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/21d7a2bf-be91-4abf-81f8-7754bdb6bc59)




## Data Analaysis and Exploration Projects


## [Wrangle and Analyze Data: We Rate Dogs](https://github.com/Rukaya-lab/Project-/blob/main/WeRateDogs/wrangle_act.ipynb)

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



## [Investigate a Dataset: No Show Appointment](https://github.com/Rukaya-lab/Project-/blob/main/Investigate_a_Dataset.ipynb)

- Project: Medical Appointment No Shows. Why do 30% of patients miss their scheduled appointments?
- Context: A person makes a doctor appointment, receives all the instructions and no-show. Who to blame?
- Dataset: This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
- Exploration:
  - People who do not have Diabetes have a average show up rate of 80% while people with Diabetes have a average show up rate of 82%.
  - There are significant number of female records as compared to the male records.
  - In teenagers, less than 70% of those with hypertension tend to show up, which is less than average of all across the ages.
  - Possible Limitations could include the geography of the hospitals and residences of the patients and how there could be possible commuting problems that could affect show up rate.

![image](https://github.com/Rukaya-lab/Rukaya-lab.github.io/assets/74497446/c5add26d-8aa8-4134-9d36-9f692b5e20af)


