#!/usr/bin/env python
# coding: utf-8

# # Receiver Operating Characteristic Curve Analysis

# index : The unique ID of each observation.
# 
# class : The true class of each observation. The classes are binary (0 or 1).
# 
# predicted_prob : The model's estimate of probability that the observation belongs to class 1.

# In[1]:


import pandas as pd
df = pd.read_csv('model_outcome.csv')


# In[2]:


df.head()


# ### Manually calculate the sensitivity and specificity of the model, using a predicted_prob threshold of greater than or equal to .5.

# In[3]:


# Set prediction threshold to greater than or equal to 0.5
df['prediction'] = df.predicted_prob.apply(lambda x : 1 if x>= 0.5 else 0)


# In[4]:


# Create a function where the conditionals determine the value of the outcome 
def classify(df):
    if (df['prediction'] == 1) & (df['prediction'] == df['class']):
        val = 'TP'
    elif (df['prediction'] == 1) & (df['prediction'] != df['class']):
        val = 'FP'
    elif (df['prediction'] == 0) & (df['prediction'] == df['class']):
        val = 'TN'
    else:
        val = 'FN'
    return val


# In[5]:


# Create column that contains the values of outcome 
df['outcome'] = df.apply(classify , axis=1)


# In[6]:


# Count Frequency of each outcome in 'outcome' column to calculate Sensitivity, Specificity
TP = (df['outcome'] == 'TP').sum() 
TN = (df['outcome'] == 'TN').sum()
FP = (df['outcome'] == 'FP').sum()
FN = (df['outcome'] == 'FN').sum()


# In[7]:


# Sensitivity and Specificity Formula
Sensitivity = TP / (TP + FN)
Specificity = TN / (FP + TN)


# In[8]:


print(Sensitivity, Specificity)


# ### Manually calculate the Area Under the Receiver Operating Characteristic Curve.

# In order to calculate the ROC curve, we need to find the True Positive Rate and False Positive Rate at different levels of threshold.
# 
# I used threshold increments of 0.1 from 0 to 1. 

# In[9]:


# Store TPR and FPR values
TPR_list = []
FPR_list = [] 


# In[10]:


#Create function where conditional determines the values of the outcome 
def classify(df,prediction_column,class_column):
    classify_list = []
    for row in range(len(df)):
        if (df.loc[row,prediction_column] == 1) & (df.loc[row,prediction_column] == df.loc[row,class_column]):
            classify_list.append('TP')
        elif (df.loc[row,prediction_column] == 1) & (df.loc[row,prediction_column] != df.loc[row,class_column]):
            classify_list.append('FP')
        elif (df.loc[row,prediction_column] == 0) & (df.loc[row,prediction_column] == df.loc[row,class_column]):
            classify_list.append('TN')
        else:
            classify_list.append('FN')
    return classify_list 


# In[ ]:





# In[11]:


def create_threshold_columns(df):
    for i in range(1,11):
        # Set threshold increment to 0.1
        threshold_i = i/10 
        
        # Create 'prediction_threshold_i' columns
        prediction = 'prediction_threshold_' + str(threshold_i)
        
        # Create 'outcome_threshold_i' column
        outcome = 'outcome_threshold' + str(threshold_i)
        
        # Store value 1 if x> threshold else 0 into each row of prediction_threshold_i column
        df[prediction] = df.predicted_prob.apply(lambda x: 1 if x > threshold_i else 0)
        
        # Create column that contains the values of outcome 
        df[outcome] = classify(df,prediction,'class')
        
        # Count Frequency of each outcome in 'outcome_threshold_i' column to calculate FPR, TPR 
        TP = (df[outcome] == 'TP').sum() 
        TN = (df[outcome] == 'TN').sum()
        FP = (df[outcome] == 'FP').sum()
        FN = (df[outcome] == 'FN').sum()
        
        # FPR and TPR formula 
        FPR = FP/(FP + TN)
        TPR = TP/(TP +FN)
        
        #Store calculated values into list
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    
    return TPR_list, FPR_list


# In[12]:


create_threshold_columns(df)


# ### Visualize the Receiver Operating Characterstic Curve.

# Plotted ROC Curve where False Positive Rate is the X-axis and True Positive Rate is the Y-axis.
# 
# Drew a line with slope 1 to represent pure guessing. 

# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[14]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[15]:


plot_roc_curve(FPR_list, TPR_list)


# In[ ]:




