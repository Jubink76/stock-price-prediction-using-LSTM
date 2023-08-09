#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern
# # Task 1- stock price prediciton with LSTM
# 

# ### importing required libraries

# In[1]:


import pandas as pd


# In[2]:


pip install yfinance


# In[3]:


import yfinance as yf


# In[4]:


pip install datetime


# In[5]:


from datetime import date, timedelta


# In[6]:


today = date.today()


# In[7]:


d1 = today .strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2


# In[8]:


data = yf.download('AAPL',
                      start=start_date,
                       end=end_date,
                       progress=False)
data["Date"] = data.index


# In[9]:


data = data[["Date","Open","High","Low","Close","Adj Close","Volume"]]
data.reset_index(drop=True, inplace = True)
data.tail()


# A candlestick chart gives a clear picture of the increase and decrease in stock prices

# In[10]:


import plotly.graph_objects as go


# In[11]:


figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                       open=data["Open"],
                                       high=data["High"],
                                       low=data["Low"],
                                       close=data["Close"])])


# In[12]:


figure.update_layout(title = "Apple Stock Price Analysis",
                    xaxis_rangeslider_visible=False)


# In[13]:


correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))


# Now i will start with training an LSTM model for predicting stock prices. I will first split the data into training and test sets:

# In[14]:


x = data[["Open","High","Low","Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[17]:


#now i will prepare a neural network architecture for LSTM


# In[18]:


pip install keras


# In[19]:


pip install tensorflow


# In[20]:


from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


# In[21]:


# Now here is how we can train our neural network model for stock price prediction.


# In[22]:


model.compile(optimizer = 'adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size = 1, epochs = 30)


# ### now let's test this model bu giving input values according to the features that we have used to train this model and predicting the final result

# In[26]:


import numpy as np
# features = [open, High, Low , Adj Close, Volume]
features = np.array([[194.669998, 196.630005,194.139999, 195.830002,48291400]])


# In[27]:


model.predict(features)


# In[ ]:




