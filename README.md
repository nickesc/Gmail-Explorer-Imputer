# Gmail Explorer Imputer

###### By [nickesc](https://github.com/nickesc) / [N. Escobar](https://nickesc.com)

Now that we have data from the last notebook, we can start to analyze it! The first thing we need to do is load in our data:


```python
import os
import csv
import base64
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import datetime


from IPython.display import clear_output, display
#from ipywidgets import *
#from tkinter import Tk, filedialog
#from math import floor
```


```python
messages = pd.read_csv("../messages.csv")
messages.head()

```


```python
fig, ax = plt.subplots(figsize=(10,10))
display(sns.heatmap(messages.isnull(),yticklabels=False, cbar=False,
           cmap="Blues",ax=ax))
```


```python
inboxes = pd.DataFrame()
chats = pd.DataFrame()
emails = pd.DataFrame()
```

## Turning our data into something useful

Now, we can start to actually look at the data and come to some conclusions. First though, even though we made the data set, we still need to do some cleanup. It's still, for the most part, the same way we got it from google, just put together a little nicer. We haven't really had to fill in or change the data yet. 

The `internalDate` is useful for tracking trends over time, but doesn't really tell us when it actually happened, so we need to convert that to `dateTime`. We also want to adjust our `from` column, which tells us the name and the address, and we really only want the address, because not all senders have a name. We also need to take care of all the `NaN`s in the data, which mostly come from weird or missed headers, but are mostly fixable too.


```python
def convertTime(epochtime):
    thetime = datetime.datetime.fromtimestamp(epochtime/1000)
    return(thetime)
def convertYear(epochtime):
    thetime = datetime.datetime.fromtimestamp(epochtime/1000)
    return(thetime.year)
def convertMonth(epochtime):
    thetime = datetime.datetime.fromtimestamp(epochtime/1000)
    return(thetime.month)
def convertDay(epochtime):
    thetime = datetime.datetime.fromtimestamp(epochtime/1000)
    return(thetime.day)

nans=[]
def convertAddress(string):
    try:
        address = string.split('<')[-1].split('>')[0]
    except:
        nans.append(string)
        address = str(string)
    return address
def convertLabels(labels):
    string=""
    x=0
    for label in labels:
        if x==0:
            string=str(label)
            x+=1
        else:
            string=string+","+str(label)
    return labels
    
    #return labels.replace("'", "").strip('][').split(', ')


inboxes['id'] = messages["id"]
inboxes['threadId'] = messages["threadId"]
inboxes["from"] = messages["from"].apply(convertAddress)
inboxes['delivered'] = messages["delivered-to"]
inboxes['to'] = messages["to"].apply(convertAddress)
inboxes['internalDate'] = messages["internalDate"]
inboxes["dateTime"] = messages["internalDate"].apply(convertTime)
inboxes["year"] = messages["internalDate"].apply(convertYear)
inboxes["month"] = messages["internalDate"].apply(convertMonth)
inboxes["day"] = messages["internalDate"].apply(convertDay)
inboxes["labels"] = messages["labels"].apply(convertLabels)
inboxes['sizeEstimate'] = messages["sizeEstimate"]
inboxes['subject'] = messages["subject"]
inboxes['body'] = messages['body']

inboxes["labels"].describe()
```

### Received vs. Chats vs. Drafts vs. Sent

Before we handle `NaN`s, we're going to split off the Google Hangouts chats. These, though not really emails, still appear as messages in your inbox. The problem is that they, in addition to drafts and sent emails, throw off a lot of the other metrics, especially because there are so many chats, including the onces I sent. The chats lack two of the addresses addresses or a subject for the most part, making them confusing blanks spots in the table (the big hole in the center on the heatmap at the top). Some of them you could figure out, but the ones I sent only have my name attached, not who I sent them to, so I decided to skip them altogether, since they're already peripheral. Drafts and sent messages, however, we can still figure out the missing information in the same way we do received messages, so we'll leave those in for now. 


```python
chats = inboxes[(inboxes['labels'].str.contains('CHAT')) == True].copy(deep=True)
emails = inboxes[(inboxes['labels'].str.contains('CHAT')) == False].copy(deep=True)
display(chats,emails)
```


```python
fig, ax = plt.subplots(figsize=(10,10))
display(sns.heatmap(inboxes.isnull(),yticklabels=False, cbar=False,
           cmap="Blues",ax=ax))
```

### Missing Values

Then there are a lot of seemingly random missing values. Mostly, they're one of the three addresses -- `from`,`delivered` and `to`. The `from` address tells you the source, the `to` address is who it was addressed to, and the `delivered` address is the email it was delivered to. The strangest, by far, is that for some reason between August 2019 and April 2020, the World Wild Life Fund was emailing me from an account without any address attached to it. The email seemed  like it was coming out of no where on the table, but I managed to find the subjects of the emails in the Gmail GUI and match them with the sender. Those are the only emails that I had track down the sender, otherwise I could fill pretty easily the `to` and `delivered` columns using each other, as they're almost always the same thing.


```python
emails.loc[(emails['from'] == "nan") & (emails["subject"]!=""), 'from'] = 'ecomments@wwwfus.org'
#emails.loc[(emails['labels'].str.contains('SENT')) == True, 'delivered'] = '???'
emails.loc[(emails['labels'].str.contains('DRAFT')) == True, 'delivered'] = '{{not delivered}}'
#emails.loc[(emails['labels'].str.contains('DRAFT') & emails['to']=="nan") == True, 'to'] = '{{not addressed}}'
emails['to']=np.where((emails['labels'].str.contains('DRAFT') & emails['to']==np.NaN) == True,emails['delivered'],emails['to'])
emails["delivered"] = np.where((emails['labels'].str.contains('SENT')) == True, emails["to"], emails["delivered"])
fig, ax = plt.subplots(figsize=(10,10))
display(sns.heatmap(emails.isnull(),yticklabels=False, cbar=False,
           cmap="Blues",ax=ax))
```


```python
emails.replace("nan",np.NaN, inplace=True)
emails.replace("",np.NaN, inplace=True)
emails.sort_values("delivered")
fig, ax = plt.subplots(figsize=(15,30))
display(sns.heatmap(emails.isnull(),yticklabels=False, cbar=False,
           cmap="Blues",ax=ax))
```


```python
emails.sort_values("to")
```


```python
#emails["delivered"] = np.where((emails['delivered'] == np.NaN), emails["delivered"], emails["to"])
emails.delivered.fillna(emails.to, inplace=True)
emails.to.fillna(emails.delivered, inplace=True)
emails.sort_values("delivered")

```

Next we fill the empty subject and body cells with a value to indicate they're empty, and lower all the addresses to be sure they're consistent when we're comparing them.


```python
emptyBody=base64.urlsafe_b64encode("{{empty message body}}".encode('utf-8'))
emptySubject="{{no subject}}"

emails.replace("undisclosed-recipients:;","{{undisclosed-recipients}}")

def lowerIt(x):
    return x.lower()

emails["from"].apply(lowerIt)
emails["delivered"].apply(lowerIt)
emails["to"].apply(lowerIt)

emails["body"].fillna(emptyBody, inplace = True)
emails["subject"].fillna(emptySubject, inplace = True)
```


```python
fig, ax = plt.subplots(figsize=(10,10))
display(sns.heatmap(emails.sort_values("delivered").isnull(),yticklabels=False, cbar=False,
           cmap="Blues",ax=ax))
nans=emails[emails['delivered'].isnull()].index.tolist()
```

### Exporting the Data


```python
sent = emails[(emails['labels'].str.contains('SENT')) == True].copy(deep=True)
drafts = emails[(emails['labels'].str.contains('DRAFT')) == True].copy(deep=True)
#chats = emails[(emails['labels'].str.contains('CHAT')) == True].copy(deep=True)

recieved = emails[(emails['labels'].str.contains('SENT') == False) &  (emails['labels'].str.contains('DRAFT') == False) & (emails['labels'].str.contains('Chats') == False)].copy(deep=True)


display(recieved.head(),sent.head(), drafts.head(),chats.head())
```


```python
recieved.sort_values('internalDate').reset_index(drop=True).to_csv('imputed/recieved.csv')
sent.sort_values('internalDate').reset_index(drop=True).to_csv("imputed/sent.csv")
drafts.sort_values('internalDate').reset_index(drop=True).to_csv('imputed/drafts.csv')
chats.sort_values('internalDate').reset_index(drop=True).to_csv('imputed/chats.csv')
```

Finally, we pull all our various `DataFRames` together and output four `.csv` files, for recieved messages, for sent messages, for drafts, and for chats, each of which is a little different to look at.
