#!/usr/bin/env python
# exported from interal repository path generative_api/notebooks/experiments/20250217-IntentDetectionPaperAnalysis.ipynb
# primarily contains code for computing summary statistics

assert (
    False
), "This script contains the contents of an internal notebook, and is provided for reference but not for execution."

# In[1]:


import dotenv

assert dotenv.load_dotenv("../../../.env", override=True)


# In[2]:


import os

os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
assert "sqlite" not in os.environ["DATABASE_URL"]


# In[3]:


from datetime import datetime, timedelta, timezone
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import sklearn.metrics

import rori_orm
from rori_orm.django.content import models, db_utils
import django.db.models
from generative_api.rori_generative_api import cms_utils


# In[4]:


activity = models.Activity.objects.get(name="TopicalConversation")
activity.name


# In[5]:


data_dir = Path("../../../data")
assert data_dir.exists()


# In[6]:

# load annotations
df = pd.read_csv(data_dir / "raw" / "exit_classifier_merged_20250217.csv")
df.head(1)


# In[7]:


df.id.nunique(), len(df)


# In[8]:


df.activity_session_id.nunique()


# In[9]:


df["Given Label **Zachary"].value_counts(dropna=False)


# In[10]:


df["Given Label **Ella"].value_counts(dropna=False)


# In[11]:


raw_annotations = df[["Given Label **Zachary", "Given Label **Ella"]]
mutual_annotations = raw_annotations.dropna().reset_index(drop=True)
len(mutual_annotations)


# In[12]:


def standardize_raw_annotations(annotation: str) -> str:
    if annotation.endswith("?"):
        return annotation[:-1]
    return annotation


mutual_annotations["zachary"] = mutual_annotations["Given Label **Zachary"].map(
    standardize_raw_annotations
)
mutual_annotations["ella"] = mutual_annotations["Given Label **Ella"].map(
    standardize_raw_annotations
)
same_label = mutual_annotations["zachary"] == mutual_annotations["ella"]
same_label.sum(), same_label.sum() / len(same_label)


# In[13]:


sklearn.metrics.cohen_kappa_score(
    mutual_annotations["ella"], mutual_annotations["zachary"]
)


# In[14]:


def get_label(row: pd.Series) -> str:
    if not pd.isna(row["Consensus Label"]):
        return row["Consensus Label"]
    elif pd.isna(row["Given Label **Zachary"]):
        return row["Given Label **Ella"]
    return row["Given Label **Zachary"]


# In[15]:


df["label"] = df.apply(get_label, axis=1)
df.label.value_counts(dropna=False)


# In[16]:


n_e = (df.label == "e").sum()
n_e, f"{n_e / len(df):.2%}"


# In[17]:


df[df.label == "e"][["message_id", "text"]].sample(n=30)


# In[18]:


message_ids = df.message_id.tolist()
messages = models.Message.objects.filter(id__in=message_ids).select_related(
    "activity_session", "activity_session__user"
)
len(messages)


# In[19]:


mdf = pd.DataFrame([m.__dict__ for m in messages])
mdf.head(1)


# In[20]:


mdf.created_at.min(), mdf.created_at.max()


# In[ ]:


# In[21]:


# how many messages and conversations were there during this time?
# i.e. how big is the sample relative to the full pool?
start_datetime = mdf.created_at.min().to_pydatetime()
end_datetime = mdf.created_at.max().to_pydatetime()

all_messages = (
    models.Message.objects.filter(
        activity_session__activity=activity,
        created_at__gte=start_datetime,
        created_at__lte=end_datetime,
        direction=models.Message.MessageDirection.INBOUND,
        activity_session__user__properties__turn_line_id="+12065906259",
    )
    .exclude(text="")
    .select_related("activity_session")
)
ds = []
for m in all_messages:
    d = m.__dict__
    d["topic"] = m.activity_session.properties["topic"]
    ds.append(d)
mfdf = pd.DataFrame(ds).drop(columns=["_state"])
mfdf.shape


# In[22]:


mfdf["id"].nunique(), mfdf.activity_session_id.nunique()


# In[23]:


mfdf.topic.value_counts()


# In[24]:


mfdf.groupby("activity_session_id").agg({"topic": "first"}).value_counts()


# In[25]:


len(mdf) / len(mfdf)


# In[27]:


users = set()
for message in messages:
    users.add(message.activity_session.user)
len(users), len(messages)


# In[41]:


ages = np.array(
    [
        int(user.properties["age"])
        for user in users
        if "age" in user.properties and user.properties["age"] is not None
    ]
)
countries = [
    user.properties["country"]
    for user in users
    if "country" in user.properties and user.properties["country"] is not None
]
len(ages), len(countries)


# In[40]:


np.mean(ages), np.median(ages)


# In[35]:


pd.Series(countries).value_counts()


# In[36]:


(86 + 44) / 158


# In[37]:


pd.Series(ages).value_counts()


# In[39]:


(ages < 18).sum() / len(ages)
