import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_json()

# df.info()


# Training dataset
# Users should be the current user and users they follow
class PostsDataset:
    def __init__(self, users, posts, likes):
        self.users = users
        self.posts = posts
        self.likes = likes


    def __len__(self):
        return len(self.users)
    
    
    def __getitem__(self, item):
        users = self.users[item]
        posts = self.posts[item]
        likes = self.likes[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "posts": torch.tensor(posts, dtype=torch.long),
            "likes": torch.tensor(likes, dtype=torch.long),
        }
    

# Model class
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_posts):
        super().__init__(n_users, n_posts)

        self.user_embed = nn.Embedding(n_users, 32)
        self.post_embed = nn.Embedding(n_posts, 32)
        self.out = nn.Linear(64, 1)

    def forward(self, users, posts, likes = None):
        user_embeds = self.user_embed(users)
        post_embeds = self.post_embed(posts)
        output = torch.cat([user_embeds, post_embeds], dim=1)

        output = self.out(output)

        return output

    
# Setting up datasets
label_user = preprocessing.LabelEncoder()
label_post = preprocessing.LabelEncoder()
df.userId = label_user.fit_transform(df.userId.values)
df.postId = label_user.fit_transform(df.postId.values)

df_train, df_valid = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.likes.values
)

train_dataset = PostsDataset(
    users=df_train.userId.values,
    posts=df_train.postId.values,
    likes=df_train.likes.values,
)

valid_dataset = PostsDataset(
    users=df_train.userId.values,
    posts=df_train.postId.values,
    likes=df_train.likes.values,
)

# Loading data from datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

dataiter = iter(train_loader)
dataloader_data = next(dataiter)
print(dataloader_data)

model = RecSysModel(
    n_users=len(label_user.classes_),
    n_posts=len(label_post.classes_)
).to(device)

optimizer = torch.optim.Adam(model.parameters())
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

loss_func = nn.MSELoss()

# Checking lengths and bounds
print(len(label_user.classes_))
print(len(label_post.classes_))
print(df.postId.max())
print(len(train_dataset))

with torch.no_grad:
    model_output = model(dataloader_data['users'], dataloader_data['posts'])

    print(f"model_output: {model_output}, size: {model_output.size()}")

likes = dataloader_data["likes"]
print(likes)
print(model_output)
