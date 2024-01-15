import math
from argparse import ArgumentParser
from itertools import permutations

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class MLP(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.fc1 = nn.Linear(4*embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape (4, k)

        # Pass each part through the embedding layer
        x_parts = [self.embedding(x_part) for x_part in x]
    
        # Concatenate the embeddings
        x = torch.cat(x_parts, dim=1)
    
        # Pass the embeddings through the MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def embed(self,x):
        return self.embedding(x)

p = 23
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(num_tokens=p+2, embedding_dim=8,hidden_dim=32, output_dim=p+2).to(device)
model.load_state_dict(torch.load('results/mlp_test_8_32.pth'))

x = torch.arange(0,p).to(device)
y = model.embed(x).cpu().detach().numpy()
# Specify the number of components to reduce to (in this case, 2)
num_components = 2

# Create a PCA instance and fit_transform the data
pca = PCA(n_components=num_components)
reduced_vector = pca.fit_transform(y)
# norm = np.sqrt(np.sum(reduced_vector**2,axis=1)).reshape(-1,1)
# reduced_vector = reduced_vector/norm
plt.figure()
plt.scatter(reduced_vector[:,0],reduced_vector[:,1])
# add names on every spot
for label, x_1, y_1 in zip(range(0,p), reduced_vector[:,0], reduced_vector[:,1]):
    plt.annotate(label,
                 (x_1, y_1),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')
plt.title('Representation Learning')
#plt.savefig('figures/embedding/scatter_plot_3.png')
plt.show()

# Create a 3D scatter plot
# fig = go.Figure(data=[go.Scatter3d(x=reduced_vector[:,0], y=reduced_vector[:,1], z=reduced_vector[:,2], mode='markers', marker=dict(color='blue', size=8))])

# # Set labels for the axes
# fig.update_layout(scene=dict(xaxis_title='X Label', yaxis_title='Y Label', zaxis_title='Z Label'))

# # Save the plot to an interactive HTML file
# fig.write_html('3d_scatter_plot.html')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(reduced_vector[:,0],reduced_vector[:,1],reduced_vector[:,2], c='b', marker='o')

# # Set labels for the axes
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


#para = list(model.embedding.parameters())[0].cpu().detach().numpy()
# 绘制热力图
# plt.imshow(para, cmap='viridis')

# # 添加颜色条
# plt.colorbar()

# # 显示图形
# plt.show()