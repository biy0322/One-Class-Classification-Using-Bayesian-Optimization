#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

class Plotly_Visualization():
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
    def visualization_df(self):
        data1 = pd.DataFrame(self.data1)
        data2 = pd.DataFrame(self.data2)
        data = pd.concat([data1, data2], axis=1)
        data["y"] = data2
        data.columns = ["X1","X2","Category", "y"]
        data["Category"] = data['Category'].astype("object")
        return data
    
    def get_2d(self):
        data = self.visualization_df()
        fig = px.scatter(data, x="X1", y="X2", color="Category", color_discrete_map={1: 'red', 0: 'blue'},
                         marginal_y = "box")
        fig.update_traces(marker=dict(size=15,
                                      line=dict(width=0.5,
                                      color='DarkSlateGrey')),
                                      selector=dict(mode='markers'))
        fig.show()
        
    def get_3d(self):
        data = self.visualization_df()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'box'}]], column_widths=[0.9,0.1])
        fig.add_trace(go.Scatter3d(x=data.X1[data.Category==0], y=data.X2[data.Category==0], z=data.y[data.Category==0], mode="markers", marker=dict(size=10, color='rgb(179,205,227)', line=dict(width=0.5, color='DarkSlateGrey')), name="normal"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=data.X1[data.Category==1], y=data.X2[data.Category==1], z=data.y[data.Category==1], mode="markers", marker=dict(size=10, color= 'rgb(251,180,174)', line=dict(width=0.5, color='DarkSlateGrey')), name="abnormal") , row=1, col=1)
        fig.add_trace(go.Box(y=data.X2[data.Category==0],  name = "normal", marker_color='#4C78A8'), row=1, col=2)
        fig.add_trace(go.Box(y=data.X2[data.Category==1],  name = "abnormal", marker_color='#E45756'), row=1, col=2)
        fig.update_layout(height=1000, width=2100, template="plotly_white")
    
        fig.show()
        
    def get_3d_1(self):
        data = self.visualization_df()
        fig = px.scatter_3d(data, x="X1", y="X2", z="y", color="Category", color_discrete_map={1:'rgb(251,180,174)' , 0:'rgb(179,205,227)' })
        fig.update_traces(marker=dict(size=8,
                             line=dict(width=1,
                                      color='DarkSlateGrey')),
                 selector=dict(mode="markers"))
        fig.update_layout(height=1000,
                         width=1500, template="plotly_white", showlegend=False)
        fig.show()
        

class Plotly_Visualization_result():
    def __init__(self, data, scoring):
        self.data = data
        self.scoring = scoring
        
    def get(self):
        fig = go.Figure()
        fig.add_trace(go.Box(y=self.data['grid'], name="Grid"))
        fig.add_trace(go.Box(y=self.data['random'], name="Random"))
        fig.add_trace(go.Box(y=self.data['bayesian'], name="Bayesian"))
        
        
        if self.scoring=="recall":
            fig.update_layout(template='plotly_white',
                             title_text = "Recall score") 
        elif self.scoring=="f1":
            fig.update_layout(template='plotly_white',
                             title_text = "F1 score")
        elif self.scoring=="roc":
            fig.update_layout(template='plotly_white',
                             title_text = "ROC-AUC score")
            
        fig.show()
        

class Seaborn_Visualization():
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
    def visualization_df(self):
        data1 = pd.DataFrame(self.data1)
        data2 = pd.DataFrame(self.data2)
        data = pd.concat([data1, data2], axis=1)
        data["y"] = data2
        data.columns = ["X1","X2","Category", "y"]
        data["Category"] = data['Category'].astype("object")
        return data
    
    def get_2d(self):
        data = self.visualization_df()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x="X1", y="X2", hue="Category", palette={1: 'red', 0: 'blue'}, s=100, edgecolor='darkgray')
        plt.title("2D Scatter Plot")
        plt.show()
        
    def get_3d(self):
        data = self.visualization_df()
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(data.X1[data.Category==0], data.X2[data.Category==0], data.y[data.Category==0], color='blue', label='normal')
        ax1.scatter(data.X1[data.Category==1], data.X2[data.Category==1], data.y[data.Category==1], color='red', label='abnormal')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('y')
        ax1.legend()
        
        ax2 = fig.add_subplot(122)
        sns.boxplot(x="Category", y="X2", data=data, palette={1: 'red', 0: 'blue'})
        plt.title("Box Plot of X2 by Category")
        
        plt.tight_layout()
        plt.show()
        
    def get_3d_1(self):
        data = self.visualization_df()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.X1, data.X2, data.y, c=data.Category.map({1: 'red', 0: 'blue'}), s=50, edgecolor='darkgray')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')
        plt.title("3D Scatter Plot")
        plt.show()
        

class Seaborn_Visualization_result():
    def __init__(self, data, scoring):
        self.data = data
        self.scoring = scoring
        
    def get(self):
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame(self.data)
        sns.boxplot(data=data, palette="Set2")
        
        if self.scoring == "recall":
            plt.title("Recall Score")
        elif self.scoring == "f1":
            plt.title("F1 Score")
        elif self.scoring == "roc":
            plt.title("ROC-AUC Score")
            
        plt.show()