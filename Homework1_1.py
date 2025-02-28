
# coding: utf-8

# # Dataset 1:heart_attack_prediction_India

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
from IPython.display import HTML # 导入HTML
import plotly.graph_objects as go
import plotly.colors as pc


# In[3]:


fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.show()


# In[4]:


plt.rcParams['figure.max_open_warning'] = 0


# In[5]:


file_path=r"F:\数据挖掘\Homework 1.1\archive\heart_attack_prediction_india.csv"
data=pd.read_csv(file_path)


# In[6]:


#数据清洗
data=data.dropna()  #清洗掉缺失值
data=data.drop_duplicates()  # 删除重复的行
data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.info()


# In[11]:


selected_data=data[['State_Name','Gender','Age','Diabetes','Obesity','Smoking','Family_History','Stress_Level','Heart_Attack_Risk']]


# In[12]:


selected_data.head()


# In[13]:


selected_data.tail()


# In[14]:


selected_data.describe()


# In[15]:


selected_data.head()


# In[16]:


heart_attack_data = selected_data[selected_data['Heart_Attack_Risk'] == 1]


# In[17]:


heart_attack_data.head()


# In[18]:


heart_attack_data=heart_attack_data[['State_Name','Gender','Age','Diabetes','Obesity','Smoking','Family_History']]


# In[19]:


heart_attack_data.head()


# In[20]:


# 测试数据是否有误

count_1=heart_attack_data[(heart_attack_data['Age']>=20)&(heart_attack_data['Age']<25)].shape[0]
count_2=heart_attack_data[(heart_attack_data['Age']>=25)&(heart_attack_data['Age']<30)].shape[0]
count_3=heart_attack_data[(heart_attack_data['Age']>=30)&(heart_attack_data['Age']<35)].shape[0]
count_4=heart_attack_data[(heart_attack_data['Age']>=35)&(heart_attack_data['Age']<40)].shape[0]
print(count_1)
print(count_2)
print(count_3)
print(count_4)


# In[21]:


#使用searborn库

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots(figsize=(8, 6))

sns.kdeplot(heart_attack_data['Age'], color='royalblue', label='Heart Attack (KDE)', shade=True, ax=ax1)


ax1.set_ylabel('Density', fontsize=12)


ax2 = ax1.twinx()
ax2.set_ylabel('Count',fontsize=12)

bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
labels = ['0-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-100']
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)



# 绘制柱状图，按年龄段显示有心脏病风险的患者数量
# 使用seaborn库中的柱状图绘制工具，输入的数据为筛选出的心脏风险患者的数据，将'Age'列作为横坐标，使用bins对横坐标进行分组，分成好多组
# ax=ax2表示在第二个y轴进行绘图，stat='count'代表纵坐标是这个区间数据点的数量
sns.histplot(heart_attack_data, x='Age', bins=bins, kde=False, multiple='dodge', stat='count', ax=ax2, alpha=0.6, color='lightcoral', label='Heart Attack (Account)')



plt.title('Age Distribution for Heart Attack')

# 设置x轴的标签
ax1.set_xlabel('Age',fontsize=12)

plt.xticks(ticks=range(20, 101, 10), labels=['20', '30', '40', '50', '60', '70', '80', '90', '100'],fontsize=12)

ax1.grid(True, linestyle='--', alpha=0.6)


ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.tick_params(axis='y',labelsize=12)  
ax2.tick_params(axis='y',labelsize=12)  
ax1.tick_params(axis='x',labelsize=12)

plt.show()


# In[22]:


heart_attack_data.describe()


# In[23]:


count_5=selected_data[(selected_data['Diabetes']==1)&(selected_data['Heart_Attack_Risk']==0)].shape[0]
print(count_5)


# In[115]:


#使用seaborn库
plt.rcParams['font.family'] = 'Times New Roman'
diseases = ['Diabetes', 'Obesity', 'Smoking']


fig, axes = plt.subplots(1, len(diseases), figsize=(6 * len(diseases), 6))


if len(diseases) == 1:
    axes = [axes]

for i in range(len(diseases)):
    ct = pd.crosstab(selected_data[diseases[i]], selected_data['Heart_Attack_Risk'])
    

    sns.heatmap(ct, annot=True, cmap='coolwarm', fmt='d', linewidths=0.5, 
                linecolor='white', cbar_kws={'label': 'Count'}, ax=axes[i],
                annot_kws={'size': 16})


    colorbar = axes[i].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=18)  

    colorbar = axes[i].collections[0].colorbar
    colorbar.set_label('Count', fontsize=18)  
    
    axes[i].set_title(f'{diseases[i]} and Heart Attack Risk', fontsize=16)
    axes[i].set_xlabel('Heart Attack Risk', fontsize=16)
    axes[i].set_ylabel(f'{diseases[i]} Status (0=No, 1=Yes)', fontsize=16)
    axes[i].tick_params(axis='both', which='both', length=0, labelsize=12)

    axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=14)  
    axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=14)  

plt.tight_layout()
plt.show()


# In[25]:


#使用pandas库
plt.rcParams['font.family'] = 'Times New Roman'
gender_risk = pd.crosstab(selected_data['Gender'], selected_data['Heart_Attack_Risk'])

gender_0_risk = gender_risk.loc['Female']  
gender_1_risk = gender_risk.loc['Male'] 


plt.figure(figsize=(10, 6))  


plt.subplot(1, 2, 1)
gender_0_risk.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'], legend=True,textprops={'fontsize': 14})
plt.title('Heart Attack Risk Distribution for Female ', fontsize=12)


plt.subplot(1, 2, 2)
gender_1_risk.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'], legend=True,textprops={'fontsize': 14})
plt.title('Heart Attack Risk Distribution for Male ', fontsize=12)


plt.subplots_adjust(wspace=0.4) 


plt.show()


# In[26]:


selected_data.describe()


# In[27]:


#使用seaborn库
plt.rcParams['font.family'] = 'Times New Roman'
disease_counts = selected_data[selected_data['Heart_Attack_Risk'] == 1].groupby('State_Name').size().reset_index(name='count')
sorted_count = disease_counts.sort_values(by='count', ascending=False)
top_10_data = sorted_count.head(15).copy()  

cmap = plt.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=top_10_data['count'].min(), vmax=top_10_data['count'].max())


top_10_data.loc[:, 'color'] = top_10_data['count'].apply(lambda x: cmap(norm(x)))


plt.figure(figsize=(10, 6))
ax = sns.barplot(x='count', y='State_Name', data=top_10_data, palette=top_10_data['color'])


plt.title('Top 15 States with Highest Heart Attack Risk', fontsize=16)
plt.xlabel('Number of Individuals with Heart Attack Risk', fontsize=14)
plt.ylabel('State', fontsize=16)
plt.yticks(fontsize=12)

plt.show()


# # Dataset 2:Mobiles Datasets

# In[74]:


file_path_2=r"F:\数据挖掘\Homework 1.1\archive (1)\Mobiles Dataset (2025).csv"


# In[75]:


dataset_2=pd.read_csv(file_path_2,encoding='ISO-8859-1')


# In[76]:


dataset_2.head()


# In[77]:


dataset_2.tail()


# In[78]:


selected_data_2=dataset_2[dataset_2['Company Name']=='Apple']


# In[79]:


selected_data_2.head()


# In[80]:


selected_data_2.tail()


# In[81]:


selected_data_2=selected_data_2[selected_data_2['Model Name'].str.contains("iPhone",na=False)]


# In[82]:


selected_data_2.tail()


# In[83]:


selected_data_2=selected_data_2[['Model Name','Mobile Weight','Launched Price (USA)','Launched Price (China)','Launched Year']]


# In[84]:


selected_data_2.head()


# In[85]:


selected_data_2.info()


# In[86]:


selected_data_2.head()


# In[87]:


selected_data_2['Launched Price (China)'] = selected_data_2['Launched Price (China)'].astype(str)
selected_data_2['Launched Price (China)'] = selected_data_2['Launched Price (China)'].str.replace('CNY ', '', regex=False)
selected_data_2['Launched Price (China)'] = selected_data_2['Launched Price (China)'].str.replace(',', '', regex=False)
selected_data_2['Launched Price (China)'] = selected_data_2['Launched Price (China)'].astype(float)


# In[88]:


selected_data_2['Launched Price (USA)'] = selected_data_2['Launched Price (USA)'].astype(str)
selected_data_2['Launched Price (USA)'] = selected_data_2['Launched Price (USA)'].str.replace('USD ', '', regex=False)
selected_data_2['Launched Price (USA)'] = selected_data_2['Launched Price (USA)'].str.replace(',', '', regex=False)
selected_data_2['Launched Price (USA)'] = selected_data_2['Launched Price (USA)'].astype(float)


# In[104]:


selected_data_2.head()


# In[90]:


save_path=r'F:\数据挖掘\Homework 1.1\archive (1)\Mobile Dataset Selected.csv'


# In[91]:


selected_data_2.to_csv(save_path)


# In[105]:


selected_data_2.head()


# In[106]:


selected_data_2.tail()


# In[94]:


selected_data_2.info()


# In[95]:


#使用plotly库
price_by_year = selected_data_2.groupby('Launched Year')['Launched Price (China)'].mean().reset_index()
max_year=price_by_year['Launched Price (China)'].idxmax()
min_year=price_by_year['Launched Price (China)'].idxmin()
colors = ['green'] * len(price_by_year)
special_years = {max_year: 'red', min_year: 'blue'} 

for year, color in special_years.items():
    colors[year]=color
fig = px.line(price_by_year, x='Launched Year', y='Launched Price (China)',
              title='不同年份产品价格的平均趋势',
              labels={'Launched Year': 'Year', 'Launched Price (China)': 'Average Price (CNY)'},
              color_discrete_sequence=['green'], 
              markers=True) 

fig.update_traces(marker=dict(color=colors, size=8))
fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        title='年份',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    yaxis=dict(
        title='平均价格 (CNY)',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    title_font=dict(size=14),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.show()


# In[96]:


print("\n按照年份分组计算平均价格的结果：")
print(price_by_year)


# In[93]:


selected_data_2.to_csv(r"F:\数据挖掘\Homework 1.1\archive (1)\clean data_2.csv")


# In[97]:


selected_data_2.describe()


# In[98]:


selected_data_2.tail()


# In[99]:


selected_data_2.tail()


# In[100]:


selected_data_2 = selected_data_2.sort_values(by='Mobile Weight')


# In[107]:


#使用plotly库
selected_data_2['Mobile Weight'] = selected_data_2['Mobile Weight'].astype(str)
selected_data_2['Mobile Weight'] = selected_data_2['Mobile Weight'].str.replace('g', '').astype(float)

fig = px.scatter(selected_data_2, x='Mobile Weight', y='Launched Price (USA)',
                 title='手机重量与价格的关系',
                 labels={'Mobile Weight': '手机重量 (g)', 'Launched Price (USA)': '价格 (USA)'},
                 hover_data=['Model Name', 'Launched Year'],  
                 color='Launched Year',  
                 size='Launched Price (USA)', 
                 opacity=0.7,  
                 trendline="ols")  


fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        title='手机重量 (g)',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    yaxis=dict(
        title='价格 (USA)',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    title_font=dict(size=14),
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title="年份",  
)


fig.update_traces(
    marker=dict(
        line=dict(width=0.5, color='DarkSlateGrey'),  
    ),
    selector=dict(mode='markers')
)


fig.show()


# In[110]:


#使用plotly库
fig = px.box(selected_data_2, x='Launched Year', y='Launched Price (China)',
             title='不同年份产品价格的箱线图',
             labels={'Launched Year': '年份', 'Launched Price (China)': '价格 (CNY)'},
             color='Launched Year',  #
             category_orders={"Launched Year": sorted(selected_data_2['Launched Year'].unique(), reverse=True)}) #图例排序


fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        title='年份',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    yaxis=dict(
        title='价格 (CNY)',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    title_font=dict(size=14),
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title="年份",
)

fig.show()


# In[111]:


#使用plotly库
fig = px.box(selected_data_2, x='Launched Year', y='Launched Price (USA)',
             title='不同年份产品价格的箱线图',
             labels={'Launched Year': '年份', 'Launched Price (USA)': '价格 (USD)'},
             color='Launched Year',  
             category_orders={"Launched Year": sorted(selected_data_2['Launched Year'].unique(), reverse=True)}) #图例排序

fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        title='Year',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    yaxis=dict(
        title='Price (USD)',
        gridcolor='lightgrey',
        titlefont=dict(size=12),
    ),
    title_font=dict(size=14),
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title="Year",
)

fig.show()


# In[112]:


#使用seaborn库
sns.pairplot(selected_data_2)
plt.show()

