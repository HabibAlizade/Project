import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import altair as alt
alt.renderers.enable('default')

import import_ipynb
import network
import network2 
import data_loader


train = data_loader.train_data
validation = data_loader.validation_data
test = data_loader.test_data

net = network2.Network([784,30,10], cost = network2.CrossEntropyCost)

def chart_maker(eta,lmbda, data, x, measure):
    if 'acc' in measure:
        chart = alt.Chart(data).transform_fold(['train_acc', 'eval_acc'])\
        .mark_line().encode(x = x, y = 'value:Q', color = 'key:N')\
        .properties(title = f'eta = {eta} | Accuracy | lmbda = {lmbda}')
    else:
        chart = alt.Chart(data).transform_fold(['train_cost', 'eval_cost'])\
        .mark_line().encode(x = x, y = 'value:Q', color = 'key:N')\
        .properties(title = f'eta = {eta} | Cost | lmbda = {lmbda}')
    return chart

def run(eta, lmbda, epoch):
    net.weight_initializer_large()
    
    tc, ta, ec, ea = net.SGD(\
        training_data=train, 
        epoch=epoch, 
        mini_batch_size=10, 
        eta = eta, 
        lmbda = lmbda,
        evaluation_data = validation,
        monitor_training_cost = True, monitor_training_accuracy = True,
        monitor_evaluation_cost = True, monitor_evaluation_accuracy = True)

    df = [list(range(epoch)), tc, ta, ec, ea]
    df = np.array(df)
    cols_names = ['index', 'train_cost', 'train_acc', 'eval_cost', 'eval_acc']
    result_data = pd.DataFrame(df.transpose(), columns = cols_names)
    
    result_data['train_cost'] = result_data['train_cost']
    result_data['eval_cost'] = result_data['eval_cost']
    
    chart = alt.Chart(result_data).mark_circle().encode(x = 'index', y = 'train_cost')
    return chart
#     for measure in [f'cost_{eta}_{lmbda}', f'acc_{eta}_{lmbda}']:
#         chart = chart_maker(eta = eta, lmbda = lmbda, data = result_data, x = 'index', measure  = measure)
#         charts[measure] = chart
        
# 
run(eta = 0.1, lmbda = 10, epoch = 30)

epoch = 30
charts = {}
for lmbda in [100,10,1,0.1,0.01]:
    run(eta = 32, lmbda = lmbda)

all_charts = (charts['cost_32_100'] | charts['acc_32_100']) & \
(charts['cost_32_10'] | charts['acc_32_10']) & \
(charts['cost_32_1'] | charts['acc_32_1']) & \
(charts['cost_32_0.1'] | charts['acc_32_0.1']) & \
(charts['cost_32_0.01'] | charts['acc_32_0.01'])

# all_charts
all_charts
