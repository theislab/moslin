#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
from os import listdir
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from math import isclose
from scipy import stats
from matplotlib import pyplot as plt
import re


# In[2]:


plt.rcParams.update({'font.size': 22})


# In[3]:


# Parameters
whitelist_col12_origin = ["Epicardium (Ventricle)", "Epicardium (Atrium)", "Fibroblasts (const.)", "Fibroblasts (cfd)", "Fibroblasts (cxcl12a)", "Fibroblasts (proliferating)"]
whitelist_nppc_origin = ["Endocardium (Ventricle)", "Endocardium (Atrium)", "Fibroblasts (spock3)"]

tmat_path = ['/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/HP_search_1',
            '/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/HP_search_2',
            '/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/HP_search_3']
#['/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/output_mid_v2_20231128',
#            '/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/lot_res',
#            '/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/ot_res']


# In[4]:


def calculate_transient_percentages(transitions, whitelist_col12_origin = whitelist_col12_origin, whitelist_nppc_origin = whitelist_nppc_origin):
    # Function to calculate col12 and nppc percentages

    # Calculate percentages from
    transitions['Percentage_from'] = transitions.groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Transfer_ratio'].transform(lambda x: 100 * x / x.sum())
    # transitions.groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Percentage_from'].sum() # All sums should be 100%

    # Calculate percentage col12 from whitelisted
    col12_selector = (transitions['Cell_type_from'].isin(whitelist_col12_origin)) & (transitions['Cell_type_to'] == 'Fibroblasts (col12a1a)') & (transitions['t1'] == 'Ctrl')
    col12_percentages = pd.DataFrame(transitions[col12_selector].groupby(['Tree_from', 'Tree_to'])['Percentage_from'].sum())
    col12_percentages.reset_index(inplace = True)

    # Calculate percentage nppc from whitelisted
    nppc_selector = (transitions['Cell_type_from'].isin(whitelist_nppc_origin)) & (transitions['Cell_type_to'] == 'Fibroblasts (nppc)') & (transitions['t1'] == '3dpi')
    nppc_percentages = pd.DataFrame(transitions[nppc_selector].groupby(['Tree_from', 'Tree_to'])['Percentage_from'].sum())
    nppc_percentages.reset_index(inplace = True)
    
    return col12_percentages, nppc_percentages


# In[5]:


sns.set_palette('colorblind')


# In[6]:


# Celltype colors
celltype_colors = pd.read_csv('../Publication_repo/moslin/data/hu_zebrafish_linnaeus/Zebrafish_cell_type_colors.csv')
#celltype_colors


# In[7]:


ct_lut = dict(zip(map(str, celltype_colors['Cell.type']), celltype_colors['color']))
#pd.Series(celltype_colors['Cell.type']).map(ct_lut)
#network_pal = sns.cubehelix_palette(network_labels.unique().size,
 #                                   light=.9, dark=.1, reverse=True,
 #                                   start=1, rot=-2)
#network_lut = dict(zip(map(str, network_labels.unique()), network_pal))

#network_colors = pd.Series(network_labels).map(network_lut)


# In[8]:


files_dt = pd.DataFrame()
for this_path in tmat_path:
    files_dt = pd.concat([files_dt,
                         pd.DataFrame({'Path' : this_path,
                                       'Run' : [f for f in listdir(this_path) if not f.startswith('.')],
                                       })],
                         axis = 0, ignore_index = True)
files_dt['Filename_array'] = files_dt.apply(lambda x:  [y for y in listdir(x.Path + '/' + x.Run) if 'argmax' in y and 'csv' in y], axis = 1)
files_dt = files_dt[[not not x for x in files_dt['Filename_array']]]
files_dt['Filename'] = files_dt.apply(lambda x: (x['Filename_array'])[0], axis = 1)


# In[9]:


files_dt


# In[10]:


files_dt['Method'] = "foo"
files_dt['alpha'] = -1
files_dt['epsilon'] = -1
files_dt['beta'] = -1
files_dt['taua'] = -1
files_dt['taub'] = -1
files_dt['accuracy'] = -1
files_dt['col12_percentage'] = -1
files_dt['nppc_percentage'] = -1
for i in files_dt.index:
    this_tmat_path = files_dt['Path'][i]
    this_dir = files_dt['Run'][i]
    this_argmax_filename = files_dt['Filename'][i]

    argmax = pd.read_csv(this_tmat_path + '/' + this_dir + '/' + this_argmax_filename, index_col = 0)
    this_accuracy = accuracy_score(y_true = argmax['Cell_type_to'], 
                                  y_pred = argmax['Cell_type_from'])
    
    this_transient_filename_array = [x for x in listdir(this_tmat_path + '/'+ this_dir) if 'transient_test' in x]
    if (len(this_transient_filename_array) > 0):
        this_transient_filename = this_transient_filename_array[0]
        this_transient_test = pd.read_csv(this_tmat_path + '/' + this_dir + '/' + this_transient_filename, index_col = 0)
        files_dt.loc[i, 'col12_percentage'] = this_transient_test.loc[0, 'Percentage_from_col12_correct']
        files_dt.loc[i, 'nppc_percentage'] = this_transient_test.loc[0, 'Percentage_from_nppc_correct']

    this_run_split = this_dir.split('_')
    
    #float('moslin_max_alpha-0.1_epsilon-1.0_beta-0.0_taua-0.9_taub-0.9_save-0 accuracy'.split('_')[2].split('-')[1])
    #files_dt['Method'][i] = this_run_split[0]
    files_dt.loc[i, 'Method'] = this_run_split[0]
    files_dt.loc[i, 'alpha'] = float(this_run_split[2].split('-')[1])
    if (this_run_split[0] == "moslin") & (float(this_run_split[2].split('-')[1]) == 0):
        files_dt.loc[i, 'Method'] = "OT"
    files_dt.loc[i, 'epsilon'] = float(this_run_split[3].split('-')[1])
    files_dt.loc[i, 'beta'] = float(this_run_split[4].split('-')[1])
    files_dt.loc[i, 'taua'] = float(this_run_split[5].split('-')[1])
    files_dt.loc[i, 'taub'] = float(this_run_split[6].split('-')[1])
    files_dt.loc[i, 'accuracy'] = this_accuracy
    #files_dt.loc[i, 'col12_percentage'] = this_transient_test.loc[0, 'Percentage_from_col12_correct']
    #files_dt.loc[i, 'nppc_percentage'] = this_transient_test.loc[0, 'Percentage_from_nppc_correct']

    #print(f'{this_dir} accuracy: {this_accuracy:.2f}')


# In[11]:


files_dt['Convergence'] = [float((re.split('_|\.csv', x))[3]) for x in files_dt['Filename']]
files_dt_all = files_dt
files_dt = files_dt[(files_dt['Convergence'] >= 0.7) & (files_dt['alpha'] != 0.05)]


# In[12]:


files_dt[(files_dt['taub'] == 1) & (files_dt['taua'] == 0.4) & (files_dt['epsilon'] == 0.1)]


# In[13]:


files_dt[(files_dt['Method'] == 'moslin') & (files_dt['taub'] == 1) & (files_dt['taua'] == 0.4)]['epsilon'].unique()


# In[14]:


df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin')].pivot_table(index='alpha', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)

sns.heatmap(df_pivot, annot=True, cmap='coolwarm')
#plt.savefig('../Images/Grid_search_epsilon_alpha.png', bbox_inches='tight')


# In[15]:


df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin') & (files_dt['beta'] == 0) & (files_dt['taua'] == 0.4)].pivot_table(index='alpha', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)
plt.rcParams.update({'font.size': 12})
g = sns.heatmap(df_pivot, annot=True, cmap='coolwarm', vmin = 0.4, vmax = 0.8)
g.set_facecolor('grey')
plt.title('Accuracy')
plt.savefig('../Images/Grid_search_epsilon_alpha.png', bbox_inches='tight')
plt.rcParams.update({'font.size': 22})


# In[16]:


df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin') & (files_dt['beta'] == 0) & (files_dt['alpha'] == 0.01)].pivot_table(index='taua', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)

plt.rcParams.update({'font.size': 12})

g = sns.heatmap(df_pivot, annot=True, cmap='coolwarm', vmin = 0.4, vmax = 0.8)
g.set_facecolor('grey')
plt.title('Accuracy')

plt.savefig('../Images/Grid_search_epsilon_taua.png', bbox_inches='tight')
plt.rcParams.update({'font.size': 22})


# In[17]:


df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin')].pivot_table(index='taua', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)

sns.heatmap(df_pivot, annot=True, cmap='coolwarm')
#plt.savefig('../Images/Grid_search_epsilon_taua.png', bbox_inches='tight')


# In[18]:


df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin')].pivot_table(index='beta', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)

sns.heatmap(df_pivot, annot=True, cmap='coolwarm')
#plt.savefig('../Images/Grid_search_epsilon_beta.png', bbox_inches='tight')


# In[19]:


files_dt_plot = files_dt[(files_dt['taub'] == 1) & (files_dt['beta'] == 0)]
files_dt_plot.sort_values(by = 'alpha', ascending = False, inplace = True)


# In[20]:


plt.rcParams["figure.figsize"] = (6,6)
g = sns.scatterplot(files_dt_plot, x = 'accuracy', y = 'col12_percentage', hue = 'Method', s = 80)#, height = 4, width = 4)

# Adding labels and title
plt.xlabel('Accuracy')
plt.ylabel('Correct col12a1a\nprecursors (%)')

# Remove legend
plt.legend([],[], frameon=False)

#plt.savefig('../Images/Col12_vs_accuracy_hyperparameters_taub1.png', bbox_inches='tight')
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[21]:


plt.rcParams["figure.figsize"] = (6,6)
g = sns.scatterplot(files_dt_plot, x = 'accuracy', y = 'nppc_percentage', hue = 'Method', s = 80)#, height = 4, width = 4)

# Adding labels and title
plt.xlabel('Accuracy')
plt.ylabel('Correct nppc\nprecursors (%)')

# Remove legend
plt.legend([],[], frameon=False)

#plt.savefig('../Images/Nppc_vs_accuracy_hyperparameters_taub1.png', bbox_inches='tight')
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[22]:


sns.scatterplot(files_dt[files_dt['taub'] == 1], x = 'col12_percentage', y = 'nppc_percentage', hue = 'Method')


# In[23]:


# Best hyperparameter set per method ot lot moslin:
lot_results = files_dt[files_dt['Method'] == 'lot']
ot_results = files_dt[(files_dt['Method'] == 'OT')]
moslin_results = files_dt[(files_dt['Method'] == 'moslin')]


# In[24]:


#lot_all_optimum = (lot_results.sort_values(by = ['accuracy'], ascending = False)).iloc[0]
lot_optimum = (lot_results[lot_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
ot_optimum = (ot_results[ot_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
moslin_optimum = (moslin_results[moslin_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
moslin_optimum


# In[25]:


ot_optimum


# In[26]:


# Plot confusion matrix for best moslin result
moslin_optimum_argmax = pd.read_csv(moslin_optimum['Path'] + '/' + moslin_optimum['Run'] + '/' + moslin_optimum['Filename'], index_col = 0)
all_labels = np.intersect1d(moslin_optimum_argmax['Cell_type_to'], 
                                                     moslin_optimum_argmax['Cell_type_from'])
persistence_cm = confusion_matrix(y_true = moslin_optimum_argmax['Cell_type_to'], 
                                  y_pred = moslin_optimum_argmax['Cell_type_from'],
                                 labels = all_labels)
persistence_precision = confusion_matrix(y_true = moslin_optimum_argmax['Cell_type_to'], 
                                  y_pred = moslin_optimum_argmax['Cell_type_from'],
                                 labels = all_labels, normalize = 'pred')
persistence_recall = confusion_matrix(y_true = moslin_optimum_argmax['Cell_type_to'], 
                                  y_pred = moslin_optimum_argmax['Cell_type_from'],
                                 labels = all_labels, normalize = 'true')
                                  #np.union1d(moslin_optimum_argmax['Cell_type_to'], 
                                  #                   moslin_optimum_argmax['Cell_type_from']))
mo_argmax_ctrl = moslin_optimum_argmax[moslin_optimum_argmax['t1'] == 'Ctrl']
mo_argmax_3dpi = moslin_optimum_argmax[moslin_optimum_argmax['t1'] == '3dpi']
persistence_cm_ctrl = confusion_matrix(y_true = mo_argmax_ctrl['Cell_type_to'], 
                                  y_pred = mo_argmax_ctrl['Cell_type_from'],
                                 labels = np.union1d(mo_argmax_ctrl['Cell_type_to'], 
                                                     mo_argmax_ctrl['Cell_type_from']))
persistence_cm_3dpi = confusion_matrix(y_true = mo_argmax_3dpi['Cell_type_to'], 
                                  y_pred = mo_argmax_3dpi['Cell_type_from'],
                                 labels = np.union1d(mo_argmax_3dpi['Cell_type_to'], 
                                                     mo_argmax_3dpi['Cell_type_from']))


# In[27]:


cm_colors = pd.Series(all_labels).map(ct_lut)


# In[28]:


g = sns.clustermap(persistence_precision, row_cluster = False, col_cluster = False,
              row_colors = cm_colors.to_numpy(), col_colors = cm_colors.to_numpy(),
                  cbar_pos=(1.02, 0.4, 0.05, 0.18), dendrogram_ratio = 0.01)
g.ax_heatmap.tick_params(right=False, bottom=False)
ax = g.ax_heatmap
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
ax.set_ylabel('True', fontsize = 24)
ax.set_xlabel('Predicted', fontsize = 24)
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.savefig('../Images/Persistence_precision_optimum_moslin.png', bbox_inches='tight')
plt.show()
# Legend annotation?


# In[29]:


g = sns.clustermap(persistence_recall, row_cluster = False, col_cluster = False,
              row_colors = cm_colors.to_numpy(), col_colors = cm_colors.to_numpy(),
                  cbar_pos=(1.02, 0.4, 0.05, 0.18), dendrogram_ratio = 0.01)
g.ax_heatmap.tick_params(right=False, bottom=False)
ax = g.ax_heatmap
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
ax.set_ylabel('True', fontsize = 24)
ax.set_xlabel('Predicted', fontsize = 24)
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.savefig('../Images/Persistence_recall_optimum_moslin.png', bbox_inches='tight')
plt.show()


# In[30]:


# Plot accuracy of best moslin, ot, lot, realistic lot
optimal_hyperparameters_and_perf =     pd.DataFrame({'algorithm' : ['moslin', 'LineageOT', 'OT'],
                  'accuracy' : [moslin_optimum['accuracy'], lot_optimum['accuracy'], ot_optimum['accuracy']],
                  'col12' : [moslin_optimum['col12_percentage'], lot_optimum['col12_percentage'], ot_optimum['col12_percentage']],
                  'nppc' : [moslin_optimum['nppc_percentage'], lot_optimum['nppc_percentage'], ot_optimum['nppc_percentage']],
                  'alpha' : [moslin_optimum['alpha'], lot_optimum['alpha'], ot_optimum['alpha']],
                  'epsilon' : [moslin_optimum['epsilon'], lot_optimum['epsilon'], ot_optimum['epsilon']],
                  'beta' : [moslin_optimum['beta'], lot_optimum['beta'], ot_optimum['beta']],
                  'taua' : [moslin_optimum['taua'], lot_optimum['taua'], ot_optimum['taua']],
                  'taub' : [moslin_optimum['taub'], lot_optimum['taub'], ot_optimum['taub']]})
optimal_hyperparameters_and_perf_melt =    pd.melt(optimal_hyperparameters_and_perf, 
            id_vars = ['algorithm', 'accuracy', 'alpha', 'epsilon', 'beta', 'taua', 'taub'],
            var_name = 'Transient_fibroblast',
            value_name = 'Percentage_correct')
optimal_hyperparameters_and_perf


# In[31]:


# Plot accuracy of best moslin, ot, lot, realistic lot
optimal_hyperparameters_and_perf =     pd.DataFrame({'algorithm' : ['moslin', 'LineageOT', 'OT'],
                  'accuracy' : [moslin_optimum['accuracy'], lot_optimum['accuracy'], ot_optimum['accuracy']],
                  'col12' : [moslin_optimum['col12_percentage'], lot_optimum['col12_percentage'], ot_optimum['col12_percentage']],
                  'nppc' : [moslin_optimum['nppc_percentage'], lot_optimum['nppc_percentage'], ot_optimum['nppc_percentage']],
                  'alpha' : [moslin_optimum['alpha'], lot_optimum['alpha'], ot_optimum['alpha']],
                  'epsilon' : [moslin_optimum['epsilon'], lot_optimum['epsilon'], ot_optimum['epsilon']],
                  'beta' : [moslin_optimum['beta'], lot_optimum['beta'], ot_optimum['beta']],
                  'taua' : [moslin_optimum['taua'], lot_optimum['taua'], ot_optimum['taua']],
                  'taub' : [moslin_optimum['taub'], lot_optimum['taub'], ot_optimum['taub']]})
optimal_hyperparameters_and_perf_melt =    pd.melt(optimal_hyperparameters_and_perf, 
            id_vars = ['algorithm', 'accuracy', 'alpha', 'epsilon', 'beta', 'taua', 'taub'],
            var_name = 'Transient_fibroblast',
            value_name = 'Percentage_correct')
optimal_hyperparameters_and_perf


# In[32]:


ax = sns.barplot(optimal_hyperparameters_and_perf, x = 'algorithm', y = 'accuracy')
ax.set(xlabel='', ylabel='Cell type persistence accuracy')
#plt.savefig('../Images/Persistence_maximum_accuracies.png', bbox_inches='tight')
plt.show()


# In[33]:


# Load predictions
moslin_optimum_argmax = pd.read_csv(moslin_optimum['Path'] + '/' + moslin_optimum['Run'] + '/' + moslin_optimum['Filename'], index_col = 0)
lot_optimum_argmax = pd.read_csv(lot_optimum['Path'] + '/' + lot_optimum['Run'] + '/' + lot_optimum['Filename'], index_col = 0)
ot_optimum_argmax = pd.read_csv(ot_optimum['Path'] + '/' + ot_optimum['Run'] + '/' + ot_optimum['Filename'], index_col = 0)


# In[34]:


# Determine confidence intervals by subsampling predictions
num_subsamples = 100
accuracy_dfs = []
for i in range(num_subsamples):
    moslin_argmax_sample = moslin_optimum_argmax.sample(frac = 0.25, axis = 0, random_state = i)
    lot_argmax_sample = lot_optimum_argmax.sample(frac = 0.25, axis = 0, random_state = i)
    ot_argmax_sample = ot_optimum_argmax.sample(frac = 0.25, axis = 0, random_state = i)
    moslin_acc_score = accuracy_score(y_true = moslin_argmax_sample['Cell_type_to'], 
                                       y_pred = moslin_argmax_sample['Cell_type_from'])
    lot_acc_score = accuracy_score(y_true = lot_argmax_sample['Cell_type_to'], 
                                       y_pred = lot_argmax_sample['Cell_type_from'])
    ot_acc_score = accuracy_score(y_true = ot_argmax_sample['Cell_type_to'], 
                                       y_pred = ot_argmax_sample['Cell_type_from'])
    temp_acc_sample = pd.DataFrame({'Sample': [i + 1], 'moslin_acc': [moslin_acc_score], 'lot_acc': [lot_acc_score], 'ot_acc': [ot_acc_score]})
    accuracy_dfs.append(temp_acc_sample)
acc_sample = pd.concat(accuracy_dfs, ignore_index=True)

print(acc_sample)


# In[41]:


err_bar_sizes = np.absolute(acc_sample[['moslin_acc', 'lot_acc', 'ot_acc']].quantile(q = [0.025, 0.975]).to_numpy() - optimal_hyperparameters_and_perf['accuracy'].to_numpy())


# In[42]:


err_bar_sizes


# In[43]:


sns.color_palette()


# In[44]:


print(sns.color_palette().as_hex())


# In[47]:


categories = optimal_hyperparameters_and_perf['algorithm']
summary_statistics = optimal_hyperparameters_and_perf['accuracy']
standard_errors = err_bar_sizes

# Plotting the barplot with error bars
plt.bar(categories, summary_statistics, yerr=standard_errors, capsize=10, color=['#176D9C', '#C38820', '#158B6A'])

# Adding labels and title
plt.xlabel('')
plt.ylabel('Accuracy')
plt.title('Cell type persistency')

# Save the plot
plt.savefig('../Images/Persistence_maximum_accuracies_subsampling_ci.png', bbox_inches='tight')

# Show the plot
plt.show()


# ### Accuracy confidence intervals over dataset combinations 

# In[21]:


accuracy_score(y_true = moslin_optimum_argmax['Cell_type_to'],
               y_pred = moslin_optimum_argmax['Cell_type_from'])


# In[22]:


moslin_optimum_argmax


# In[41]:


moslin_combination_accuracies =     moslin_optimum_argmax.         groupby(['Tree_from', 'Tree_to']).         apply(lambda x: accuracy_score(y_true = x['Cell_type_to'], y_pred = x['Cell_type_from'])).         reset_index().rename({0 : 'Accuracy'}, axis = 1)
lot_combination_accuracies =     lot_optimum_argmax.         groupby(['Tree_from', 'Tree_to']).         apply(lambda x: accuracy_score(y_true = x['Cell_type_to'], y_pred = x['Cell_type_from'])).         reset_index().rename({0 : 'Accuracy'}, axis = 1)
ot_combination_accuracies =     ot_optimum_argmax.         groupby(['Tree_from', 'Tree_to']).         apply(lambda x: accuracy_score(y_true = x['Cell_type_to'], y_pred = x['Cell_type_from'])).         reset_index().rename({0 : 'Accuracy'}, axis = 1)
#stats.tstd(a = moslin_combination_accuracies['Accuracy'])


# In[42]:


cells_to_moslin = moslin_optimum_argmax[['To', 'Tree_from', 'Tree_to']].drop_duplicates().groupby(['Tree_to', 'Tree_from']).count().reset_index().rename({'To' : 'Cells'}, axis = 1)
moslin_combination_accuracies_weights = moslin_combination_accuracies.merge(cells_to_moslin)
moslin_combination_accuracies_weights['Norm_weight'] = moslin_combination_accuracies_weights['Cells']/sum(moslin_combination_accuracies_weights['Cells'])
sewm_moslin_95 = 2 * stats.tstd(a = moslin_combination_accuracies['Accuracy']) * sum(np.square(moslin_combination_accuracies_weights['Norm_weight']))


# In[43]:


# All passed - could also do for lot and ot
sum(moslin_combination_accuracies_weights['Norm_weight']) == 1
sum(moslin_combination_accuracies_weights['Cells']) == len(moslin_optimum_argmax)
isclose(sum(moslin_combination_accuracies_weights['Accuracy'] * moslin_combination_accuracies_weights['Norm_weight']), moslin_optimum['accuracy'])


# In[44]:


cells_to_lot = lot_optimum_argmax[['To', 'Tree_from', 'Tree_to']].drop_duplicates().groupby(['Tree_to', 'Tree_from']).count().reset_index().rename({'To' : 'Cells'}, axis = 1)
lot_combination_accuracies_weights = lot_combination_accuracies.merge(cells_to_lot)
lot_combination_accuracies_weights['Norm_weight'] = lot_combination_accuracies_weights['Cells']/sum(lot_combination_accuracies_weights['Cells'])
sewm_lot_95 = 2 * stats.tstd(a = lot_combination_accuracies['Accuracy']) * sum(np.square(lot_combination_accuracies_weights['Norm_weight']))


# In[45]:


cells_to_ot = ot_optimum_argmax[['To', 'Tree_from', 'Tree_to']].drop_duplicates().groupby(['Tree_to', 'Tree_from']).count().reset_index().rename({'To' : 'Cells'}, axis = 1)
ot_combination_accuracies_weights = ot_combination_accuracies.merge(cells_to_ot)
ot_combination_accuracies_weights['Norm_weight'] = ot_combination_accuracies_weights['Cells']/sum(ot_combination_accuracies_weights['Cells'])
sewm_ot_95 = 2 * stats.tstd(a = ot_combination_accuracies['Accuracy']) * sum(np.square(ot_combination_accuracies_weights['Norm_weight']))


# In[46]:


sns.color_palette("deep")


# In[47]:


print(sns.color_palette().as_hex())


# In[30]:


err_bar_sizes


# In[56]:


err_bar_sizes = np.array(([sewm_moslin_95, sewm_lot_95, sewm_ot_95], [sewm_moslin_95, sewm_lot_95, sewm_ot_95]))

categories = optimal_hyperparameters_and_perf['algorithm']
summary_statistics = optimal_hyperparameters_and_perf['accuracy']
standard_errors = err_bar_sizes

# Plotting the barplot with error bars
plt.bar(categories, summary_statistics, yerr=standard_errors, capsize=10, color=['#176D9C', '#C38820', '#158B6A'])

# Adding labels and title
plt.xlabel('')
plt.ylabel('Accuracy')
plt.title('Cell type persistence')

# Save the plot
plt.savefig('../Images/Persistence_maximum_accuracies_sewm95.png', bbox_inches='tight')

# Show the plot
plt.show()


# ## Transient fibroblasts

# In[52]:


optimal_hyperparameters_and_perf_melt.replace(to_replace = {'col12' : 'col12a1a'}, inplace = True)


# In[53]:


ax = sns.barplot(optimal_hyperparameters_and_perf_melt, x = 'Transient_fibroblast', y = 'Percentage_correct', hue = 'algorithm')
#ax.set(xlabel='', ylabel='Percentage correct')

# Increase y-axis range
plt.ylim(0, 99)

# Adding labels and title
plt.xlabel('')
plt.ylabel('Correct (%)')
plt.title('Transient fibroblast precursors')

# Legend
plt.legend([],[], frameon=False)
#ax._legend.remove()
#sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.6), ncol=1, title=None, frameon=False)
#sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.savefig('../Images/Transient_fibroblast_performance.png', bbox_inches='tight')
plt.show()


# In[ ]:


# Significance testing of moslin and OT.


# In[54]:


# Load transitions
moslin_optimum_transfers = pd.read_csv(moslin_optimum['Path'] + '/' + moslin_optimum['Run'] + '/transfer_ratios_1.0.csv', index_col = 0)
ot_optimum_transfers = pd.read_csv(ot_optimum['Path'] + '/' + ot_optimum['Run'] + '/transfer_ratios_1.0.csv', index_col = 0)

# Calculate percentages from whitelisted cell types per dataset combination
moslin_col12, moslin_nppc = calculate_transient_percentages(transitions = moslin_optimum_transfers)
ot_col12, ot_nppc = calculate_transient_percentages(transitions = ot_optimum_transfers)

# Merge technique results and calculate differences
moslin_col12.rename(mapper = {'Percentage_from' : 'Percentage_moslin'}, axis = 1, inplace = True)
moslin_nppc.rename(mapper = {'Percentage_from' : 'Percentage_moslin'}, axis = 1, inplace = True)
ot_col12.rename(mapper = {'Percentage_from' : 'Percentage_ot'}, axis = 1, inplace = True)
ot_nppc.rename(mapper = {'Percentage_from' : 'Percentage_ot'}, axis = 1, inplace = True)
col12_comparison = moslin_col12.merge(ot_col12, on = ['Tree_from', 'Tree_to'])
col12_comparison['Diff'] = col12_comparison['Percentage_moslin'] - col12_comparison['Percentage_ot']
nppc_comparison = moslin_nppc.merge(ot_nppc, on = ['Tree_from', 'Tree_to'])
nppc_comparison['Diff'] = nppc_comparison['Percentage_moslin'] - nppc_comparison['Percentage_ot']

# Calculate average difference and t-test
col12_diff = col12_comparison['Diff'].mean()
col12_t = stats.ttest_1samp(a = col12_comparison['Diff'], popmean = 0, alternative = 'greater')
nppc_diff = nppc_comparison['Diff'].mean()
nppc_t = stats.ttest_1samp(a = nppc_comparison['Diff'], popmean = 0, alternative = 'greater')
print(f'Col12 average difference: {col12_diff:.2f}; p-value: {col12_t.pvalue:.2E}')
print(f'Nppc average difference: {nppc_diff:.2f}; p-value: {nppc_t.pvalue:.2E}')


# In[101]:


# Cleveland plot for col12 (see https://stats.stackexchange.com/questions/423735/what-is-the-name-of-this-plot-that-has-rows-with-two-connected-dots)
yrange = np.arange(1, len(col12_comparison.index) + 1)
colors = np.where(col12_comparison['Percentage_moslin'] > col12_comparison['Percentage_ot'], '#d9d9d9', '#d57883')
plt.hlines(y=yrange, xmin=col12_comparison['Percentage_ot'], xmax=col12_comparison['Percentage_moslin'], color=colors, lw = 5)
plt.scatter(col12_comparison['Percentage_ot'], yrange, s = 100, color='#029e73', label='OT', zorder=3)
plt.scatter(col12_comparison['Percentage_moslin'], yrange, s = 100, color='#0173b2', label='moslin', zorder=3)
plt.legend()
 
# Add title and axis names
plt.yticks(yrange, col12_comparison['Tree_from'] + '- ' + col12_comparison['Tree_to'])
plt.title('Correct precursors identified - col12 fibroblasts', loc='center')
plt.xlabel('Percentage')
plt.ylabel('Dataset combination')

# Show the graph
plt.savefig('../Images/Col12_moslin_OT_comparison.png', bbox_inches='tight')
plt.show()


# In[55]:


# Difference histogram col12
ax = sns.histplot(col12_comparison, x = 'Diff')
ax.set_title('Correct col12a1a precursors')
ax.set(xlabel='Difference moslin - OT (%)', ylabel='Dataset combinations')
plt.savefig('../Images/Col12_performance_differences.png', bbox_inches='tight')
plt.show()


# In[102]:


# Cleveland plot for nppc
yrange = np.arange(1, len(nppc_comparison.index) + 1)
colors = np.where(nppc_comparison['Percentage_moslin'] > nppc_comparison['Percentage_ot'], '#d9d9d9', '#d57883')
plt.hlines(y=yrange, xmin=nppc_comparison['Percentage_ot'], xmax=nppc_comparison['Percentage_moslin'], color=colors, lw = 5)
plt.scatter(nppc_comparison['Percentage_ot'], yrange, s = 100, color='#029e73', label='OT', zorder=3)
plt.scatter(nppc_comparison['Percentage_moslin'], yrange, s = 100, color='#0173b2', label='moslin', zorder=3)
plt.legend()
 
# Add title and axis names
plt.yticks(yrange, nppc_comparison['Tree_from'] + '- ' + nppc_comparison['Tree_to'])
plt.title('Correct precursors identified - nppc fibroblasts', loc='center')
plt.xlabel('Percentage')
plt.ylabel('Dataset combination')

# Show the graph
plt.savefig('../Images/nppc_moslin_OT_comparison.png', bbox_inches='tight')
plt.show()


# In[59]:


# Difference histogram nppc
ax = sns.histplot(nppc_comparison, x = 'Diff')
ax.set_title('Correct nppc precursors')
ax.set(xlabel='Difference moslin - OT (%)', ylabel='Dataset combinations')
plt.savefig('../Images/Nppc_performance_differences.png', bbox_inches='tight')
plt.show()


# # OLD CODE

# In[44]:


# Plot accuracy of best moslin, ot, lot, realistic lot
optimal_hyperparameters_and_perf =     pd.DataFrame({'algorithm' : ['moslin', 'LineageOT', 'LineageOT realistic', 'OT'],
                  'accuracy' : [moslin_optimum['accuracy'], lot_optimum['accuracy'], lot_optimum_realistic['accuracy'], ot_optimum['accuracy']],
                  'col12' : [moslin_optimum['col12_percentage'], lot_optimum['col12_percentage'], lot_optimum_realistic['col12_percentage'], ot_optimum['col12_percentage']],
                  'nppc' : [moslin_optimum['nppc_percentage'], lot_optimum['nppc_percentage'], lot_optimum_realistic['nppc_percentage'], ot_optimum['nppc_percentage']],
                  'alpha' : [moslin_optimum['alpha'], lot_optimum['alpha'], lot_optimum_realistic['alpha'], ot_optimum['alpha']],
                  'epsilon' : [moslin_optimum['epsilon'], lot_optimum['epsilon'], lot_optimum_realistic['epsilon'], ot_optimum['epsilon']],
                  'beta' : [moslin_optimum['beta'], lot_optimum['beta'], lot_optimum_realistic['beta'], ot_optimum['beta']],
                  'taua' : [moslin_optimum['taua'], lot_optimum['taua'], lot_optimum_realistic['taua'], ot_optimum['taua']],
                  'taub' : [moslin_optimum['taub'], lot_optimum['taub'], lot_optimum_realistic['taub'], ot_optimum['taub']]})
optimal_hyperparameters_and_perf_melt =    pd.melt(optimal_hyperparameters_and_perf, 
            id_vars = ['algorithm', 'accuracy', 'alpha', 'epsilon', 'beta', 'taua', 'taub'],
            var_name = 'Transient_fibroblast',
            value_name = 'Percentage_correct')
optimal_hyperparameters_and_perf


# In[45]:


ax = sns.barplot(optimal_hyperparameters_and_perf, x = 'algorithm', y = 'accuracy')
ax.set(xlabel='', ylabel='Cell type persistence accuracy')
#plt.savefig('../Images/Persistence_maximum_accuracies.png', bbox_inches='tight')
plt.show()


# In[51]:


ax = sns.barplot(optimal_hyperparameters_and_perf_melt, x = 'Transient_fibroblast', y = 'Percentage_correct', hue = 'algorithm')
ax.set(xlabel='', ylabel='Percentage correct')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
#plt.savefig('../Images/Transient_fibroblast_performance.png', bbox_inches='tight')
plt.show()


# In[41]:





# In[40]:


lot_optimum = (lot_results.sort_values(by = ['accuracy'], ascending = False)).iloc[0]
lot_optimum


# In[41]:


lot_transient_filename = [x for x in listdir(lot_optimum['Path'] + '/'+ lot_optimum['Run']) if 'transient_test' in x][0]
#lot_optimum['Path'] + '/'+ lot_optimum['Run'] + '/' + lot_transient_filename
pd.read_csv(lot_optimum['Path'] + '/'+ lot_optimum['Run'] + '/' + lot_transient_filename, index_col = 0)


# In[42]:


lot_optimum_realistic = (lot_results[lot_results['taua'] > 0.7].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
lot_optimum_realistic


# In[43]:


lot_realistic_transient_filename = [x for x in listdir(lot_optimum_realistic['Path'] + '/'+ lot_optimum_realistic['Run']) if 'transient_test' in x][0]
pd.read_csv(lot_optimum_realistic['Path'] + '/'+ lot_optimum_realistic['Run'] + '/' + lot_realistic_transient_filename, index_col = 0)


# In[13]:


ot_optimum = (ot_results.sort_values(by = ['accuracy'], ascending = False)).iloc[0]
ot_optimum


# In[14]:


ot_transient_filename = [x for x in listdir(ot_optimum['Path'] + '/'+ ot_optimum['Run']) if 'transient_test' in x][0]
pd.read_csv(ot_optimum['Path'] + '/'+ ot_optimum['Run'] + '/' + ot_transient_filename, index_col = 0)


# In[15]:


moslin_optimum = (moslin_results.sort_values(by = ['accuracy'], ascending = False)).iloc[0]
moslin_optimum


# In[16]:


moslin_transient_filename = [x for x in listdir(moslin_optimum['Path'] + '/'+ moslin_optimum['Run']) if 'transient_test' in x][0]
pd.read_csv(moslin_optimum['Path'] + '/'+ moslin_optimum['Run'] + '/' + moslin_transient_filename, index_col = 0)


# In[ ]:


cm_plot = ConfusionMatrixDisplay(persistence_cm, 
                                 display_labels = np.union1d(argmax['Cell_type_to'], 
                                                             argmax['Cell_type_from']))
cm_plot.plot()
plt.xticks(rotation=90)
#plt.savefig('../Images/Persistence_cm_tmats_18102023_ctrl.png', bbox_inches='tight')
plt.show()


# In[17]:


ot_high_eps_results = files_dt[(files_dt['Method'] == 'moslin') & (files_dt['alpha'] == 0) & (files_dt['epsilon'] > 0.01)]
ot_high_eps_optimum = (ot_high_eps_results.sort_values(by = ['accuracy'], ascending = False)).iloc[0]
ot_high_eps_optimum


# In[18]:


ot_high_eps_transient_filename = [x for x in listdir(ot_high_eps_optimum['Path'] + '/'+ ot_high_eps_optimum['Run']) if 'transient_test' in x][0]
pd.read_csv(ot_high_eps_optimum['Path'] + '/'+ ot_high_eps_optimum['Run'] + '/' + ot_high_eps_transient_filename, index_col = 0)


# In[30]:


# Plot accuracy across hyperparameters for moslin, ot, lot
## moslin
moslin_results[moslin_results['alpha'] == 0.5]


# In[21]:


moslin_optimum


# In[23]:


moslin_results[(moslin_results['epsilon'] == 0.05) & (moslin_results['beta'] == 0) & (moslin_results['taua'] == 0.4) & (moslin_results['taub'] == 1.0)]


# In[24]:


sns.barplot(moslin_results[(moslin_results['epsilon'] == 0.05) & (moslin_results['beta'] == 0) & (moslin_results['taua'] == 0.4) & (moslin_results['taub'] == 1.0)], x = "alpha", y = "accuracy")


# In[34]:


moslin_results.groupby('alpha')['accuracy'].mean()


# In[31]:


sns.violinplot(data=moslin_results, x="alpha", y="accuracy", inner="point")


# In[32]:


sns.violinplot(data=moslin_results, x="beta", y="accuracy", inner="point")


# In[33]:


sns.violinplot(data=moslin_results, x="epsilon", y="accuracy", inner="point")


# In[25]:


sns.barplot(moslin_results[(moslin_results['alpha'] == 0.01) & (moslin_results['beta'] == 0) & (moslin_results['taua'] == 0.4) & (moslin_results['taub'] == 1.0)], x = "epsilon", y = "accuracy")


# In[26]:


sns.barplot(moslin_results[(moslin_results['alpha'] == 0.01) & (moslin_results['epsilon'] == 0.05) & (moslin_results['taua'] == 0.4) & (moslin_results['taub'] == 1.0)], x = "beta", y = "accuracy")


# In[27]:


sns.barplot(moslin_results[(moslin_results['alpha'] == 0.01) & (moslin_results['epsilon'] == 0.05) & (moslin_results['beta'] == 0) & (moslin_results['taub'] == 1.0)], x = "taua", y = "accuracy")


# In[28]:


sns.barplot(moslin_results[(moslin_results['alpha'] == 0.01) & (moslin_results['epsilon'] == 0.05) & (moslin_results['beta'] == 0) & (moslin_results['taua'] == 0.4)], x = "taub", y = "accuracy")


# In[ ]:


# Plot confusion matrix for best performing hyperparameter setting


# In[ ]:


# Plot transient fibroblast results for optimal hyperparameter values


# In[ ]:





# In[19]:


moslin_results.sort_values(by = ['accuracy'], ascending = False)


# In[54]:


# Accuracy for argmax:
## Argmax has selected the highest-probability predecessor for every t2 cell. We test whether persistent cell types indeed
## mean that cells have a predecessor in the same cell type. In other words, the true value is the t2 cell type, the
## predicted value is its t1 predecessor cell type.
## Preprocessing to select only persistent cell types - only include t2 cell types that are also included in the t1 dataset
## has already been done.


# In[55]:


persistence_cm = confusion_matrix(y_true = argmax['Cell_type_to'], 
                                  y_pred = argmax['Cell_type_from'],
                                 labels = np.union1d(argmax['Cell_type_to'], 
                                                     argmax['Cell_type_from']))


# In[56]:


accuracy_score(y_true = argmax['Cell_type_to'], 
                                  y_pred = argmax['Cell_type_from'])


# In[ ]:


# Not very high? But get into loop first to scan all Zoe's results.


# In[57]:


cm_plot = ConfusionMatrixDisplay(persistence_cm, 
                                 display_labels = np.union1d(argmax['Cell_type_to'], 
                                                             argmax['Cell_type_from']))
cm_plot.plot()
plt.xticks(rotation=90)
#plt.savefig('../Images/Persistence_cm_tmats_18102023_ctrl.png', bbox_inches='tight')
plt.show()


# In[25]:


# Preprocessing
allowed_cell_types =     pd.DataFrame(argmax.groupby(['Tree_from'])['Cell_type_from'].unique()).     rename({'Cell_type_from' : 'Allowed_cell_types'}, axis = 1)
argmax_select = argmax.join(allowed_cell_types, on = 'Tree_from')
#argmax_select[argmax_select['Cell_type_to'].isin(argmax_select['Allowed_cell_types'])]
argmax_select[argmax_select.apply(lambda x: x['Cell_type_to'] in x['Allowed_cell_types'], axis = 1) == True]
#df.apply(lambda x: hilbert(x['col_1'], x['col_2']), axis=1)


# In[13]:


# Load data
#transfer_ratios = dict()
#for this_dir in [f for f in listdir(tmat_path) if not f.startswith('.')]:
    #print(this_dir)
#    transfer_ratios[this_dir] = pd.read_csv(tmat_path + '/' + this_dir + '/transfer_ratios.csv', index_col=0)


# In[4]:


# The basic test is whether cells in a t2 type are clearly from that same type at t1. We did this with a Welch test but a Wilcoxon rank test is more appropriate since the t1 probabilities are linked.
# The algorithms are very good for a wide range of hyperparameters (AUROCS are typically above 0.98) which makes it very hard to distinguish the 'best' hyperparameter settings.
# Essentially we are doing a multi-class classification task where we are classifying cell type 2 based on the cell type 1 that it is most likely to originate from (lowest p-value).
# We could, of course, use an argmax to make this into a 'proper' prediction and then print a confusion matrix over all dataset combinations to give a better overview of the data.
# Afterwards, we can calculate accuracy or F1.
# The reason to do this is the hope that accuracy/F1 could be a much more coarse-grained (and understandable) way of computing performance than the ROC-curves. 


# In[24]:


# Still, we have to start by calculating or loading p-values - for a given cell type at time t2, what is the p-value of cell type A's transition values being higher than any other cell type?
# Loading:
#results_path = '/Volumes/Macintosh HD/Users/bastiaanspanjaard/Documents/Projects/Moscot/Data/Hyperparameters_transient_fibroblasts_2'
#these_hyperparameters_path = 'moslin_alpha-0.01_epsilon-0.01_beta-0.2_taua-0.6'
#persistence_test_results = pd.read_csv(results_path + '/' + these_hyperparameters_path + '/ct_persistency_results.csv', index_col=0)
#persistence_test_results


# In[43]:


hyperpar_combination = 'moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.6'
ct_persistency_results = pd.read_csv('../Data/Hyperparameters_transient_fibroblasts_2/' +
                                     hyperpar_combination + 
                                     '/ct_persistency_results.csv', index_col = 0)
#ct_persistency_results = pd.read_parquet()
# /Hyperparameters_transient_fibroblasts_2/moslin_alpha-0.1_epsilon-0.01_beta-0.2_taua-0.9/ct_persistency_results.csv
# Python_CT_welch_for_ROC_tmats_unbalanced_alpha-0.1_epsilon-0.001_beta-0.2_taua-0.8
# Python_CT_welch_for_ROC_tmats_unbalanced_alpha-0.0_epsilon-0.001_beta-0.2_taua-0.8
# Python_CT_welch_for_ROC_tmats_18102023_for_test - for alpha-0.1_epsilon-0.01_beta-0.2_taua-0.6_taub-1.0
persistence_test_results = ct_persistency_results


# In[49]:


persistence_test_results[['t1', 'Tree_to']].drop_duplicates()


# In[44]:


#persistence_test_results.rename(columns = {'Data_t1' : 'Tree_from', 'Data_t2' : 'Tree_to', 'Welch.p' : 'Welch_p'}, inplace = True)
persistence_test_results_filter = persistence_test_results[persistence_test_results['Welch_p'] != -1]
# 23184 rows, 99 dataset combinations, 4 ctrl hearts, 9 3dpi hearts, 7 7dpi hearts -> 4*9 + 9 * 7=36 + 63 = 99.
# This should be the result for any welch test computation on the zebrafish data.
#persistence_test_results[persistence_test_results['t1'] == "3dpi"]['Tree_to'].drop_duplicates()
#(persistence_test_results['Welch_p'] == -1).sum()
# No -1's.

# For /Hyperparameters_transient_fibroblasts_2/moslin_alpha-0.1_epsilon-0.01_beta-0.2_taua-0.9/ct_persistency_results.csv:
# 71190 rows, 67653 failed Welch tests.
print('For ' + hyperpar_combination + ":")
print(str((persistence_test_results['Welch_p'] == -1).sum()) + " out of", str(persistence_test_results.shape[0]) + 
      " Welch tests failed (expected: 0 out of 23184), resulting in " + str(persistence_test_results_filter.shape[0]) +
     " successful Welch tests (expected: 23184)")
datasets_ctrl = persistence_test_results_filter[persistence_test_results_filter['t1'] == "Ctrl"]['Tree_from'].drop_duplicates().shape[0]
datasets_3dpi_to = persistence_test_results_filter[persistence_test_results_filter['t1'] == "Ctrl"]['Tree_to'].drop_duplicates().shape[0]
datasets_3dpi_from = persistence_test_results_filter[persistence_test_results_filter['t1'] == "3dpi"]['Tree_from'].drop_duplicates().shape[0]
datasets_7dpi = persistence_test_results_filter[persistence_test_results_filter['t1'] == "3dpi"]['Tree_to'].drop_duplicates().shape[0]
print(str(datasets_ctrl) + " to " + str(datasets_3dpi_to) + " from control (should be 4 and 9), " +
     str(datasets_3dpi_from) + " to " + str(datasets_7dpi) + " from 3dpi (should be 9 and 7)")
combinations_found = persistence_test_results_filter[['Tree_from', 'Tree_to']].drop_duplicates().shape[0]
print(str(datasets_ctrl * datasets_3dpi_to + datasets_3dpi_from * datasets_7dpi) + " combinations expected, " +
      str(combinations_found) + " found.")


# In[ ]:


# moslin_alpha-0.1_epsilon-0.01_beta-0.2_taua-0.9 has a lot of failed Welch tests but in the end we have successful Welch tests
# across almost every combination - 90 combinations, 3537 successful Welch tests.
# Exactly the same for moslin_alpha-0.0_epsilon-0.01_beta-0.2_taua-0.9
# Only 2162 successful Welch tests for moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.6.


# In[51]:


persistence_test_ctrl = persistence_test_results[persistence_test_results['t1'] == 'Ctrl']
cell_types_ctrl_from = pd.DataFrame(persistence_test_ctrl.groupby(['Tree_from', 'Tree_to'])['Cell_type_from'].unique().reset_index())
cell_types_ctrl_to = pd.DataFrame(persistence_test_ctrl.groupby(['Tree_from', 'Tree_to'])['Cell_type_to'].unique().reset_index())
cell_types_shared_ctrl_3dpi = cell_types_ctrl_from.merge(cell_types_ctrl_to, how = 'inner', on = ['Tree_from', 'Tree_to'])
cell_types_shared_ctrl_3dpi['Shared_cell_types'] = cell_types_shared_ctrl_3dpi.apply(lambda x: np.intersect1d(x['Cell_type_from'], x['Cell_type_to']), axis = 1)
#cell_types_shared_ctrl_3dpi
#(cell_types_shared_ctrl_3dpi.iloc[4]['Cell_type_from'], cell_types_shared_ctrl_3dpi.iloc[4]['Cell_type_to'], cell_types_shared_ctrl_3dpi.iloc[4]['Shared_cell_types'],
#np.intersect1d(cell_types_shared_ctrl_3dpi.iloc[4]['Cell_type_from'], cell_types_shared_ctrl_3dpi.iloc[4]['Cell_type_to']))
persistence_test_ctrl = persistence_test_ctrl.merge(cell_types_shared_ctrl_3dpi[['Tree_from', 'Tree_to', 'Shared_cell_types']], on = ['Tree_from', 'Tree_to'])
persistence_test_ctrl['Persistence'] = persistence_test_ctrl.apply(lambda x: (x['Cell_type_from'] in x['Shared_cell_types']) &
                            (x['Cell_type_to'] in x['Shared_cell_types']), axis = 1)
persistence_test_ctrl = persistence_test_ctrl[persistence_test_ctrl['Persistence']]
#persistence_test_ctrl
persistence_predictions_ctrl = persistence_test_ctrl.set_index('Cell_type_from').groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Welch_p'].idxmin()
persistence_predictions_ctrl = pd.DataFrame(persistence_predictions_ctrl).reset_index().rename(columns = {'Welch_p' : 'Cell_type_from'})
f1_ctrl = f1_score(y_true = persistence_predictions_ctrl['Cell_type_to'], y_pred = persistence_predictions_ctrl['Cell_type_from'], average = 'micro')
f1_ctrl


# In[53]:


persistence_test_3dpi = persistence_test_results[persistence_test_results['t1'] == '3dpi']
cell_types_3dpi_from = pd.DataFrame(persistence_test_3dpi.groupby(['Tree_from', 'Tree_to'])['Cell_type_from'].unique().reset_index())
cell_types_3dpi_to = pd.DataFrame(persistence_test_3dpi.groupby(['Tree_from', 'Tree_to'])['Cell_type_to'].unique().reset_index())
cell_types_shared_3dpi_7dpi = cell_types_3dpi_from.merge(cell_types_3dpi_to, how = 'inner', on = ['Tree_from', 'Tree_to'])
cell_types_shared_3dpi_7dpi['Shared_cell_types'] = cell_types_shared_3dpi_7dpi.apply(lambda x: np.intersect1d(x['Cell_type_from'], x['Cell_type_to']), axis = 1)
persistence_test_3dpi = persistence_test_3dpi.merge(cell_types_shared_3dpi_7dpi[['Tree_from', 'Tree_to', 'Shared_cell_types']], on = ['Tree_from', 'Tree_to'])
persistence_test_3dpi['Persistence'] = persistence_test_3dpi.apply(lambda x: (x['Cell_type_from'] in x['Shared_cell_types']) &
                            (x['Cell_type_to'] in x['Shared_cell_types']), axis = 1)
persistence_test_3dpi = persistence_test_3dpi[persistence_test_3dpi['Persistence']]
persistence_predictions_3dpi = persistence_test_3dpi.set_index('Cell_type_from').groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Welch_p'].idxmin()
persistence_predictions_3dpi = pd.DataFrame(persistence_predictions_3dpi).reset_index().rename(columns = {'Welch_p' : 'Cell_type_from'})
f1_3dpi = f1_score(y_true = persistence_predictions_3dpi['Cell_type_to'], y_pred = persistence_predictions_3dpi['Cell_type_from'], average = 'micro')
f1_3dpi


# In[56]:


persistence_ctrl_cm = confusion_matrix(y_true = persistence_predictions_ctrl['Cell_type_to'], 
                                  y_pred = persistence_predictions_ctrl['Cell_type_from'],
                                 labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                     persistence_predictions_ctrl['Cell_type_from']))
cm_plot_ctrl = ConfusionMatrixDisplay(persistence_ctrl_cm, 
                                 display_labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                             persistence_predictions_ctrl['Cell_type_from']))
cm_plot_ctrl.plot()
plt.xticks(rotation=90)
plt.savefig('../Images/Persistence_cm_tmats_18102023_ctrl.png', bbox_inches='tight')
plt.show()


# In[57]:


persistence_3dpi_cm = confusion_matrix(y_true = persistence_predictions_3dpi['Cell_type_to'], 
                                  y_pred = persistence_predictions_3dpi['Cell_type_from'],
                                 labels = np.union1d(persistence_predictions_3dpi['Cell_type_to'], 
                                                     persistence_predictions_3dpi['Cell_type_from']))
cm_plot_3dpi = ConfusionMatrixDisplay(persistence_3dpi_cm, 
                                 display_labels = np.union1d(persistence_predictions_3dpi['Cell_type_to'], 
                                                             persistence_predictions_3dpi['Cell_type_from']))
cm_plot_3dpi.plot()
plt.xticks(rotation=90)
plt.savefig('../Images/Persistence_cm_tmats_18102023_3dpi.png', bbox_inches='tight')
plt.show()


# In[ ]:


# Refactor the above
ct_persistency_results = pd.read_csv('../Data/Python_CT_welch_for_ROC_tmats_18102023_for_test', index_col = 0)
persistence_test_ctrl = persistence_test_results[persistence_test_results['t1'] == 'Ctrl']
persistence_predictions_ctrl = PredictPersistence(...)

persistence_test_3dpi = persistence_test_results[persistence_test_results['t1'] == '3dpi']
persistence_predictions_3dpi = PredictPersistence(...)


# In[42]:


persistence_test_ctrl
# Merge with existing cell types. Remove line if cell type does not exit. Continue by finding idxmin.
persistence_test_ctrl = persistence_test_ctrl.merge(cell_types_shared_ctrl_3dpi[['Tree_from', 'Tree_to', 'Shared_cell_types']], on = ['Tree_from', 'Tree_to'])
persistence_test_ctrl


# In[49]:


persistence_test_ctrl['Persistence'] = persistence_test_ctrl.apply(lambda x: (x['Cell_type_from'] in x['Shared_cell_types']) &
                            (x['Cell_type_to'] in x['Shared_cell_types']), axis = 1)
persistence_test_ctrl[persistence_test_ctrl['Persistence']]


# In[5]:


# Persistence test per timepoint
persistence_test_ctrl = persistence_test_results[persistence_test_results['t1'] == 'Ctrl']
shared_ct_ctrl = np.intersect1d(persistence_test_ctrl['Cell_type_to'].unique(), persistence_test_ctrl['Cell_type_from'].unique())
persistence_test_ctrl = persistence_test_ctrl[(persistence_test_ctrl['Cell_type_from'].isin(shared_ct_ctrl)) &
                                             (persistence_test_ctrl['Cell_type_to'].isin(shared_ct_ctrl))]
persistence_predictions_ctrl = persistence_test_ctrl.set_index('Cell_type_from').groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Welch_p'].idxmin()
persistence_predictions_ctrl = pd.DataFrame(persistence_predictions_ctrl).reset_index().rename(columns = {'Welch_p' : 'Cell_type_from'})
f1_ctrl = f1_score(y_true = persistence_predictions_ctrl['Cell_type_to'], y_pred = persistence_predictions_ctrl['Cell_type_from'], average = 'micro')

persistence_test_3dpi = persistence_test_results[persistence_test_results['t1'] == '3dpi']
shared_ct_3dpi = np.intersect1d(persistence_test_3dpi['Cell_type_to'].unique(), persistence_test_3dpi['Cell_type_from'].unique())
persistence_test_3dpi = persistence_test_3dpi[(persistence_test_3dpi['Cell_type_from'].isin(shared_ct_3dpi)) &
                                             (persistence_test_3dpi['Cell_type_to'].isin(shared_ct_3dpi))]
persistence_predictions_3dpi = persistence_test_3dpi.set_index('Cell_type_from').groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Welch_p'].idxmin()
persistence_predictions_3dpi = pd.DataFrame(persistence_predictions_3dpi).reset_index().rename(columns = {'Welch_p' : 'Cell_type_from'})
f1_3dpi = f1_score(y_true = persistence_predictions_3dpi['Cell_type_to'], y_pred = persistence_predictions_3dpi['Cell_type_from'], average = 'micro')
(f1_ctrl, f1_3dpi)


# In[6]:


# A couple of issues, most of all that these values are very low, but also that most of the results I got from Zoe have a surprisingly
# low number of rows and lots of -1's, indicating the computation didn't work.
# Are all cell types always present at ctrl and 3dpi? Otherwise this is not a true persistence test.


# In[7]:


persistence_ctrl_cm = confusion_matrix(y_true = persistence_predictions_ctrl['Cell_type_to'], 
                                  y_pred = persistence_predictions_ctrl['Cell_type_from'],
                                 labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                     persistence_predictions_ctrl['Cell_type_from']))
cm_plot_ctrl = ConfusionMatrixDisplay(persistence_ctrl_cm, 
                                 display_labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                             persistence_predictions_ctrl['Cell_type_from']))
cm_plot_ctrl.plot()
plt.xticks(rotation=90)
plt.show()


# In[23]:


# Turn these p-values into predictions for each cell type at t2 - which cell type at t1 has the lowest p-value?
# Remove -1 p-values: these tests have not been done due to missing cell types
#persistence_test_results = persistence_test_results[~(persistence_test_results['Welch_p'] == -1)]
#persistence_test_results
# Only 3537 remaining? Seems like very few but this is true for all hyperparameter settings
# Add minimum p-value per t2 cell type, tree_from and tree_to
# Set index as cell_type_from, groupby cell_type_to, Tree_from and Tree_to, idxmin for Welch_p
#persistence_test_predictions = persistence_test_results.set_index('Cell_type_from').groupby(['Cell_type_to', 'Tree_from', 'Tree_to'])['Welch_p'].idxmin()
#persistence_test_predictions = pd.DataFrame(persistence_test_predictions).reset_index().rename(columns = {'Welch_p' : 'Cell_type_from'})
#persistence_test_predictions


# In[17]:


#persistence_cm = confusion_matrix(y_true = persistence_test_predictions['Cell_type_to'], y_pred = persistence_test_predictions['Cell_type_from'],
#                                 labels = np.union1d(persistence_test_predictions['Cell_type_to'], persistence_test_predictions['Cell_type_from']))


# In[18]:


#cm_plot = ConfusionMatrixDisplay(persistence_cm, display_labels = np.union1d(persistence_test_predictions['Cell_type_to'], persistence_test_predictions['Cell_type_from']))
#cm_plot.plot()
#plt.xticks(rotation=90)
#plt.show()


# In[19]:


#f1_score(y_true = persistence_test_predictions['Cell_type_to'], y_pred = persistence_test_predictions['Cell_type_from'], average = 'micro')


# In[ ]:


# F1 score is very low, is this the case for all hyperparameter combinations? Looks like it, yes. Check if these predictions are indeed made (are there no T-cells? Since the number of true label T-cells is 0).


# In[ ]:


#    this_shared_ct = np.intersect1d(ROC_df['Cell_type_from'].unique(), ROC_df['Cell_type_to'].unique())
#    ROC_df = ROC_df[(ROC_df['Cell_type_from'].isin(this_shared_ct)) & (ROC_df['Cell_type_to'].isin(this_shared_ct))]


# In[ ]:


# This is still quite low


# In[7]:


persistence_cm = confusion_matrix(y_true = persistence_predictions_ctrl['Cell_type_to'], 
                                  y_pred = persistence_predictions_ctrl['Cell_type_from'],
                                 labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                     persistence_predictions_ctrl['Cell_type_from']))
cm_plot = ConfusionMatrixDisplay(persistence_cm, 
                                 display_labels = np.union1d(persistence_predictions_ctrl['Cell_type_to'], 
                                                             persistence_predictions_ctrl['Cell_type_from']))
cm_plot.plot()
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# I think we have to constrain to cell types that exist at both timepoints.
# Weigh by size?

