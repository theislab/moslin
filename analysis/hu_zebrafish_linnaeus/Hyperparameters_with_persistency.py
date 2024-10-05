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


# In[8]:


# Celltype colors
celltype_colors = pd.read_csv('../../data/hu_zebrafish_linnaeus/Zebrafish_cell_type_colors.csv')


# In[9]:


ct_lut = dict(zip(map(str, celltype_colors['Cell.type']), celltype_colors['color']))


# In[10]:


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


# In[11]:


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
    
    files_dt.loc[i, 'Method'] = this_run_split[0]
    files_dt.loc[i, 'alpha'] = float(this_run_split[2].split('-')[1])
    if (this_run_split[0] == "moslin") & (float(this_run_split[2].split('-')[1]) == 0):
        files_dt.loc[i, 'Method'] = "OT"
    files_dt.loc[i, 'epsilon'] = float(this_run_split[3].split('-')[1])
    files_dt.loc[i, 'beta'] = float(this_run_split[4].split('-')[1])
    files_dt.loc[i, 'taua'] = float(this_run_split[5].split('-')[1])
    files_dt.loc[i, 'taub'] = float(this_run_split[6].split('-')[1])
    files_dt.loc[i, 'accuracy'] = this_accuracy


# In[12]:


files_dt['Convergence'] = [float((re.split('_|\.csv', x))[3]) for x in files_dt['Filename']]
files_dt_all = files_dt
files_dt = files_dt[(files_dt['Convergence'] >= 0.7) & (files_dt['alpha'] != 0.05)]


# In[18]:


# Part of Supp. Fig. 18
df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin') & (files_dt['beta'] == 0) & (files_dt['taua'] == 0.4)].pivot_table(index='alpha', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)
plt.rcParams.update({'font.size': 12})
g = sns.heatmap(df_pivot, annot=True, cmap='coolwarm', vmin = 0.4, vmax = 0.8)
g.set_facecolor('grey')
plt.xlabel('$\\epsilon$')
plt.ylabel('$\\alpha$')
plt.title('Accuracy')
#plt.savefig('../../figures/hu_zebrafish_linnaeus/Grid_search_epsilon_alpha.png', bbox_inches = 'tight')
plt.rcParams.update({'font.size': 22})


# In[20]:


# Part of Supp. Fig. 18
df_pivot = files_dt[(files_dt['taub'] == 1) & (files_dt['Method'] == 'moslin') & (files_dt['beta'] == 0) & (files_dt['alpha'] == 0.01)].pivot_table(index='taua', 
                          columns='epsilon', 
                          values='accuracy', 
                          aggfunc='mean').sort_index(ascending=False)

plt.rcParams.update({'font.size': 12})

g = sns.heatmap(df_pivot, annot=True, cmap='coolwarm', vmin = 0.4, vmax = 0.8)
g.set_facecolor('grey')
plt.xlabel('$\\epsilon$')
plt.ylabel('$\\tau_a$')
plt.title('Accuracy')

#plt.savefig('../../figures/hu_zebrafish_linnaeus/Grid_search_epsilon_taua.png', bbox_inches='tight')
plt.rcParams.update({'font.size': 22})


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


# Figure 4d
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


# Figure 4e
plt.rcParams["figure.figsize"] = (6,6)
g = sns.scatterplot(files_dt_plot, x = 'accuracy', y = 'nppc_percentage', hue = 'Method', s = 80)#, height = 4, width = 4)

# Adding labels and title
plt.xlabel('Accuracy')
plt.ylabel('Correct nppc\nprecursors (%)')

# Remove legend
plt.legend([],[], frameon=False)

#plt.savefig('../Images/Nppc_vs_accuracy_hyperparameters_taub1.png', bbox_inches='tight')
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


# In[23]:


# Best hyperparameter set per method ot lot moslin:
lot_results = files_dt[files_dt['Method'] == 'lot']
ot_results = files_dt[(files_dt['Method'] == 'OT')]
moslin_results = files_dt[(files_dt['Method'] == 'moslin')]


# In[24]:


lot_optimum = (lot_results[lot_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
ot_optimum = (ot_results[ot_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]
moslin_optimum = (moslin_results[moslin_results['taub'] == 1].sort_values(by = ['accuracy'], ascending = False)).iloc[0]


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


# Part of Supp. Fig. 17c
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


# In[29]:


# Part of Supp. Fig. 17c
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


# In[35]:


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


# In[47]:


# Supp. Fig. S17b
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
# plt.savefig('../Images/Persistence_maximum_accuracies_subsampling_ci.png', bbox_inches='tight')

# Show the plot
plt.show()


# ### Accuracy confidence intervals over dataset combinations 

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


# In[56]:


# Supp. Fig. S17a
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
#plt.savefig('../Images/Persistence_maximum_accuracies_sewm95.png', bbox_inches='tight')

# Show the plot
plt.show()


# ## Transient fibroblasts

# In[52]:


optimal_hyperparameters_and_perf_melt.replace(to_replace = {'col12' : 'col12a1a'}, inplace = True)


# In[53]:


# Supp. Fig. 19b
ax = sns.barplot(optimal_hyperparameters_and_perf_melt, x = 'Transient_fibroblast', y = 'Percentage_correct', hue = 'algorithm')

# Increase y-axis range
plt.ylim(0, 99)

# Adding labels and title
plt.xlabel('')
plt.ylabel('Correct (%)')
plt.title('Transient fibroblast precursors')

# Legend
plt.legend([],[], frameon=False)

#plt.savefig('../Images/Transient_fibroblast_performance.png', bbox_inches='tight')
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


# In[55]:


# Part of Supp. Fig. 19a
# Difference histogram col12
ax = sns.histplot(col12_comparison, x = 'Diff')
ax.set_title('Correct col12a1a precursors')
ax.set(xlabel='Difference moslin - OT (%)', ylabel='Dataset combinations')
#plt.savefig('../Images/Col12_performance_differences.png', bbox_inches='tight')
plt.show()


# In[59]:


# Part of Supp. Fig. 19a
# Difference histogram nppc
ax = sns.histplot(nppc_comparison, x = 'Diff')
ax.set_title('Correct nppc precursors')
ax.set(xlabel='Difference moslin - OT (%)', ylabel='Dataset combinations')
#plt.savefig('../Images/Nppc_performance_differences.png', bbox_inches='tight')
plt.show()

