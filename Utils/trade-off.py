import matplotlib.pyplot as plt
import seaborn as sns

methods_name = ['LLM only', 'RAG', 'Filter', 'Filter*', 'Self-Train', 'Small-Train']
model_name = ['Llama3.2-1b', 'Mistral-7b', "Llama3-8b"]

font1 = {'size': 35, 'weight': 'bold'}
labelsize = 25
labelsize_x = 25

# mean_result_dict1 = {'LLM only': {'Fairness': [0.023], 'Utility': [0.112]},
#                      'RAG': {'Fairness': [0.571], 'Utility': [0.119]},
#                      'Filter': {'Fairness': [0.023], 'Utility': [0.114]},
#                      'Filter*': {'Fairness': [0.044], 'Utility': [0.113]},
#                      'Self-Train': {'Fairness': [], 'Utility': []},
#                      'Small-Train': {'Fairness': [0.242], 'Utility': [0.118]}}

# mean_result_dict2 = {'LLM only': {'Fairness': [0.335], 'Utility': [0.132]},
#                      'RAG': {'Fairness': [0.396], 'Utility': [0.197]},
#                      'Filter': {'Fairness': [0.425], 'Utility': [0.197]},
#                      'Filter*': {'Fairness': [0.426], 'Utility': [0.197]},
#                      'Self-Train': {'Fairness': [0.386], 'Utility': [0.195]},
#                      'Small-Train': {'Fairness': [0.375], 'Utility': [0.195]}}

# mean_result_dict3 = {'LLM only': {'Fairness': [0.059], 'Utility': [0.161]},
#                      'RAG': {'Fairness': [0.25], 'Utility': [0.214]},
#                      'Filter': {'Fairness': [0.037], 'Utility': [0.2]},
#                      'Filter*': {'Fairness': [0.042], 'Utility': [0.21]},
#                      'Self-Train': {'Fairness': [0.247], 'Utility': [0.215]},
#                      'Small-Train': {'Fairness': [0.22], 'Utility': [0.215]}}


mean_result_dict1 = {'LLM only': {'Fairness': [0.023], 'Utility': [0.112]},
                     'RAG': {'Fairness': [0.431], 'Utility': [0.121]},
                     'Filter': {'Fairness': [0.036], 'Utility': [0.105]},
                     'Filter*': {'Fairness': [0.06], 'Utility': [0.102]},
                     'Self-Train': {'Fairness': [], 'Utility': []},
                     'Small-Train': {'Fairness': [0.399], 'Utility': [0.118]}}

mean_result_dict2 = {'LLM only': {'Fairness': [0.335], 'Utility': [0.132]},
                     'RAG': {'Fairness': [0.356], 'Utility': [0.166]},
                     'Filter': {'Fairness': [0.368], 'Utility': [0.166]},
                     'Filter*': {'Fairness': [0.367], 'Utility': [0.166]},
                     'Self-Train': {'Fairness': [0.346], 'Utility': [0.166]},
                     'Small-Train': {'Fairness': [0.381], 'Utility': [0.164]}}

mean_result_dict3 = {'LLM only': {'Fairness': [0.059], 'Utility': [0.161]},
                     'RAG': {'Fairness': [0.163], 'Utility': [0.179]},
                     'Filter': {'Fairness': [0.035], 'Utility': [0.181]},
                     'Filter*': {'Fairness': [0.044], 'Utility': [0.185]},
                     'Self-Train': {'Fairness': [0.15], 'Utility': [0.178]},
                     'Small-Train': {'Fairness': [0.14], 'Utility': [0.181]}}

result_list = [mean_result_dict1, mean_result_dict2, mean_result_dict3]

total_mean_result_dict = {
    'Llama3.2-1b': result_list[0],
    'Mistral-7b': result_list[1],
    'Llama3-8b': result_list[2]
}

sns.set_theme(context="paper", style="darkgrid", font="Times New Roman")
fig, axs = plt.subplots(1, 3, figsize=(27, 9), sharex=False, sharey=False)

def get_marker(name):
    markers = {"LLM only": "o", "RAG": "s", "Filter": "<", "Filter*": "^", "Self-Train": "h", "Small-Train": "*"}
    return markers.get(name, "o")

def get_color(name):
    colors = {"LLM only": "b", "RAG": "g", "Filter": "y", "Filter*": "c", "Self-Train": "m", "Small-Train": "r"}
    return colors.get(name, "#ededf4")

sns.set_palette(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])

for i, model in enumerate(model_name):
    mean_result_dict = total_mean_result_dict[model]

    for method, method_result_dict in mean_result_dict.items():
        marker = get_marker(method)
        color = get_color(method)
        axs[i].scatter(method_result_dict['Fairness'], method_result_dict['Utility'], label=method if i == 0 else "", marker=marker, s=200, c=color)

    axs[i].tick_params(axis='both', which='both', direction='in')
    axs[i].set_ylabel('Utility (accuracy)', font1)
    axs[i].tick_params(axis='x', labelsize=labelsize_x)
    axs[i].tick_params(axis='y', labelsize=labelsize)

    if model == 'Llama3.2-1b':
        # axs[i].set_xlim(0, 0.6)
        # axs[i].set_ylim(0.109, 0.12)
        axs[i].set_xlim(0, 0.5)
        axs[i].set_ylim(0.101, 0.125)
        axs[i].set_xlabel("Fairness (agreement-ratio) \n (a) Llama3.2-1b", font1)
    elif model == 'Mistral-7b':
        # axs[i].set_xlim(0.3, 0.45)
        # axs[i].set_ylim(0.118, 0.22)
        axs[i].set_xlim(0.33, 0.39)
        axs[i].set_ylim(0.12, 0.17)
        axs[i].set_xlabel("Fairness (agreement-ratio) \n (b) Mistral-7b", font1)
    elif model == 'Llama3-8b':
        # axs[i].set_xlim(0, 0.3)
        # axs[i].set_ylim(0.158, 0.22)
        axs[i].set_xlim(0.025, 0.175)
        axs[i].set_ylim(0.15, 0.20)
        axs[i].set_xlabel("Fairness (agreement-ratio) \n (c) Llama3-8b", font1)

legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=30)
for text in legend.get_texts():
    text.set_fontweight("bold")

legend.get_frame().set_height(0.15)
plt.show()
fig.savefig("4.pdf", bbox_inches='tight')
