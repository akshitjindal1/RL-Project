import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def summary_plot(epoch, y , save_file):
    x=[]
    for i in range(epoch):
        x.append(i+1)
    plt.plot(x, y)
    plt.title("Normal Summary Vs Custom Summary")
    plt.ylabel('10 seconds bar')
    plt.xlabel('Time')
    #plt.show()
    plt.savefig('./plot_comparison/plot_vs_reward_' + save_file)
    plt.close()

def get_existing_summary(file_name):
    with open(file_name) as f:
        labels=f.read().split('\n')
    labels=labels[:-1]
    return labels

def hist_cluster(summary_path, win):
    labels = get_existing_summary(summary_path)
    frames=len(labels)
    bars=int(frames/win)
    hist_clstr=[]
    for i in range(bars):
        temp= labels[i*win: (i+1)*win]
        temp = [int(i) for i in temp]
        hist_clstr.append( np.sum(temp))
    print (len(hist_clstr))
    return hist_clstr

def plot_normal_and_custom_summary(normal_summary_path, customized_summary_path,dataset_name, video_name):
    win= 10
    hist_cluster_normal = hist_cluster(normal_summary_path, win)
    hist_cluster_custom = hist_cluster(customized_summary_path, win)
    
    pickle_base_path = f'./plot_comparison/plot_vs_reward_{dataset_name}_{video_name}'
    
    normal_summary = pickle_base_path + 'Plot_normal_summary.pkl'
    customized_summary = pickle_base_path + 'Plot_customized_summary.pkl'
    
    with open(normal_summary, 'wb') as f:
        pkl.dump(hist_cluster_normal, f)
    
    
    with open(customized_summary, 'wb') as f:
        pkl.dump(hist_cluster_custom, f)
        
    # summary_plot(len(hist_cluster_normal), hist_cluster_normal, 'Plot_normal_summary.png')
    # summary_plot(len(hist_cluster_custom), hist_cluster_custom, 'Plot_customized_summary.png')

