import numpy as np
import matplotlib.pyplot as plt 

dir = 'C:/Education/University of Toronto/Year 4/Zhen Lab/Z Alignment/Z_Alignment_v2/'
savedir = 'C:/Education/University of Toronto/Year 4/Zhen Lab/Z Alignment/Z_Alignment_v2/visualizations/'

pre_mpts = np.load(dir + 'mpts_700_724_filt.npy', allow_pickle=True)[()]
post_mpts = np.load(dir + 'post_mpts.npy', allow_pickle=True)[()]
start = 700
end = 724
depth = 1

def get_pairs(start, end, depth):
    
    pairs = []
    
    i = start
    while i < end:
        
        if end - i < depth:
            depth -= 1
        
        for j in range(1, depth + 1):
            pairs.append((i, i + j))
        
        i += 1
    
    return pairs

pairs = get_pairs(start, end, depth)

for pair in pairs:

    pre = np.array(pre_mpts[pair])
    post = np.array(post_mpts[pair])

    source_pre = pre[:,0]
    target_pre = pre[:,1]
    source_post = post[:,0]
    target_post = post[:,1]

    plt.figure(figsize=(20,10))
    plt.scatter(source_pre[:,0], source_pre[:,1], label = 'sources')
    plt.scatter(target_pre[:,0], target_pre[:,1], label = 'targets')
    plt.legend()
    plt.savefig(savedir + f'pre_mpts_{pair}.png')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.scatter(source_post[:,0], source_post[:,1], label = 'sources')
    plt.scatter(target_post[:,0], target_post[:,1], label = 'targets')
    plt.legend()
    plt.savefig(savedir + f'post_mpts_{pair}.png')
    plt.show()