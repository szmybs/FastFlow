import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec


def plt_density_map(imgs, mask=None, save_path=None, only_save=False, plt_duration=20):        
    fig = plt.figure(figsize=[23, 4])
    gs = matplotlib.gridspec.GridSpec(1, 22)
    
    # ax0
    ax0 = fig.add_subplot(gs[0, 0:7])
    im0 = ax0.imshow(imgs[0])
    ax0.axis("off")
    
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=0.0)
    # ax1
    ax1 = fig.add_subplot(gs[0, 7:14])
    im1 = ax1.imshow(imgs[0])
    
    alpha = 0.25
    if mask is not None:
        alpha = mask * alpha
    im1 = ax1.imshow(imgs[1], cmap='jet', norm=norm, alpha=alpha)
    ax1.axis("off")
    
    # ax2
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=0.0)
    ax2 = fig.add_subplot(gs[0, 14:21])
    im2 = ax2.imshow(imgs[1], cmap='jet', norm=norm)
    ax2.axis("off")
    
    # coloar-bar
    cbar_ax = fig.add_subplot(gs[0, -1])  
    cbar = fig.colorbar(im2, cax=cbar_ax)
            
    if save_path is None:
        plt.show()
    else:
        if not only_save:
            plt.savefig(save_path, pad_inches=0.2, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(int(plt_duration))
            plt.close()
        else:
            plt.savefig(save_path, pad_inches=0.2, bbox_inches='tight')
            plt.close()                
    return