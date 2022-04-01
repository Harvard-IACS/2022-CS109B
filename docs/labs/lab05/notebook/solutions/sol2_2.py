fig, axs = plt.subplots(1,3, figsize=(18,5))
                       
axs[0].loglog(history.history['loss'],linewidth=4, label = 'Training')
axs[0].loglog(history.history['val_loss'],linewidth=4, label = 'Validation', alpha=0.7)
axs[0].set_ylabel('Loss')


axs[1].plot(history.history['acc'], label='Training')
axs[1].plot(history.history['val_acc'], label='Validation')

axs[2].plot(history.history['auc'], label='Training')
axs[2].plot(history.history['val_auc'], label='Validation')

titles = ['Categorical Crossentropy Loss', 'Accuracy', 'AUC']
for ax, title in zip(axs, titles):
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    ax.legend()