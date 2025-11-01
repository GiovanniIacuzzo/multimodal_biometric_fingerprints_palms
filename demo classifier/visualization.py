import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

# -------------------------------
# Conteggi classi e cluster
# -------------------------------
def plot_class_distribution(final_results):
    df = pd.DataFrame(final_results, columns=["filename","path","global_class","cluster_in_class"])
    
    # Distribuzione classi globali
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="global_class", palette="Set2", hue="global_class", dodge=False)
    plt.title("Distribuzione classi globali")
    plt.show()
    
    # Distribuzione cluster interni
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x="cluster_in_class", palette="tab20", hue="cluster_in_class", dodge=False)
    plt.title("Distribuzione cluster interni")
    plt.show()

# -------------------------------
# t-SNE interattivo
# -------------------------------
def interactive_tsne(features_dict, final_results, perplexity=30, max_iter=1000):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np

    X = np.array([r[1] for cls_list in features_dict.values() for r in cls_list])
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=max_iter).fit_transform(X)

    y_global = [r[2] for r in final_results]
    y_cluster = [f"{r[2]}_C{r[3]}" for r in final_results]
    imgs_cache = [cv2.resize(cv2.imread(r[1], cv2.IMREAD_GRAYSCALE), (64,64)) for r in final_results]

    fig, ax = plt.subplots(figsize=(12,8))
    classes_sorted = sorted(list(set(y_global)))
    palette = sns.color_palette("Set1", n_colors=len(classes_sorted))
    color_map = {cls: palette[i] for i, cls in enumerate(classes_sorted)}
    colors = [color_map[cls] for cls in y_global]
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=colors, s=80, alpha=0.8)

    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in classes_sorted]
    ax.legend(handles=handles, title="Global Class", bbox_to_anchor=(1.05,1), loc='upper left')

    annot_box = AnnotationBbox(OffsetImage(imgs_cache[0], zoom=2, cmap='gray'),
                               xy=(0,0), xybox=(50,50), xycoords='data', boxcoords="offset points",
                               arrowprops=dict(arrowstyle="->"))
    annot_box.set_visible(False)
    ax.add_artist(annot_box)

    def on_click(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                annot_box.xy = X_2d[idx]
                annot_box.offsetbox = OffsetImage(imgs_cache[idx], zoom=2, cmap='gray')
                annot_box.set_visible(True)
                fig.canvas.draw_idle()
            else:
                annot_box.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    ax.set_title("t-SNE 2D projection - Global Classes")
    plt.show()
