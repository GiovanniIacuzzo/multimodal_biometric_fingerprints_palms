import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from demo.config import FIGURE_DIR

# -------------------------------
# Conteggi classi e cluster
# -------------------------------
def plot_class_distribution(final_results):
    df = pd.DataFrame(final_results, columns=["filename","path","global_class","cluster_in_class"])
    
    # Distribuzione classi globali
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="global_class", palette="Set2", hue="global_class", dodge=False)
    plt.title("Distribuzione classi globali")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution_global.png")
    plt.show()
    
    # Distribuzione cluster interni
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x="cluster_in_class", palette="tab20", hue="cluster_in_class", dodge=False)
    plt.title("Distribuzione cluster interni")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution_clusters.png")
    plt.show()

# -------------------------------
# t-SNE interattivo
# -------------------------------
def interactive_tsne(features_dict, final_results, FIGURE_DIR, perplexity=30, max_iter=1000):
    """
    Visualizza proiezione t-SNE 2D delle feature del dataset.
    Gestisce casi di feature costanti, NaN/Inf, e dataset piccoli.

    Parametri:
    -----------
    features_dict : dict
        {classe: [(path, feature_array), ...]}
    final_results : list of tuples
        [(filename, path, global_class, cluster_label)]
    FIGURE_DIR : str o Path
        Cartella dove salvare figura t-SNE
    perplexity : int
        Parametro t-SNE
    max_iter : int
        Numero massimo iterazioni t-SNE
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    FIGURE_DIR = Path(FIGURE_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ====================================
    # COSTRUZIONE VETTORI X
    # ====================================
    X = []
    for r in final_results:
        path = Path(r[1])
        cls = r[2]

        feats_list = [f for p,f in features_dict.get(cls, []) if Path(p) == path]
        if feats_list:
            feat = np.nan_to_num(feats_list[0], nan=0.0, posinf=0.0, neginf=0.0)
            if np.allclose(feat, 0):
                feat += np.random.rand(*feat.shape)*1e-3
            X.append(feat)
        else:
            # fallback: array di piccola dimensione con rumore
            example_feat = next(iter(features_dict.get(cls, [(None, np.zeros(10))])))[1]
            example_feat = np.nan_to_num(example_feat, nan=0.0, posinf=0.0, neginf=0.0)
            if np.allclose(example_feat, 0):
                example_feat += np.random.rand(*example_feat.shape)*1e-3
            X.append(example_feat)

    X = np.array(X, dtype=np.float32)

    # ====================================
    # SE TROPPO COSTANTI → skip TSNE
    # ====================================
    if np.allclose(np.std(X, axis=0), 0):
        print("Tutte le feature sono costanti, TSNE saltato.")
        return

    # ====================================
    # Standardizzazione + PCA
    # ====================================
    X = StandardScaler().fit_transform(X)
    n_components = min(50, X.shape[1])
    X = PCA(n_components=n_components).fit_transform(X)

    # ====================================
    # t-SNE 2D
    # ====================================
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=max_iter).fit_transform(X)

    # ====================================
    # GLOBAL & CLUSTER LABELS
    # ====================================
    y_global = [r[2] for r in final_results]
    y_cluster = [f"{r[2]}_C{r[3]}" for r in final_results]

    # ====================================
    # IMMAGINI PER ANNOTAZIONE
    # ====================================
    imgs_cache = []
    for r in final_results:
        img = cv2.imread(r[1], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64,64), dtype=np.uint8)
        else:
            img = cv2.resize(img, (64,64))
        imgs_cache.append(img)

    # ====================================
    # PLOT
    # ====================================
    fig, ax = plt.subplots(figsize=(12,8))
    classes_sorted = sorted(list(set(y_global)))
    palette = sns.color_palette("Set1", n_colors=len(classes_sorted))
    color_map = {cls: palette[i] for i, cls in enumerate(classes_sorted)}
    colors = [color_map[cls] for cls in y_global]
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=colors, s=80, alpha=0.8)

    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in classes_sorted]
    ax.legend(handles=handles, title="Global Class", bbox_to_anchor=(1.05,1), loc='upper left')

    # ====================================
    # ANNOTAZIONE INTERATTIVA
    # ====================================
    annot_box = AnnotationBbox(OffsetImage(imgs_cache[0], zoom=2, cmap='gray'),
                               xy=(0,0), xybox=(50,50), xycoords='data', boxcoords="offset points",
                               arrowprops=dict(arrowstyle="->"))
    annot_box.set_visible(False)
    ax.add_artist(annot_box)

    def on_click(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont and len(ind["ind"]) > 0:
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
    plt.tight_layout()
    plt.savefig(FIGURE_DIR/"tsne_global_classes.png")
    plt.show()
