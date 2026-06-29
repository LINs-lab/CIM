import torch
import openTSNE
import numpy as np
from torchvision import datasets, transforms, models

def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    save_dir="./tsne.png",
    **kwargs
):
    import matplotlib
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    if save_dir is not None:
        plt.savefig(save_dir, format = save_dir[-3:])


def load_tsne(
    dataset,
    model,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    save_dir="./tsne.png"
):

    # load datasetloader
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # define the extract_features
    def extract_features(image):
        with torch.no_grad():
            features = model(image)
        return features

    # Using OpenTSNE for dimensionality reduction and visualization.
    all_features = []
    all_labels = []

    for images, labels in trainloader:
        features = extract_features(images)
        all_features.append(features)
        all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Using OpenTSNE for dimensionality reduction
    embedding = openTSNE.TSNE().fit(all_features)

    # visualize
    # import matplotlib.pyplot as plt
    plot(
        x=embedding,
        y=all_labels.numpy(),
        ax=ax,
        title=title,
        draw_legend=draw_legend,
        draw_centers=draw_centers,
        draw_cluster_labels=draw_cluster_labels,
        colors=colors,
        legend_kwargs=legend_kwargs,
        label_order=label_order,
        save_dir=save_dir
    )
    print("t-SNE figure saved!")

if __name__ == "__main__":
    
    from data.utils.load_model import load_model
    from data.utils.load_dataset import load_dataset

    dataset = load_dataset(
        dataset="cifar-10",
        train=True
    )    
    model = load_model(    
        model_name="conv3",
        dataset="cifar10",
        pretrained=True,
    )
    model.eval()
    load_tsne(    
        dataset,
        model,
        ax=None,
        title="CIFAR10-CONV3",
        draw_legend=True,
        draw_centers=True,
        draw_cluster_labels=True,
        colors=None,
        legend_kwargs=None,
        label_order=list(range(10)),
        save_dir="./tsne.png"
    )