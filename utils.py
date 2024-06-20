from shutil import rmtree
import os
from sklearn.neighbors import KDTree
from skimage.measure import regionprops, label
from skimage.morphology import diamond, ball, dilation, square
from functools import reduce
from skimage.segmentation import watershed
import numpy as np
import matplotlib.pyplot as plt
import random


def segment_cells(
        cellpose_instance, img, ch=-1):
    """Segment cells using cellpose model and remove masks based on centroid distance and axis ratio."""
    cp = cellpose_instance
    img = img[:, :, ch]
    masks, _, _, _ = cp.eval(
        img, diameter=50, flow_threshold=1.2, cellprob_threshold=1)
    return masks


def extract_cells(
        image: np.ndarray, mask: np.ndarray,
        target_size=128,
        ):
    cell_rois = []
    cell_props = []
    cell_masks = []

    props = regionprops(mask)

    for prop in props:
        y0, x0, y1, x1 = prop.bbox
        w, h = x1 - x0, y1 - y0
        long_side = max(w, h)
        if long_side > target_size:
            continue
        roi = np.zeros(
            (target_size, target_size, image.shape[2]),
            dtype=image.dtype
        )
        start_y = (target_size - h) // 2
        start_x = (target_size - w) // 2
        coords = prop.coords
        roi_y = coords[:, 0] - y0 + start_y
        roi_x = coords[:, 1] - x0 + start_x
        roi[roi_y, roi_x, :] = image[coords[:, 0], coords[:, 1], :]
        cell_rois.append(roi)
        cell_props.append(prop)
        cell_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        cell_mask[roi_y, roi_x] = 1
        cell_masks.append(cell_mask)
    return cell_rois, cell_masks, cell_props


def cc_sub(im: np.ndarray, seed: np.ndarray, connectivity=2) -> np.ndarray:
    """Subtract the Connected Components in image which overlap with seed.

    :param im: mask image to be subtract CC.
    :param seed: mask image.
    :param connectivity: connectivity to calculate label, see:
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    :return: CC in im without overlap with seed.
    """
    lb = label(seed, connectivity=connectivity)
    w = watershed(im, markers=lb, connectivity=connectivity, mask=im)
    o = w > 0
    d = im ^ o
    return d


def mask_sub(oriangal: np.ndarray,
             masks: list[np.ndarray],
             ) -> np.ndarray:
    o = oriangal
    for m in masks:
        o = cc_sub(o, m)
    return o


def coordinates_to_mask(
        points: np.ndarray, shape: tuple | None = None) -> np.ndarray:
    points = points.astype(np.int64)
    dim_max = tuple([points[:, i].max()+1 for i in range(points.shape[1])])
    if shape is None:
        shape = dim_max
    else:
        assert len(shape) == points.shape[1]
        shape = tuple([shape[i] or dim_max[i] for i in range(points.shape[1])])
    arr = np.zeros(shape, dtype=np.bool_)
    ix = tuple(points[:, d] for d in range(points.shape[1]))
    arr[ix] = True
    return arr


def cc_centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.dtype == bool:
        mask = label(mask.astype(int))
    ccs = regionprops(mask)
    centroids, labels = [], []
    for cc in ccs:
        centroids.append(cc.centroid)
        labels.append(cc.label)
    return np.array(centroids), np.array(labels)


def spots_sub(spots_a: np.ndarray, spots_b: np.ndarray, radius: int):
    assert spots_a.shape[1] == spots_b.shape[1]
    dim = spots_a.shape[1]
    assert 2 <= dim <= 3
    shape = tuple([max([int(pts[:, i].max()) for pts in [spots_a, spots_b]]) + 1
                   for i in range(dim)])
    mask_a = coordinates_to_mask(spots_a, shape)
    se = diamond(radius) if dim == 2 else ball(radius)
    mask_a = dilation(mask_a, se)
    mask_b = coordinates_to_mask(spots_b, shape)
    res_mask = mask_sub(mask_a, [mask_b])
    return cc_centroids(res_mask)[0]


def call_spots(ufish_instance, img, ch=[0, 1]):
    """Call spots using ufish model."""
    uf = ufish_instance
    spots_list = []
    for c in ch:
        spots, _ = uf.predict(img[:, :, c])
        spots = spots.values
        spots_list.append(spots)
    return spots_list


def get_merge_spots(spots_list, max_dist=2.0, ch=[0, 1]):
    """Get the coordinates of all merged spots from multiple channels."""
    spots_ch1 = spots_list[ch[0]]
    spots_ch2 = spots_list[ch[1]]
    tree = KDTree(spots_ch1)
    dist, ind = tree.query(spots_ch2, k=1)
    merge_spots = spots_ch1[ind[dist < max_dist]]
    if len(ch) == 3:
        spots_ch3 = spots_list[ch[2]]
        tree = KDTree(merge_spots)
        dist, ind = tree.query(spots_ch3, k=1)
        merge_spots = merge_spots[ind[dist < max_dist]]
    return merge_spots


def assign_spots(
        spots: np.ndarray,
        mask: np.ndarray,
        dist_th: float,
        ) -> np.ndarray:
    assert len(mask.shape) in (2, 3)
    centers, labels = cc_centroids(mask)
    assert centers.shape[1] == len(mask.shape)
    pos_each_axes = np.where(mask > 0)
    pos_ = np.c_[pos_each_axes]
    tree = KDTree(pos_)
    dist, idx = tree.query(spots)
    dist, idx = np.concatenate(dist), np.concatenate(idx)
    clost = pos_[idx, :]
    if centers.shape[1] == 2:
        mask_val = mask[clost[:, 0], clost[:, 1]]
    else:
        mask_val = mask[clost[:, 0], clost[:, 1], clost[:, 2]]
    res = mask_val
    res[dist > dist_th] = 0
    return res


def plot_cell_and_spots(img, mask, signals, colors, number):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(img)
    ax[0].axis("off")

    bmask = mask > 0
    edges = dilation(bmask > 0, diamond(1)) & ~bmask
    img_spots = img.copy()
    img_spots[edges] = 200 
    ax[1].imshow(img_spots)
    ax[1].axis("off")

    for name, spot in signals.items():
        if len(spot) > 0:  # check if there are spots
            ax[1].scatter(
                spot[:, 1], spot[:, 0], s=20,
                edgecolors='none', c=colors[name], label=name)

    channel_counts = [len(signals[channel]) for channel in signals.keys()]
    text_colors = [colors[channel] for channel in signals.keys()]
    text_items = [
        f"{count}" for _, count in
        zip(signals.keys(), channel_counts)
    ]

    for i, (text, color) in enumerate(zip(text_items, text_colors)):
        fig.text(
            -0.1 + 0.1 * i, 0.15, text, color=color, fontsize=25,
            alpha=0.8, transform=ax[1].transAxes, ha='center', va='top')

    fig.text(
        0.03, 0.97, str(number), color='white', fontsize=30,
        alpha=1, transform=ax[0].transAxes, ha='left', va='top')

    fig.tight_layout(pad=0)
    return fig


def plot_all_rois(rois, masks, signals, colors, max_plots=50):
    selected_index = list(range(len(rois)))
    if len(rois) > max_plots:
        # randomly select max_plots 
        random.shuffle(selected_index)
        selected_index = selected_index[:max_plots]
    rois = [rois[i] for i in selected_index]
    masks = [masks[i] for i in selected_index]
    signals = [signals[i] for i in selected_index]
    num_plots = len(rois)
    rows = int(np.ceil(num_plots / 5))
    cols = min(5, num_plots)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3*rows), facecolor='k')

    for i, (ax, roi, mask, signal) in enumerate(zip(axs.flat, rois, masks, signals)):
        bmask = mask > 0
        edges = dilation(bmask > 0, diamond(1)) & ~bmask
        img = roi.copy()
        img[edges, :] = 255
        ax.imshow(img)

        for name, spot in signal.items():
            if name in colors and len(spot) > 0:
                ax.scatter(spot[:, 1], spot[:, 0], s=15, edgecolors=colors[name], facecolors="None", linewidths=1, label=name)

        channel_names = list(signal.keys())
        channel_counts = [len(signal[channel]) for channel in channel_names]
        text_colors = [colors.get(channel, 'white') for channel in channel_names]
        text_items = [f"{count}" for count in channel_counts]

        for j, (text, color) in enumerate(zip(text_items, text_colors)):
            ax.text(0.05 + 0.1 * j, 0.12, text, color=color, fontsize=20, alpha=0.8, transform=ax.transAxes, ha='center', va='top')

        ax.text(0.03, 0.97, str(selected_index[i]+1), color='white', fontsize=25, alpha=1, transform=ax.transAxes, ha='left', va='top')
        ax.axis("off")

    for ax in axs.flat[num_plots:]:
        ax.set_facecolor('k')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_figs(cell_rois, cell_masks, cell_signals, res_dir):
    fig_dir = f"{res_dir}/figures"
    if os.path.exists(fig_dir):
        rmtree(f"./{fig_dir}")
    os.mkdir(f"./{fig_dir}")
    for w in range(len(cell_rois)):
        fig = plot_cell_and_spots(
            cell_rois[w], cell_masks[w], cell_signals[w],
            colors={
                "ch1": "hotpink",
                "ch2": "lime",
                "ch1+ch2": "yellow",
            },
            number=w+1)
        plt.text(50, -5, f"cell-{w+1}", fontsize=12)
        fig.savefig(f"{fig_dir}/cell-{w+1}.pdf")
        plt.close(fig)


def extract_signal_mask(
        cell_im_ch, cell_spots, quantile=25, square_size=3,
        hard_threshold=None):
    mask = np.zeros_like(cell_im_ch)
    mask[cell_spots[:, 0], cell_spots[:, 1]] = 1
    mask = dilation(mask, square(square_size))
    signals = cell_im_ch[mask > 0]
    threshold = np.percentile(signals, quantile)
    if hard_threshold is not None:
        threshold = np.max([threshold, hard_threshold])
    signal_mask = cell_im_ch > threshold
    return signal_mask


def get_signal_masks(
        ufish_instance, image, channels,
        quantile=25, square_size=3,
        mask_dilation_size=3,
        hard_threshold=None,
        ):
    signal_masks = []  # signal masks for each channel

    for ch in channels:
        cell_spots = call_spots(
            ufish_instance, image, ch=[ch])[0]
        cell_im_ch = image[:, :, ch]
        if len(cell_spots) == 0:
            signal_mask = np.zeros_like(cell_im_ch)
        else:
            signal_mask = extract_signal_mask(
                cell_im_ch, cell_spots,
                quantile=quantile, square_size=square_size,
                hard_threshold=hard_threshold)
        signal_mask = dilation(signal_mask, square(mask_dilation_size))
        signal_masks.append(signal_mask)

    merge_mask = reduce(lambda x, y: x & y, signal_masks)
    # signal masks for each channel after subtracting merged mask
    signal_masks_sub = []

    # remove connected components which with overlap with the merged
    for ch_sig_mask in signal_masks:
        signal_masks_sub.append(mask_sub(ch_sig_mask, [merge_mask]))
    signal_masks_sub = np.array(signal_masks_sub)
    return merge_mask, signal_masks_sub


def plot_on_img(img, props):
    scale = 0.01
    figsize = (img.shape[1] * scale, img.shape[0] * scale)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    for i, prop in enumerate(props):
        ax.text(
            prop.centroid[1],
            prop.centroid[0],
            f"{i+1}",
            color="white"
        )
    fig.set_tight_layout(True)
    return fig
