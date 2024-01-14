import matplotlib.pyplot as plt


def apply_bb(axis, x, y, w, h, color='r', **kwargs):
    props = {
        'facecolor': 'none',
        'edgecolor': color,
        'linewidth': 2,
    }
    props.update(kwargs)

    axis.add_patch(plt.Rectangle((x, y), w, h, **props))


def show_with_bbs(ax, image, bbs, color='r'):
    ax.imshow(image)

    for bb in bbs:
        apply_bb(ax, *bb, color=color)
