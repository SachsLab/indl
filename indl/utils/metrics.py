from typing import List


__all__ = ['dprime', 'quickplot_history']


def dprime(y_true, y_pred, pmarg: float = 0.01, outputs: List[str] = ['dprime', 'bias', 'accuracy']) -> tuple:
    """
    Calculate D-Prime for binary data.
    70% for both classes is d=1.0488.
    Highest possible is 6.93, but effectively 4.65 for 99%

    http://www.birmingham.ac.uk/Documents/college-les/psych/vision-laboratory/sdtintro.pdf

    This function is not designed to behave as a valid 'Tensorflow metric'.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        pmarg:
        outputs: list of outputs among 'dprime', 'bias', 'accuracy'

    Returns:
        Calculated d-prime value.
    """

    import numpy as np
    from scipy.stats import norm

    # TODO: Adapt this function for tensorflow
    # y_pred = ops.convert_to_tensor(y_pred)
    # y_true = math_ops.cast(y_true, y_pred.dtype)
    # return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

    # TODO: Check that true_y only has 2 classes, and test_y is entirely within true_y classes.
    b_true = y_pred == y_true
    b_pos = np.unique(y_true, return_inverse=True)[1].astype(bool)

    true_pos = np.sum(np.logical_and(b_true, b_pos))
    true_neg = np.sum(np.logical_and(b_true, ~b_pos))
    false_pos = np.sum(np.logical_and(~b_true, b_pos))
    false_neg = np.sum(np.logical_and(~b_true, ~b_pos))

    tpr = true_pos / (true_pos + false_neg)
    tpr = max(pmarg, min(tpr, 1-pmarg))
    fpr = false_pos / (false_pos + true_neg)
    fpr = max(pmarg, min(fpr, 1 - pmarg))
    ztpr = norm.ppf(tpr, loc=0, scale=1)
    zfpr = norm.ppf(fpr, loc=0, scale=1)

    # Other measures of performance:
    # sens = tp ./ (tp+fp)
    # spec = tn ./ (tn+fn)
    # balAcc = (sens+spec)/2
    # informedness = sens+spec-1

    output = tuple()
    for out in outputs:
        if out == 'dprime':
            dprime = ztpr - zfpr
            output += (dprime,)
        elif out == 'bias':
            bias = -(ztpr + zfpr) / 2
            output += (bias,)
        elif out == 'accuracy':
            accuracy = 100 * (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg)
            output += (accuracy,)

    return output


def quickplot_history(history) -> None:
    """
    A little helper function to do a quick plot of model fit results.
    Args:
        history (tf.keras History):
    """
    import matplotlib.pyplot as plt
    if hasattr(history, 'history'):
        history = history.history
    hist_metrics = [_ for _ in history.keys() if not _.startswith('val_')]

    for m_ix, m in enumerate(hist_metrics):
        plt.subplot(len(hist_metrics), 1, m_ix + 1)
        plt.plot(history[m], label='Train')
        plt.plot(history['val_' + m], label='Valid.')
        plt.xlabel('Epoch')
        plt.ylabel(m)
    plt.legend()
    plt.tight_layout()
    plt.show()
