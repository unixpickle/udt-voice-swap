import numpy as np


def audio_chunk_pca(chunks, num_vecs):
    """
    Compute the principal components for audio data.

    :param chunks: an iterator over audio chunks.
    :param num_vecs: the number of principle components to capture.
    :return: a 2-D numpy array, where the outer dimension is the number of PCA
             vectors, and the inner dimension is the chunk size.
    """
    cov_mat = None
    count = 0.0
    for chunk in chunks:
        if cov_mat is None:
            cov_mat = np.zeros([len(chunk), len(chunk)], dtype=chunk.dtype)
        cov_mat += chunk[:, None] @ chunk[None]
        count += 1.0
    cov_mat /= count
    u, _, _ = np.linalg.svd(cov_mat)
    return u.T[:num_vecs]


def audio_chunk_pca_fix_skew(chunks, pca_vecs):
    """
    Adjust the sign of any PCA vectors wich negative skew.

    :param chunks: an iterator (with a length) over audio chunks.
    :param pca_vecs: an [N x D] array of PCA vectors.
    :return: a new [N x D] array of PCA vectors.
    """
    it = iter(chunks)

    # Compute the mean using the first half of the data.
    mean_count = len(chunks) // 2
    dot_sum = None
    for i in range(mean_count):
        chunk = next(it)
        local_dots = (pca_vecs @ chunk[:, None]).flatten()
        if dot_sum is None:
            dot_sum = local_dots
        else:
            dot_sum += local_dots
    dot_mean = dot_sum / mean_count

    # Now that we have a mean, we can compute the skew efficiently.
    skewness = np.zeros_like(dot_sum)
    for chunk in it:
        skewness += ((pca_vecs @ chunk[:, None]).flatten() - dot_mean) ** 3

    return np.where((skewness > 0)[:, None], pca_vecs, -pca_vecs)


def audio_chunk_pca_mse(chunks, pca_vecs):
    """
    Compute the MSE of the PCA-compressed chunks.

    :param chunks: an iterator over audio chunks.
    :param pca_vecs: an [N x D] array of PCA vectors.
    :return: an MSE estimate, averaged over dimensions and chunks.
    """
    total_mse = 0.0
    count = 0.0
    for chunk in chunks:
        proj = (pca_vecs.T @ (pca_vecs @ chunk[:, None])).flatten()
        total_mse += float(np.mean((proj - chunk) ** 2))
        count += 1.0
    return total_mse / count
