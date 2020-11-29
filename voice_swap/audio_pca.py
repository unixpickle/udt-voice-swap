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
