import numpy as np


def audio_chunk_pca(chunks, num_vecs):
    """
    Compute the principal components for audio data.

    :param chunks: an iterator over audio chunks.
    :param num_vecs: the number of principle components to capture.
    :return: a 2-D numpy array, where the outer dimension is the number of PCA
             vectors, and the inner dimension is the chunk size.
    """
    cov = OuterMean()
    for chunk in chunks:
        cov.add(chunk)
    u, _, _ = np.linalg.svd(cov.mean())
    return u.T[:num_vecs]


def audio_chunk_pca_fix_skew(pca_chunks, pca_vecs):
    """
    Adjust the sign of any PCA vectors with negative skew in place.

    Automatically modifies the columns of pca_chunks and the rows of pca_vecs
    to give every principal component positive skew.

    :param pca_chunks: an [B x N] array of audio chunks in PCA space.
    :param pca_vecs: an [N x D] array of PCA vectors.
    """
    mean = np.mean(pca_chunks, axis=0)
    skewness = np.mean((pca_chunks - mean) ** 3, axis=0)
    sign_flips = (skewness > 0).astype(pca_chunks.dtype) * 2 - 1
    pca_vecs *= sign_flips[:, None]
    pca_chunks *= sign_flips


def audio_chunk_pca_mse(chunks, pca_vecs, batch_size=128):
    """
    Compute the MSE of chunks after they have been compressed with PCA.
    This operation never explicitly allocates an encoded and decoded array of
    chunks, but rather performs the computation in batches.

    :param chunks: an iterator over audio chunks.
    :param pca_vecs: an [N x D] array of PCA vectors.
    :param batch_size: the number of chunks per batch, to avoid using too much
                       memory.
    :return: an MSE estimate, averaged over dimensions and chunks.
    """
    # Stored in lists to be mutable from nested function.
    total_mse = [0.0]
    count = [0.0]

    def flush_batch(batch):
        if len(batch):
            proj = (batch @ pca_vecs.T) @ pca_vecs
            total_mse[0] += np.sum(np.mean((proj - batch) ** 2, axis=-1))
            count[0] += batch.shape[0]

    batch = np.zeros([batch_size, pca_vecs.shape[1]], dtype=pca_vecs.dtype)
    batch_idx = 0
    for chunk in chunks:
        batch[batch_idx] = chunk
        batch_idx += 1
        if batch_idx == len(batch):
            flush_batch(batch)
            batch_idx = 0
    flush_batch(batch[:batch_idx])

    return total_mse[0] / count[0]


class OuterMean:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self._buffer = None
        self._cov = None
        self._count = 0
        self._buffer_count = 0

    def mean(self):
        return self._cov

    def add(self, vec):
        if self._buffer is None:
            self._buffer = np.zeros([self.batch_size, len(vec)], dtype=vec.dtype)
            self._cov = np.zeros([len(vec)] * 2, dtype=vec.dtype)
        self._buffer[self._buffer_count] = vec
        self._buffer_count += 1
        if self._buffer_count == self.batch_size:
            self.flush()

    def flush(self):
        if not self._buffer_count:
            return
        buf = self._buffer[: self._buffer_count]
        outer = buf.T @ buf
        self._cov += (outer - self._cov * self._buffer_count) / (
            self._count + self._buffer_count
        )
        self._count += self._buffer_count
        self._buffer_count = 0
