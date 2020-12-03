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
    u, sigma, _ = np.linalg.svd(cov.mean())
    return u.T[:num_vecs] / np.sqrt(sigma[:num_vecs, None])


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
    pca_vecs = pca_vecs / np.sqrt(np.sum(pca_vecs ** 2, axis=-1, keepdims=True))
    total_mse = 0.0
    count = 0.0
    for chunk in chunks:
        proj = (pca_vecs.T @ (pca_vecs @ chunk[:, None])).flatten()
        total_mse += float(np.mean((proj - chunk) ** 2))
        count += 1.0
    return total_mse / count


def audio_chunks_apply_pca(chunks, pca_vecs, batch_size=128):
    """
    Iterate over transformed audio chunks as (chunk, pca_chunk).

    Automatically batches matrix operations for faster computation.
    """
    batch = np.zeros([batch_size, pca_vecs.shape[1]], dtype=pca_vecs.dtype)
    batch_idx = 0

    def flush_batch(batch):
        if len(batch):
            transformed = batch @ pca_vecs.T
            yield from zip(batch, transformed)

    for chunk in chunks:
        batch[batch_idx] = chunk
        batch_idx += 1
        if batch_idx == len(batch):
            yield from flush_batch(batch)
            batch_idx = 0

    yield from flush_batch(batch[:batch_idx])


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
