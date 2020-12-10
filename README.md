# udt-voice-swap

This project is an experiment in using a [linear image-to-image translation algorithm](https://arxiv.org/abs/2007.12568) for unpaired audio data.

# How it works

In this setup, we have two domains (e.g. audio files of different speakers) and we would like to translate from one to the other without any matching pairs.

The orthogonal UDT algorithm is quite simple. First, we apply PCA separately to each domain, and sort the principal components by eigenvalue. If the domains are exact rotations of each other, then the (sorted) principal components of one domain should exactly correspond to those of the other, and we can immediately derive a rotation matrix between the two. If there is some amount of noise in data collection/pairing, then we can tune a rotation matrix to better match up pairs of closest points between the two domains.

In the audio domain, we can't directly apply the orthogonal UDT approach described above because of phase shifts. In particular, we should expect to see many phase shifted variations of the largest principal component, and this makes the ordering of the principal components ambiguous. As a result, I found that very few data points between the two domains matched up well in PCA space, and that it was pretty much impossible to iteratively learn a good rotation matrix between the two.

To address the above issue, I needed a non-linear transformation that would abstract away phase information, and potentially other kinds of redundant information as well. I turned to [MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).), a set of audio features commonly used in signal-processing to analyze speech. By computing MFCCs before applying PCA, we can get a more well-behaved set of principal components with nicely descending eigenvalues.

# Usage

If you have two directories `data/domain_1` and `data/domain_2` containing one or more audio files, you can learn a rotation between the two domains like so:

```
python learn_mapping.py \
    --sample_rate 22050 \
    --chunk_size 22050 \
    --num_chunks 200000 \
    --max_it 1000 \
    --pca_dim 2000 \
    --output udt_22050.npz \
    data/domain_1 data/domain_2
```

In the above example, we use a raw audio sample rate of 22050, and operate on chunks of 22050 samples (i.e. one second of audio). Since we load 200,000 chunks per domain, this script may take several hours to complete. In practice, I didn't find a big advantage from using this much data, so you may get away with less.

Once we have computed a transformation, we can apply it to an audio file:

```
python apply_mapping.py \
    --sample_rate 22050 \
    --chunk_size 22005 \
    udt_22050.npz \
    data/domain_1/input.mp3 output.mp3
```

# Results

The results of this model are not particularly satisfying. I downloaded several audio books with different speakers, and tried translating a clip from one audio book to the other. Here is the original clip:

[original clip link](samples/original.mp3)

Here is the reconstruction of the clip from `signal -> MFCC -> PCA -> MFCC -> signal`:

[reconstructed clip](samples/reconstruction.mp3)

Here is a version of the clip "translated" into a different audio book domain. The result sounds completely unintelligible:

[translated clip](samples/translated.mp3)
