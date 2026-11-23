# Pinho et al. (2009), distributed with permission

This directory holds the original 2009 release of the minimal-absent-words software of
A. J. Pinho, P. J. S. G. Ferreira, S. P. Garcia and J. M. O. S. Rodrigues (University of Aveiro):

> Pinho, A. J., Ferreira, P. J. S. G., Garcia, S. P., & Rodrigues, J. M. O. S. (2009).
> On finding minimal absent words. *BMC Bioinformatics* 10, 137.

It is the software of our 2015 study of minimal absent words, and the reference against which the
paper in `paper/` measures ten years of progress. Its original site is offline. **Armando J. Pinho
kindly agreed (October 2026) that the original code be shared as part of this project**, and we thank
him for it. Please credit the authors in any use of it.

* `maws-2009-original.tar.gz` is the original archive (the same file as `bench/vendor/`).
* The sources here are those of the archive, with one change for current zlib in
  `MinimalAbsentWords.c`: `gzFile *fp` became `gzFile fp`.
* The copyright notice in the source files applies; this permission covers distribution with this
  project, not the MIT license of the rest of the repository.
