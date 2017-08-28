wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh;
export PATH="$HOME/miniconda/bin:$PATH"
rehash
conda install conda-build
VERSION=TRAVIS_TAG conda build tsfresh -c conda-forge --old-build-string
conda install anaconda-client
anaconda -t $CONDA_UPLOAD_TOKEN upload -u nbraun -l nightly $HOME/miniconda/conda-bld/$OS/$PKG_NAME-$TRAVIS_TAG-0.tar.bz2 --force
