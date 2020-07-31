# Untapped Energy 

## An Interactive Introduction to Unsupervised Learning

This is the repository to accompany my presentation at Untapped Energy on January 21, 2021, which provides an interactive introduction to unsupervised learning with Python.

You've probably been introduced to supervised learning techniques, where you try to predict a numeric, binary, or categorical target variable from numerous features. But what if you don't have a target? What if you're trying to reduce the dimensionality of your data, find similar observations, or identify patterns from many parameters? Unsupervised learning may be less widely understood in the energy industry than its supervised partner, but it can be incredibly useful. This provides a hands on introduction to some of the most popular unsupervised learning techniques, with a focus on demystifying some of the terminology and providing a simple, intuitive understanding of the fundamental concepts. 

The agenda includes:

1. An overview of the concepts of similarity and distance, introduced through distance metrics and the k-nearest neighbors algorithm.

2. An introduction to principal component analysis through two exercises - reducing the dimensionality of a stack of geology maps and looking at well completion parameters in British Columbia.

3. A brief introduction to partitional (k-means), density (DBSCAN), and hierarchical clustering algorithms, along with some measures of clustering 'goodness' using well completion parameters.


Each section runs in a separate Jupyter notebook. These are designed to work with minimal dependencies on a windows/mac operating system with the Anaconda distribution. To keep things simple we've limited the packages to Pandas, Numpy, Matplotlib, Scipy and Scikit-Learn.

You have two options to get up and running before the workshop:

a) Install the miniconda distribution and the requirements. Navigate to https://docs.conda.io/en/latest/miniconda.html and get miniconda running. Then run: `conda install -r requirements.txt` from conda bash.

b) Install the full blown anaconda package from here: https://www.anaconda.com/distribution/

Before the workshop please make sure that:

- You have a working [Anaconda distribution](https://www.anaconda.com/distribution/) and can launch [Jupyter notebook](https://jupyter.org/)
- You can access this repository from [The Github repository](https://github.com/ScottHMcKean/untapped_energy_clustering), either through cloning it to your hard drive, or downloading it as a .zip file
- Can import the packages in each notebook and run the notebooks start to finish


