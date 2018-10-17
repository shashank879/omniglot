# Model for one-shot learning tested on Omniglot dataset

*Please refer to the original Omniglot repository https://github.com/brendenlake/omniglot for an explanation on how to unzip the data in appropriate repositories.*

The jupyter notebook that explores the results of this model can be found at [python/one-shot-classification/omniglot.ipynb](https://github.com/shashank879/omniglot/blob/master/python/one-shot-classification/omniglot.ipynb)

### Pretrained model
Download the pretrained model from this [link](https://drive.google.com/open?id=1nj7CEVWcgHDRAfw6BY3AabUo49AOv4Ap). Unzip the contents in the folder *python/one-shot-classification/AE_classifier/*.

After this, the model can be loaded as,

```python
from ae_classifier import AE_classifier

ae_cl = AE_classifier(classes, name='default')
ae_cl.load()
```

A new model can be trained using the function:

```python
ae_cl.train(images, labels)
```

The model has 2 functions that can be used for outputs:

```python
ae_cl.img_reconstruction(sample_imgs)
ae_cl.feature_distance(img1, img2)
```

Details of all the functions are provided as comments in the files.
