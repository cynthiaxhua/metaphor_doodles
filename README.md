# Metaphor Drawings

Project Description: http://cynthiaxhua.com/art-works/metaphor-drawings

A fundamental aspect of human creativity involves mixing symbols and meanings in new ways, to generate fantasies, metaphors and imagined realities. Many existing artificial intelligence (AI) programs facilitate the translation of a word to an image in a 1:1 relationship. However, I was interested in constructing an AI that is able to draw sporadically, changing its mind partway. The machine trained for this project is able to draw more naturalistically by having the ability to focus on different subjects while still producing a single cohesive image.

![sample](https://github.com/cynthiaxhua/metaphor_doodles/blob/master/cloud_camel_final.png)

## Training a Model

`python sketch_rnn_train_cnn_tf.py --resume_training=True --log_root=checkpoint_path/lightning_sheep --data_dir=datasets/ --hparams="data_set=[lightning.npz,sheep.npz],ims_set=[lightning.npy,sheep.npy],conditional=True,num_steps=200000,kl_weight=0.0,save_every=25000"`

## Preprocessing Images for the CNN

To apply a CNN encoder, we needed to use pixel-based data in addition to the stroke-based data provided by Google Quickdraw. We feed pixel-based image files to the CNN in the form of numpy arrays derived from a grayscale bitmap of the stroke-based image. In order to generate these files, and link them to corresponding stroke-based files, we prepro- cessed the previously used stroke-based data image by im- age. Preprocessing involved generating an .svg image from the stroke-based data, converting it to png format, reading the png file back in, and converting to grayscale bitmap to save in array format. During this conversion, we centered the png image to be square (since the doodles varied in size) and scaled them down to a 48x48 size.

`python preprocess.py --data_dir=datasets/ --ds="data_set=[lightning.npz]"`


