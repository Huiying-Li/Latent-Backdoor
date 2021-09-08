# Latent Backdoor Attacks on Deep Neural Networks

This is the documentation of the `Tensorflow/Keras` implementation of Latent Backdoor Attacks. Please see the paper for details [Latent Backdoor Attacks on Deep Neural Networks, CCS'19](https://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf). Please find our official website for this project [here](https://sandlab.cs.uchicago.edu/latent/).

## Dependencies

- `keras==2.3.1`
- `numpy==1.16.4`
- `tensorflow-gpu==1.14.0`
- `h5py==2.10.0`

The code has been tested on `Python 3.7`.

Click [here](https://people.cs.uchicago.edu/~huiyingli/files/data_models.zip) (890MB) to download example dataset and model.

## Directory Layout

```
latent_utils.py               # Utility functions.
pattern_gen.py                # Trigger optimization utility functions.
vggface_pubfig_attack.py      # Example script to perform attack.
data/                         # Directory to store data.
    data.txt                  # Put PubFig dataset in h5 format here, excluded because of GitHub file size limit. 
models/                       # Directory to store models.
    model.txt                 # Put VGG-Face model in h5 format here, excluded because of GitHub file size limit.
```

## Usage

The following script shows an example to attack a [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) Teacher model and then, through transfer learning, infect a Student model trained on [PubFig](http://www.cs.columbia.edu/CAVE/databases/pubfig/) dataset.

```
python vggface_pubfig_attack.py
```

The script does the following:

1. Alter Teacher model to include target class
2. Retrain Teacher model
3. Generate optimized latent backdoor trigger
4. Train latent backdoor into Teacher model
5. Transfer learning: build a Student model from the infected Teacher model
6. Train Student model on clean Student data
7. Test attack success rate on the Student model

Click [here](https://sandlab.cs.uchicago.edu/latent/infected_student.h5) (706MB) to download a copy of infected student model resulted from the script.

## Citation

Please cite the paper as follows

```
@inproceedings{yao2019latent,
  title={Latent Backdoor Attacks on Deep Neural Networks},
  author={Yao, Yuanshun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle={Proc. of CCS},
  year={2019},
}
```

## Contact

Huiying Li ([huiyingli@uchicago.edu](mailto:huiyingli@cs.uchicago.edu))

Kevin Yao ([ysyao@cs.uchicago.edu](mailto:ysyao@cs.uchicago.edu))
