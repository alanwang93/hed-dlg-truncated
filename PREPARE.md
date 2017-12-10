## Dependencies

* Python3
* Theano

## Prepare the data

1. Firstly, create ``data`` and ``Ouput`` folder

```
mkdir data
mkdir Output
```

2. cd into ``data`` and download the datasets you desire
* Cornell Movie Dataset

  Download the processed dataset from

  ```
  https://drive.google.com/open?id=1tAyEOrseoh106XzI4rcBAUtRgvjFoSXV
  ```

  And unzip the files into ``CornellMovie`` folder.

  **Or**

  Download the raw dataset

  ```
  https://drive.google.com/open?id=1zEgtbeEcFoSKfaEQvxb2Axv7p1M1mU1I
  ```

  And unzip files into ``CornellMovie`` folder. Then *come back to the root folder* of the project and run

  ```
  python convert-text2dict.py data/CornellMovie/cornell_movie_train.txt --cutoff 20000 data/CornellMovie/Training_20000 && 

  python convert-text2dict.py data/CornellMovie/cornell_movie_val.txt --dict=data/CornellMovie/Training_20000.dict.pkl --cutoff 20000 data/CornellMovie/Validation_20000 && 

  python convert-text2dict.py data/CornellMovie/cornell_movie_test.txt --dict=data/CornellMovie/Training_20000.dict.pkl --cutoff 20000 data/CornellMovie/Test_20000
  ```

* Untuntu Dialogue Corpus

  Download the processed dataset

  ```
  wget http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip
  ```

  Unzip it into ``UnbuntuData`` folder

  ```
  unzip -o UbuntuDialogueCorpus.zip -d UbuntuData/
  ```

  ​

## Train the model

* VHRED with Cornell Movie dataset
  * With GPU

    ```
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32  python train.py --prototype prototype_cornell_VHRED > Cornell_Output.txt
    ```
  * with CPU
    ```
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32  python train.py --prototype prototype_cornell_VHRED > Cornell_Output.txt
    ```

  You can insepct the output in ``Cornell_Output.txt`` .

* VHRED with Ubuntu dataset

  * With GPU

    ```
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32  python train.py --prototype prototype_ubuntu_VHRED > Ubuntu_Output.txt
    ```

  * With CPU

    ```
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32  python train.py --prototype prototype_ubuntu_VHRED > Ubuntu_Output.txt
    ```

* Use ``--save_every_valid_iteration`` if you would like to save the model every time we perform validation

* To train a HRED model, simply change the ``VHRED`` inscipts to ``HRED`` .




## Test & Sampling

```
THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python sample.py models/Cornell_VHRED_1/Cornell_VHRED_1 data/CornellMovie/cornell_movie_test_tiny.txt models/Cornell_VHRED_1/samples.txt --beam_search --n-samples=2 --ignore-unk --verbose
```
Replace ``models/Cornell_VHRED_1/Cornell_VHRED_1`` by your model root, replace ``models/Cornell_VHRED_1/samples.txt`` by the output file.