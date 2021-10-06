# pytorch-image-classifier

<img src="./assets/pexels-tara-winstead-8386440.jpg">

> Photo by [Tara Winstead](https://www.pexels.com/@tara-winstead?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/robot-pointing-on-a-wall-8386440/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)



## Setup

To be able to try the code you will need to

1. Download the [`flowers.zip`](https://www.mediafire.com/file/87yctfoff1sqi8n/flowers.zip/file) file
2. Extract it to the root of the project as `flowers/`
3. Install the dependencies
**Using Anaconda (recommended)**
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c anaconda pillow 
```

The `flowers.zip` has 3 folders that contain our flower images
- `train` this is our training data
- `valid` this is the data used for evaluating our classifier accuracy during training
- `test`  used for sanity checking 

The `cat_to_name.json` has the mappings between the flower ids and their actual names

## 📖 Train your classifier
Now it is time to do fun stuff!

Lets train our classifier

**If you have CUDA compatible gpu**
```bash
python train.py flowers --epochs=15 --gpu
```

**Otherwise**
```bash
python train.py flowers --epochs=15
```

This will start the training process for each epoch the tool will train the classifier and will evaluate the classifier accuracy.

When the training is completed the tool will save a checkpoint in `checkpoints/alexnet_checkpoint.pth`

We will need this for making predictions later!