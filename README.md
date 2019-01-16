# informatiCup2019-LabWITT-EA


This repository is part of a solution for the [informatiCup 2019](http://www.informaticup.de) competition
hosted by the [Gesellschaft für Informatik](https://gi.de) (GI).
The [task](https://github.com/InformatiCup/InformatiCup2019/blob/master/Irrbilder.pdf) 
of the 14th informatiCup is to generate [adversarial examples](https://blog.openai.com/adversarial-example-research/) 
for a given neural network based classification API.

Our solution in this repository tries to solve the use case:
* The attacker has access to the dataset used to train the API classifier
* The API classifier only returns the top-5 results (class label and confidence) 
* He wants to keep the number of necessary API queries as low as possible

We solved it by using an Evolutionary Algorithm (EA) in combination with an ensemble of local 
trained models 
assuming that a generated fooling image also fools another network.

The theoretical background, implementation details and results are highlighted in our [short paper](http://jtheiner.de/a6sg26io/paper.pdf). To reproduce the results of our black-box analysis you can use the raw data ([test_images_analysis_full.csv](http://jtheiner.de/a6sg26io/test_images_analysis_full.csv)) and the respective analysis script [analyze_blackbox.py](experiments/analysis/analyze_blackbox.py).

## Install

## Prerequisites
__Python3.6__ is required, which is the default python version in __Ubuntu 18.04 LTS__.
In order to execute this tool on a target model that is provided by the GI, an API key is required 
and has to be specified in 
[config.json](
./config.json).

## Create virtual environment and install packages
```bash
cd informatiCup2019-LabWITT-EA
sudo apt install python3-venv python3-tk python3-dev
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execute
Make sure you are in the previously created virtual env, indicated by `(venv)` at the beginning of your bash. (Enter the virtual env with `source venv/bin/activate` and leave it by typing `deactivate`)
#### Random fooling image for specific class
Basic approach, random image, e.g. class index 2.
``` bash
python3 __main__.py -c 2 -l
```

![](examples/examples_readme_random_02.png "Zulässige Höchstgeschwindigkeit (50): 94%")
![](examples/examples_readme_random_02.gif "Zulässige Höchstgeschwindigkeit (50): 94%")

#### Random fooling image with Polygons for specific class
Uses random polygons with 3-5 edges to create the fooling image, e.g. class index 2.
``` bash
python3 __main__.py -l -p -c 2
```

![](examples/examples_readme_poly_02.png "Zulässige Höchstgeschwindigkeit (50): 95%")
![](examples/examples_readme_poly_02.gif "Zulässige Höchstgeschwindigkeit (50): 95%")

#### Fooling image for specific class using an input image as aid
Some classes are hard to fool, since they are most likely not in the top-5 result from a random 
image. Therefore a fooling image for a stop sign can be created by starting off with a stop sign. The input image has to be 64x64 and should be in the top 5 results of the target network.
``` bash
python3 __main__.py -o path-to-your-png -l -c 2
```

![](examples/examples_readme_original_02.png "Original: Zulässige Höchstgeschwindigkeit (50): 99%")
![](examples/examples_readme_original_fool_02.png "Zulässige Höchstgeschwindigkeit (50): 95%")
![](examples/examples_readme_original_fool_02.gif "Zulässige Höchstgeschwindigkeit (50): 95%")

#### Fooling image for any class using grey polygons
A gray fooling image with polygons is created, which class it fools is not further specified (-1).
``` bash
python3 __main__.py -p -g -l -c -1
```

![](examples/examples_readme_poly_03.png "Zulässige Höchstgeschwindigkeit (60): 98%")
![](examples/examples_readme_poly_03.gif "Zulässige Höchstgeschwindigkeit (60): 98%")

### Results
The fooling image is stored in [results](./results). Additionally a GIF showing the image evolution
 is stored and a minimal graph plots the confidence over the iterations.

### Parameter Usage
| Option | Type | Description | 
|---------------|----------|---------|
| ['-c', '--classes'] | Integer | Space separated class label list, indicating for which classes a fooling image should be produced| 
| ['-l', '--local_models'] | Flag | Use local models for fooling| 
| ['-g', '--grayscale'] | Flag | Generate grayscale images | 
| ['-p', '--poly'] | Flag | Use polygons instead of random pixels | 
| ['-o', '--original'] | Image Path | Input image as startup aid | 
| [-s, '--statistic'] | Flag | Save statistic to file |

### Further configuration
A [config](./config.json) file contains some more values which can be changed, but there is no need to do it. E.g. changing the minimum confidence of a fooling image.

### Train your own models
By default, two local networks are used for our approach.
If you want to train your own models, check out our repo:
https://github.com/MaximilianIdahl/gtsrb-models-keras.
