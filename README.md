# nlp_2020

![image](https://img.shields.io/pypi/v/nlp_2020.svg)

![image](https://img.shields.io/travis/ustcsse308/nlp_2020.svg)

![image](https://readthedocs.org/projects/nlp-2020/badge/?version=latest)

Python Boilerplate contains all the boilerplate you need to create a Python package. 

* Free software: MIT license
* Documentation: https://nlp-2020.readthedocs.io.

## Baseline

### Dataset
 
Five categories: `news_culture, news_car, news_edu, news_house, news_agriculture` . 
Format with `example_id, category_code(non-sense), category, example` . 

``` sh
# Example with fields separated by '\t'
6523865677881672199	101	news_culture	黄氏祖训、家训——黄姓人家可以鉴读一下
```

### usage

Modify `scripts/train_classification.sh` before training/reproducing this baseline. 
NOTE: this baseline is **not finished yet**, that means you'll find many useless or redundant features. 

``` sh
# Install package if you want use this baseline
pip install -e ./ --no-binary :all:
```

### pkgs

packages not list in `requirements_dev.txt` 

``` 
pytorch==1.4.0
cudatoolkit==9.2
tensorboard==2.2.1
scikit-learn==0.22
jieba==0.42.1
```

### visualize baseline

``` sh
tensorboard --logdir=./runs
```

### Reference

Data link: [link](https://pan.baidu.com/s/1TprekQac-yzNHMsREWZe9g), Verification Code: uhxt  
Pretrained-embedding: [link](https://pan.baidu.com/s/1svFOwFBKnnlsqrF1t99Lnw)  
Reference: https://github.com/Embedding/Chinese-Word-Vectors 

## Features

* TODO

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)  project template. 
