# NLP Hoax Recoginition
Current accuracy : 83.3%

![image](https://user-images.githubusercontent.com/22569688/95654410-fc562200-0b29-11eb-8fae-42a0b2b09c96.png)

## Purpose

This program used to classify hoax from given naration for IPB Big Data Challange.

- Datasets File Name : `Data Latih/Data Latih BDC.csv`
- Datasets Structure : `ID,label,tanggal,judul,narasi,nama file gambar`

Installation

- Clone the repository to your local machine
- Navigate to the directory
- Run following command

```shell
pip install requirements.txt
```

## Documentation

- `nlp_train.py` : Python file to train Machine Learning Model
- `data_cleaning.py` : Python file to balance Datasets(The datasets given are not balanced)
