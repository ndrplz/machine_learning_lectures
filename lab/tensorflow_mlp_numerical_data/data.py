import numpy as np
import csv
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader


categoricals = {
    'workclass': [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
        'State-gov', 'Without-pay', 'Never-worked', '?'
    ],
    'education': [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
        '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'
    ],
    'marital_status': [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'
    ],
    'occupation': [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'
    ],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
    'sex': ['Female', 'Male', '?'],
    'native_country': [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
        'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
        'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
        'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
        'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
        'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?'
    ],
    'income': ['<=50K', '>50K']
}


class Person(object):

    def __init__(self, age, workclass, fnlwgt, education, education_num,
                 marital_status, occupation, relationship, race, sex, capital_gain,
                 capital_loss, hours_per_week, native_country, income):
        self.age = int(age)
        self.workclass = workclass
        self.fnlwgt = int(fnlwgt)
        self.education = education
        self.education_num = int(education_num)
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = int(capital_gain)
        self.capital_loss = int(capital_loss)
        self.hours_per_week = int(hours_per_week)
        self.native_country = native_country
        self.income = income

    @staticmethod
    def to_categorical(key, value):
        values = categoricals[key]
        cat = np.zeros(shape=len(values))
        cat[values.index(value)] = 1
        return cat

    @property
    def to_numeric(self):
        list = [[self.age],
                self.to_categorical('workclass', self.workclass),
                [self.fnlwgt],
                self.to_categorical('education', self.education),
                [self.education_num],
                self.to_categorical('marital_status', self.marital_status),
                self.to_categorical('occupation', self.occupation),
                self.to_categorical('relationship', self.relationship),
                self.to_categorical('race', self.race),
                self.to_categorical('sex', self.sex),
                [self.capital_gain],
                [self.capital_loss],
                [self.hours_per_week],
                self.to_categorical('native_country', self.native_country),
                self.to_categorical('income', self.income),
                ]
        return list

GoogleDriveDownloader.download_file_from_google_drive(file_id='1Dr8ybk7vEFVdZzDi_YHkFoHSQQatqduS',
                                                      dest_path='./income.zip',
                                                      overwrite=True,
                                                      unzip=True)


def load_csv(csv_name):
    with open(csv_name, 'rt') as file:
        csv_reader = csv.reader(file)

        samples = []
        for row in tqdm(csv_reader):
            row = [s.strip() for s in row]
            samples.append([item for sublist in Person(*row).to_numeric for item in sublist])

    samples = np.stack(samples)
    return samples


def load_income_dataset():

    train_samples = load_csv('income_train.csv')
    test_samples = load_csv('income_test.csv')
    x_train = train_samples[:, :-2]
    y_train = train_samples[:, -2:]
    x_test = test_samples[:, :-2]
    y_test = test_samples[:, -2:]

    x_train /= np.max(x_train + np.finfo(np.float32).eps, axis=0, keepdims=True)
    x_test /= np.max(x_train + np.finfo(np.float32).eps, axis=0, keepdims=True)

    return x_train, y_train, x_test, y_test
