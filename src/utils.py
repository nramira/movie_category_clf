import os
import sys
import re
import numpy as np
import pandas as pd
import dill

from nltk.corpus import stopwords
from src.exception import CustomException

def save_object(file_path, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def clean_text(texts:np.ndarray) -> list:
    try:
        stop_words = stopwords.words('english')
        cleaned_texts = []
        for features in texts:
            text = ' '.join(features)
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower()) # remove special characters 
            words = text.split()
            words = [word for word in words if word not in stop_words] # remove stop words
            cleaned_texts.append(' '.join(words))

        return cleaned_texts
    
    except Exception as e:
        raise CustomException(e, sys)