import json
import random

from loguru import logger
from torch.utils.data import Dataset
from src.utils.draw import  draw_word_custom
from torchvision import transforms as T
from pathlib import Path

font_styles = ['VerilySerifMono.otf', 'ABeeZee-Regular.otf',]  # 'Actor-Regular.ttf', 'Adamina-Regular.ttf', 'Alef-Regular.ttf', 'Alberta-Regular.ttf', 'Almarai-Bold.ttf', 'Barlow-ExtraBold', 
# 
class CustomDataset(Dataset):
    def __init__(self, dict_file1: Path, typeface_dir: bool = False):
        '''
        Initializes the dataset with two dictionary files.
        Each dictionary file contains words that will be used to generate images.
        '''
        self.dict_file1 = dict_file1
        with open(dict_file1, 'r', encoding='utf-8') as json_file:
            self.words1 = json.load(json_file)
        with open(dict_file1, 'r', encoding='utf-8') as json_file:
            self.words2 = json.load(json_file)
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((64, 192)),
        ])
        self.augment = T.Compose([
            T.RandomInvert(),
        ])
        
        self.words1 = random.sample(self.words1, len(self.words1) // 20)
        self.words2 = random.sample(self.words2, len(self.words2) // 20)
        
        if typeface_dir:
            self.words1 = random.sample(self.words1, len(self.words1) // 2)
            self.words2 = random.sample(self.words2, len(self.words2) // 2)
            
            
        logger.info(f'Total words in dictionary 1: {len(self.words1)}')
        logger.info(f'Total words in dictionary 2: {len(self.words2)}')
    def __len__(self):
        return min(len(self.words1), len(self.words2))

    def __getitem__(self, index):
        try:
            word1 = random.choice(self.words1)
            word2 = random.choice(self.words2)

            allowed_symbols = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            word1 = ''.join([i for i in word1 if i in allowed_symbols])
            word2 = ''.join([i for i in word2 if i in allowed_symbols])

            while not word1 or not word2 or word1 == word2:
                # word1 = random.choice(self.words1)
                word2 = random.choice(self.words2)
                # word1 = ''.join([i for i in word1 if i in allowed_symbols])
                word2 = ''.join([i for i in word2 if i in allowed_symbols])

            font_style = random.choice(font_styles)
            
            img_word1 = self.transform(draw_word_custom(word1, font_style))
            img_word2 = self.transform(draw_word_custom(word2, font_style))

            img_word1 = self.augment(img_word1)
            img_word2 = self.augment(img_word2)

            content_style = word1
            content_style = ''.join([i for i in content_style if i in allowed_symbols])
            if not content_style:
                content_style = 'o'
            img_content_style = self.transform(draw_word_custom(content_style, font_style))
            
            return img_word1, img_word2, word2, img_content_style, content_style

        except Exception as e:
            logger.error(f'Exception at index {index}, {e}')
            raise e