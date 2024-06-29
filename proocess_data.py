import shutil
from pathlib import Path
import json
import random

def split_dataset(base_dir: Path, train_ratio=0.8):
    words_path = base_dir / 'words.json'
    with words_path.open('r', encoding='utf-8') as f:
        words = json.load(f)

    # Shuffle and split annotations
    items = list(words.items())
    random.shuffle(items)
    split_point = int(len(items) * train_ratio)
    train_items, val_items = items[:split_point], items[split_point:]

    train_dir = Path('data') / 'Custom' / 'train'
    val_dir = Path('data') / 'Custom' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def process_items(items, target_dir):
        new_words = {}
        for ann_id, word in items:
            new_words[ann_id] = word
            ann_id = str(ann_id)
            src_img_path = base_dir / f'{ann_id}.png'
            dst_img_path = target_dir / f'{ann_id}.png'
            shutil.move(src_img_path, dst_img_path)
        with (target_dir / 'words.json').open('w', encoding='utf-8') as f:
            json.dump(new_words, f, ensure_ascii=False)

    process_items(train_items, train_dir)
    process_items(val_items, val_dir)

if __name__ == '__main__':
    base_dir = Path('images/whole')
    split_dataset(base_dir)