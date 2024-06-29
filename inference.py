import torch
import numpy as np
from PIL import Image
from pathlib import Path
from random import shuffle
from src.models.stylegan import StyleBased_Generator
from src.models.embedders import ContentResnet, StyleResnet
from src.utils.draw import draw_word
from src.models.nlayer_discriminator import NLayerDiscriminator

from torchvision import transforms as T
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    weights_folder_name = 'stylegan(pretrained_on_content)_typeface_ocr_adv_192x64/9'
    weights_folder = f'checkpoints/{weights_folder_name}'

    model_G = StyleBased_Generator(dim_latent=512)
    model_G.load_state_dict(torch.load(f'{weights_folder}/model_G'))
    model_G.to(device)

    style_embedder = StyleResnet().to(device) 
    style_embedder.load_state_dict(torch.load(f'{weights_folder}/style_embedder'))

    content_embedder = ContentResnet().to(device)
    content_embedder.load_state_dict(torch.load(f'{weights_folder}/content_embedder'))

    model_G.eval()
    content_embedder.eval()
    style_embedder.eval()

    # # Word 'dictionary'
    # with open('br-utf8.txt', 'r', encoding='UTF-8') as f:
    #     lines = f.readlines()
    lines = ["DNDUS", "PLLL", "LOLOC", "PING", "KILL"]
    shuffle(lines)

    style_imgs = []

    # same as draw.img_to_tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((64, 192)),
    ])

    to_pil_image = T.ToPILImage()

    for img_path in Path('images').glob('*.png'):
        with Image.open(img_path) as im:
            img_style = transform(im.convert('RGB'))
            style_imgs.append(img_style)

    desired_content = []
    for i, word in enumerate(lines[:len(style_imgs)]):
        img = draw_word(word)

        img.save(f'results/sample_{i}.png')

        img_content = transform(img)
        desired_content.append(img_content)
   
    new_style_imgs = []
    for style_img in style_imgs:
        style_img = np.array(style_img)
        new_style_imgs.append(style_img)
        
    new_desired_content = []
    for desired_content_img in desired_content:
        desired_content_img = np.array(desired_content_img)
        new_desired_content.append(desired_content_img)
        
    # import pdb; pdb.set_trace()    
    
    style_imgs = torch.from_numpy(np.array(new_style_imgs))
    desired_content = torch.from_numpy(np.array(new_desired_content))

    style_imgs = style_imgs.to(device)
    desired_content = desired_content.to(device)

    style_embeds = style_embedder(style_imgs)
    content_embeds = content_embedder(desired_content)

    preds = model_G(content_embeds, style_embeds)

    for i, pred in enumerate(preds):
        img = to_pil_image(pred)
        img.save(f'results/{i}.png')

if __name__ == '__main__':
    main()
