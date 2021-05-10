import requests
import tarfile
import pathlib
import shutil
import argparse
from PIL import Image
import progress.bar


parser = argparse.ArgumentParser('fpds.download')
parser.add_argument('data_path', type=pathlib.Path, help='Data root directory e.g. /tmp/fpds.')
args = parser.parse_args()

def transform(old_line, imwidth, imheight):
    class_, x1px, x2px, y1px, y2px = map(int, old_line.split(' '))
    xmidpx = (x1px + x2px) / 2
    ymidpx = (y1px + y2px) / 2
    widthpx = x2px - x1px
    heightpx = y2px - y1px
    xmid = xmidpx / imwidth
    ymid = ymidpx / imheight
    width = widthpx / imwidth
    height = heightpx / imheight
    return f'{class_} {xmid:.6f} {ymid:.6f} {width:.6f} {height:.6f}'

def download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        bar = progress.bar.ShadyBar(max=int(r.headers['Content-Length']) // 1024**2, suffix='%(index)d/%(max)d MiB')
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): 
                f.write(chunk)
                bar.next()
        bar.finish()

def fixclass(line: str) -> str:
    if line[:2] == '-1':
        line = f'0{line[2:]}'
    return line
    

DATA_ROOT_PATH = args.data_path

TRAINSET_URL = 'https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EXQImG_yi5xOifMZYz79_hcBlxATrYEZP5mCu-li4dcWDw?&Download=1'
VALIDSET_URL = 'https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EULm_4e4bgBKqnsTxDB5Br4BKf9rApBjYi7T0QrWyJrppw??&Download=1'
TESTSET_URL = 'https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EXYxgnEftbtCp2iCgAaWDDQBcAuouxLrV_2kxBDalj3m4w?&Download=1'

TRAINSET_FILENAME = 'train.tar.gz'
VALIDSET_FILENAME = 'valid.tar.gz'
TESTSET_FILENAME = 'test.tar.gz'

SPLITS = [
    ('train', TRAINSET_URL, TRAINSET_FILENAME),
    ('valid', VALIDSET_URL, VALIDSET_FILENAME),
    ('test', TESTSET_URL, TESTSET_FILENAME),
]


DATA_ROOT_PATH.mkdir(parents=True)

for splitname, url, fname in SPLITS:
    print(f'INFO :: Downloading {fname}.')
    download_file(url, DATA_ROOT_PATH / fname)

    print(f'INFO :: Extracting {fname}.')
    with tarfile.open(DATA_ROOT_PATH / fname) as ftar:
        ftar.extractall(path=DATA_ROOT_PATH / 'raw' / splitname)
    
    # delete .tar.gz
    (DATA_ROOT_PATH / fname).unlink()

    label_paths = [path for path in (DATA_ROOT_PATH / 'raw' / splitname).rglob('*.txt')]

    print(f'INFO :: Transforming {splitname}.')

    (DATA_ROOT_PATH / splitname / 'labels').mkdir(parents=True)
    (DATA_ROOT_PATH / splitname / 'images').mkdir(parents=True)

    bar = progress.bar.ShadyBar(max=len(label_paths), suffix='%(index)d/%(max)d')

    for path in label_paths:
        bar.next()
        with open(path) as f:
            content = f.read().splitlines(keepends=False)

        im_path = path.with_suffix('.png')

        try:
            with Image.open(im_path) as im:
                width, height = im.width, im.height

        except FileNotFoundError:
            continue
        
        content = map(fixclass, content)
        content = '\n'.join(transform(line, width, height) for line in content)

        with open(DATA_ROOT_PATH / splitname / 'labels' / path.name, mode='w') as f:
            f.write(content)
        
        new_im_path = DATA_ROOT_PATH / splitname / 'images' / im_path.name
        im_path.link_to(new_im_path)
    bar.finish()

print(f'INFO :: Deleting raw data.')
shutil.rmtree(DATA_ROOT_PATH / 'raw')
print(f'INFO :: Finished.')
