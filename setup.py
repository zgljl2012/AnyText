from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.1'

setup(
    name='anytext_z',  # package name
    version=VERSION,  # package version
    description='AnyText',  # package description
    packages=find_packages("."),
    package_data={'anytext': ['font/*.ttf', 'models_yaml/*.yaml', 'ocr_recog/*.txt', 'ocr_weights/*.txt']},
    zip_safe=False,
    install_requires=[
        "pytorch-lightning>=1.5.0",
        "jieba==0.42.1",
        "subword_nmt==0.3.8",
        "sacremoses==0.0.53",
        # "tensorflow==2.13.0",
        # "typing_extensions==4.6.1"
    ]
)
