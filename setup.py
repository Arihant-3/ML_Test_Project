from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    '''
    Ths function will return a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
    
    

setup(
name=  "ML_Test_Project",
version= "0.1.0",
author= "Aryan Burnwal",
author_email= "aburnwal26@gmail.com",
packages= find_packages(),
install_requires=get_requirements("requirements.txt"),
description= "A Machine Learning Project Template only for testing purposes",
)