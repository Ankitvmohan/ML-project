from setuptools import find_packages, setup
from typing import List

#This is to trigger the setup file from the requirements.txt file
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:
    '''
    This function will return the list of requirements from the requirements.txt file.
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        #To get rid of the /n spacing 
        requirements = [req.replace('\n',"") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup (
name= 'ML-Project',
version= '1.0',
author= 'Ankit',
author_email= 'ankitvmohan0401@gmail.com',
packages= find_packages(),
install_requires= get_requirements('requirements.txt')

)