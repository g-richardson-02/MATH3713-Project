# MATH3713-Project
This is the repository for the code used in the MATH3713 Project Report "Scaling Wind Farm Layout Optimisation to Meet UK Energy Demands". Please refer to the following setup to enable the code works as intended:

# For code labelled with (PyWake):
 - Create a new environment in the Anaconda Powershell Prompt
       conda update --all
       conda activate
The environment should now show (conda)   
       conda create --name pywake_env python
       conda activate pywake_env
The environment should show (pywake_env) instead of (conda) or (base)   
- install PyWake library into the environment
        conda install -n pywake_env py_wake
  OR
        pip install pywake
The environment should now contain PyWake, this can be checked through:
        conda list

The Python files will need to be run through this environment, following this process:
        cd "location of python files"
        
