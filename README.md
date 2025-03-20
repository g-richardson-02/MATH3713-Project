# MATH3713-Project
This is the repository for the code used in the MATH3713 Project Report "Scaling Wind Farm Layout Optimisation to Meet UK Energy Demands". Please refer to the following setup to enable the code works as intended:

# For code labelled with (PyWake):
Create a new environment in the Anaconda Powershell Prompt

       conda update --all 
       
       conda activate 

       conda create --name pywake_env python
       
       conda activate pywake_env
       
The environment should show (pywake_env) instead of (conda) or (base)  

install PyWake library into the environment

        conda install -n pywake_env py_wake
        
  OR
  
        pip install pywake
        
The environment should now contain PyWake, this can be checked through:

        conda list

The Python files will need to be run through this environment, following this process:

        cd "paste location of python files here"

The directory should now show the file location, then to run the file:

       python filename.py

The code has print functions for each step in the algorithm, as to show the code is running as expected. For more computationally demanding code/layouts please allow some time for this to work.


# For files WITHOUT (PyWake):
Please run these files through your normal Python editor. These can also be run through the environment, although this has not been tested so there is no gurantee they work as intended.

# For NSGA-II files:
Please install DEAP onto your Python editor/conda environment. This can be done through the terminal:

       pip install deap

