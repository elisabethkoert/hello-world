'''
this file contains variables, that should be accsessible by all functions during an experiment
'''

#different ids to create filenames with a continuous numbering

figid=0 #um mehrere plots zu speichern


#some other parameters that are especially useful when new things are tried and also help with the saving and Data handling
save = False
display = False #lets graphs be displayed, in the functions that are not especially for displaying experiments
trec = False #make sure just one timetrail is recorded 
doprint=False #allows output about what is done at the moment
folder='test' #for a specific saving location

rho_e=3 #Ohm*m  from 0.3kOhmcm
