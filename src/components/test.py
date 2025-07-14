import os
print(os.path.dirname(__file__) )
print(os.getcwd()       )   # Returns the current working directory
# print(os.chdir('/artifacts') )   # Changes the current working directory to 'path'
print(os.listdir('.')   )   # Lists all files and folders in the current directory
