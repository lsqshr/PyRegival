import os
import subprocess
import re
from os.path import * 
import csv
from models import *

def viewslice(file):
    path, fname = split(file)
    subprocess.Popen(["tkmedit",'-f', file])
