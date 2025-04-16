# Adds src/ to sys.path so modules can be imported cleanly
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
