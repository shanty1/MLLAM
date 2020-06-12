import os
print(os.path.dirname(__file__))
print(os.path.abspath(os.path.join(
    'os.path.basename(__file__)', "./new file/dsf/sd.dsf")))
print(os.path.abspath(
os.path.join(
    os.path.dirname(__file__), '../../save_model/model_self.pth')))
