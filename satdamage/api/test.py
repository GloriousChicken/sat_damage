import time
import requests

deb = time.time()
url = 'http://127.0.0.1:8000/predict'
files = [('files', open('/home/tonio/code/GloriousChicken/sat_damage/data/samples/images/hurricane-florence_00000147_pre_disaster.tif', 'rb')),
    ('files', open('/home/tonio/code/GloriousChicken/sat_damage/data/samples/images/hurricane-florence_00000147_post_disaster.tif', 'rb')),
    ('files', open('/home/tonio/code/GloriousChicken/sat_damage/data/samples/labels/hurricane-florence_00000147_pre_disaster.json', 'rb')),
    ('files', open('/home/tonio/code/GloriousChicken/sat_damage/data/samples/labels/hurricane-florence_00000147_post_disaster.json', 'rb'))
    ]
resp = requests.post(url=url, files=files, timeout=60)
print(resp.status_code)
print(resp.text)
if resp.ok and resp.text:
    print(resp.json())
fin = time.time()
print(f"Request completed in {fin - deb:.2f} seconds.")
