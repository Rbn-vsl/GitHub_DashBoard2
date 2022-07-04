import requests
# c'est la requete qui envoie le dictionnaire

# http://127.0.0.1:5000/api?customer_ID=4

# url = 'http://localhost:5000/api'
url = 'http://127.0.0.1:5000/api'
# r = requests.get(url, json={"customer_ID": 5})
r = requests.get(url, data={"customer_ID": 5})
print(r.json())




