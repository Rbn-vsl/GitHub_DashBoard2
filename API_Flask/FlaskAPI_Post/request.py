import requests
# c'est la requete qui envoie le dictionnaire

# url = 'http://localhost:5000/api'
url = 'https://rob128.pythonanywhere.com/api'
# r = requests.post(url, json={"customer_ID": 5})
r = requests.post(url=url, json={"customer_ID": 0})
print(r.json())


response = r.json()
print(response["probabilite"], response["solvabilite"])