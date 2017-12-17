from urllib import request
import secrets

for i in range(866, 1000):
    try:
        path = 'test-images/' + str(i) + '.png'
        request.urlretrieve(secrets.image_url, path)
        print('donwloaded ' + str(i))
    except Exception as e:
        print(e)

