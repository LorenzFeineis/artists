from augm import get_augmentation


artists_info = pd.read_csv('artists.csv')


os.chdir('/resized') ### working directory is now /resized
print(os.getcwd())

print("# of pictures in folder resized:", len(os.listdir()))


### Create list of artists
artists = []
for artist in artists_info["name"]:
    artists.append(artist.replace("ü", "ue")) ### replace ü with u because of Dürer

### Create dictionary with keys: artists, values: id
### Create dictionary with keys: artists, values: id
artist2id = dict(zip
                 (artists,artists_info["id"]))
id2artist = dict(zip
                 (artists_info["id"], artists))

index = 0
for image in tqdm(os.listdir()):
    index +=1
    parts = image.split("_")
    artist = parts[0]
    for part in parts[1:-1]:
        artist += " " + part
    if artist[0:8] == "Albrecht":
        artist = "Albrecht Duerer"
    try:
        label = artist2id[artist]
        labeled_vec = label_vector.copy()
        labeled_vec[label] = 1
        image_jpg = Image.open(image)
        dataset=np.array([[image_jpg, labeled_vec]])
        get_augmentation(dataset,id2artist,index)
        image_jpg.close()

    except KeyError:
        print(artist,"not found in dictionary")
