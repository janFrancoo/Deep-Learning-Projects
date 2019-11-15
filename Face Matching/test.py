import matplotlib.pyplot as plt

epsilon = 0.40

def verifyFace(img1, img2):
    img_1 = vgg_face_descriptor.predict(preprocess_image(path + img1))[0,:]
    img_2 = vgg_face_descriptor.predict(preprocess_image(path + img2))[0,:]
    
    cosine_similarity = findCosineSimilarity(img_1, img_2)
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img(path + img1))
    plt.xticks([])
    plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img(path + img2))
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)
    
    print("Cosine similarity: ", cosine_similarity)
    
    if(cosine_similarity < epsilon):
        print("MATCH!!!")
    else:
        print("They are not same person!")

verifyFace("angelina.jpg", "Monica.jpg")
verifyFace("angelina.jpg", "Rachel.jpg")
verifyFace("angelina.jpg", "angelina2.jpg")
verifyFace("angelina.jpg", "angelina3.jpg")
