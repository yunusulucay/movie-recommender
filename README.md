# Movie Recommender

Bu projede temel amaç Clustering Algoritmalarının uygulanmasıdır. Fakat clustering ile birlikte bir dizi tavsiye sistemi oluşturulması amaçlanmıştır. Bunun yanı sıra ise text datası kullanılarak da clustering uygulanmıştır. Burada 3 farklı yöntem uygulanarak dizi tavsiye sistemi oluşturulmuştur.

1- Clustering:
Clustering içerisinde 4 farklı clustering yöntemi uygulanmıştır. Bunlar Agglomerative, K-Means, Density-Based, Gaussian Mixture Model.

2- Clustering with AutoEncoder:
AutoEncoder used when clustering. AutoEncoder encoder ve decoder biçiminde iki yapıdan oluşmaktadır. Giriş ve çıkış aynıdır. Giriş değerini bir darboğaza sokarak aynı çıkışı elde etmeye çalışır. Bu sayede ise girdi değerini en çok ifade eden featureları elde etmiş oluruz.

3- Clustering with CrossTab:
PCA kullanılmıştır. User temelli bir clustering kullanılmıştır. Her bir kullanıcı için izledi/izlemedi(0/1) şeklinde veri düzenlenerek bir clustering gerçekleştirilmiştir.

4- Association Rules Mining:
Burada ise filmlerin aralarındaki ilişkiye dayanılarak bir tahmin modeli oluşturulması amaçlanmıştır. Örneğin 2 kişi var. Bunlardan user1 {Yüzüklerin Efendisi, Kuzuların Sessizliği, Forrest Gump} user2 {Yüzüklerin Efendisi, Kuzuların Sessizliği, Jurassic Park}, user3 de Yüzüklerin Efendisi izlediyse Kuzuların Sessizliği'ni ona önermek uygun olacaktır.

![image](https://user-images.githubusercontent.com/42489236/156826723-2f71151f-1327-4152-8655-c2bf1b035de0.png)


![image](https://user-images.githubusercontent.com/42489236/156830186-432296fa-c9a0-4300-ac8e-394dcb33de2c.png)
**Comments About who watched Ace Ventura: Pet Detective and Forrest Gump movies**
%26 of people watched Ace Ventura: Pet Detective movie (antecedent support)
%50 of people watched Forrest Gump (consequent support)
%21 of people watched both of them (support)
%81 of people who watched Ace Ventura also watched Forrest Gump (confidence)
The people who watched them both is %8 more than who watched them separately (leverage)
The rate of the movies related each other is 2.68 (conviction)




