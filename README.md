# FindingCorrelationBetweenHumansAndObjects

# TR

 
## Projenin Amacı

Proje görüntüler üzerinden insan davranışlarını analiz etmeyi amaçlar. Örnek vermek gerekirse günümüzde kullanılan nesne tespit algoritmaları sadece nesneleri tespit eder.  Projenin amacı ise sadece nesneleri tespit etmek değil, aynı zamanda insanlarla bu nesneler arasında ilişki kurmaktır. Tespit edilen bu ilişkilerde adım hesabı, kalori hesabı gibi nitelikler de tespit edilebilmektedir.


## Projenin Hedefi

İnsanlar ile nesneler arasında ilişki kurarken benzin bidonu ile yürüyen insan senaryosu düşünülmüştür. Bunun sebebi ise ülkemizde son zamanlarda çıkan yangınlar olarak gösterilebilir. Bu nesneler arasındaki ilişkinin yanı sıra tespit edilen insanın kameraya göre açısı, adım sayısı, yaktığı kalori gibi değerlerin de elde edilmesi hedeflenmiştir.

## Projede Kullanılan Yöntemler

- Projede insan vücutlarının tespiti için `“Mediapipe Pose”` kullanılmıştır.

- Nesne tespiti için ise `Yolov3` algoritması kullanılmıştır.

## Benzin Bidonlarının Tespiti İçin Kullanılan Veri seti

Yolov3 ile kullanılan veri seti 238 görüntüden oluşur. Bu görüntülerden bazıları internetten toplandı. Geri kalan kısmı ise  Mustafa Kemal GÖKÇE’ nin çektiği görüntülerden oluşur. Bu verilerin çekilmesi sırasında Huawei P20 marka telefondan yararlanılmıştır. Veriler 720p çözünürlükte 30FPS olarak çekilmiştir.

## Test Veri seti

Modeli eğitmek ve test olarak kullanmak için dairesel bir yörüngede yürüdüm.(YARIÇAPI 2- 2.5 METRE) Bu yürüme sırasında sabit bir kamera (Huawei P20 720p 30FPS) yardımıyla çekilen görüntüler veri setini oluşturdu.

 <img src="https://user-images.githubusercontent.com/46056478/146639320-5c9b5f46-f9bd-4492-b714-713eb23d224a.png" alt="Way" height="300">

#### [Hazırlanan Veri seti](/Examples)

## Yolo Model Sonuçları

<img src="https://user-images.githubusercontent.com/46056478/146639512-8731fcbe-862a-4ce0-ac0d-dc0fe76588d1.png" alt="Result1" height="200"> <img src="https://user-images.githubusercontent.com/46056478/146639520-88578777-aaf8-461b-bbc6-6fd68f1ad8ec.png" alt="Result2" height="200">

## Kalori Hesaplanması

Projede kullanılan `“Mediapipe Pose”` modelinden yararlanarak görseller üzerinden kişinin adım atarken attığı kalori hesaplanmıştır. Bu kalori hesaplanmasında kullanılan algoritma aşağıda gösterilmiştir.

`yakılan kalori = [ zaman(S)/ 60 *(MET * 3.5 * ağırlık(80 kg)] /200 `


1- Normal yürüyüş :  bir dakikada adimlar < 100 :  MET=2

2- Tempolu yürüyüş : bir dakikada adimlar 100-119 : MET=3,3 

3- Koşma : bir dakikada adimlar >119 : MET=6

## Adım Tespiti

Adımları tespit ederken kişinin ayak landmarkları üzerinden tespit yapılmıştır.  İlk başta videodan 14 adet (deneme- yanılma) frame alınır, bu alınan 14 frame içinde kişinin 30 ve 31. landmarklar arasındaki mesafenin minimum olduğu konumdan sonra aradaki mesafe tekrar artıyorsa kişinin adım attığı tespit edilmiş olur.

<img src="https://user-images.githubusercontent.com/46056478/146639786-0e7e0d9b-b302-4153-9b65-0a047c6598e4.png" alt="landmarks" height="200">

## İnsanın Kameraya Göre Yönünün Tespiti

Kameraya göre konum tespiti yaparken kişinin omuzları arasındaki mesafe(12. ve 11. landmarklar) dikkate alınmıştır. Kişinin omuzları arasındaki mesafe çok kısa ise kişi kameraya karşı yan durmuş demektir. (Mesafe kısa değilse ya ön tarafı ya da arka tarafı kameraya dönüktür.)Yan durduğu tespit edildikten sonra bu omuzların Z değerleri üzerinden kişinin hangi omzunun kameraya yakın olduğu tespit edilir. Bu sayede kişinin kameraya göre yönü belirlenmiş olur.

<img src="https://user-images.githubusercontent.com/46056478/146639840-0d66896b-7604-4d6c-8d64-375d85342550.png" alt="landmarks2" height="300">

# Proje Sonuçları

<img src="https://user-images.githubusercontent.com/46056478/146639899-dfacdd42-9733-43f8-b6a7-fc8e306d3078.png" alt="finalresult1" height="300" width="400"> <img src="https://user-images.githubusercontent.com/46056478/146639879-7b9ec6cd-c749-4eb8-ad42-34e21d38007f.png" alt="finalresult2" height="300" width="400"> <img src="https://user-images.githubusercontent.com/46056478/146639902-40a91b22-0b1a-4f43-9cb7-e298e81369d7.png" alt="finalresult3" height="300" width="400">
<img src="https://user-images.githubusercontent.com/46056478/146639906-8871c25d-f814-45b0-9a61-5051d4ae8e67.png" alt="finalresult4" >



# Kaynaklar

S. Silat and L. Sadath, "Behavioural Biometrics in Feature Profiles-Engineering Healthcare and Rehabilitation Systems," 2021 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), 2021, pp. 160-165, doi: 10.1109/ICCIKE51210.2021.9410778.
2021 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE)

https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses

Bazarevsky , Valentine and Grishchenko , Ivan. (2020) . On-device Real-time Body Pose Tracking with MediaPipe BlazePose , Google Research 
https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html

The Compendium Of Physical Activities Tracking Guide ,Prevention Research Center , University Of Carolina 
http://prevention.sph.sc.edu/tools/docs/documents_compendium.pdf

Alsaadi, Israa. (2021). Study On Most Popular Behavioral Biometrics, Advantages, Disadvantages And Recent Applications 
