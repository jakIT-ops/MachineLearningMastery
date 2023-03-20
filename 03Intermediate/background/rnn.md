# Crash Course in Recurrent Neural Networks for Deep Learning

Another type of neural network is dominating difficult machine learning problems involving sequences of inputs: recurrent neural networks.

Recurrent neural networks have connections that have loops, adding feedback and memory to the networks over time. This memory allows this type of network to learn and generalize across sequences of inputs rather than individual patterns.

A powerful type of Recurrent Neural Network called the Long Short-Term Memory Network has been shown to be particularly effective when stacked into a deep configuration, achieving state-of-the-art results on a diverse array of problems from language translation to automatic captioning of images and videos.

In this post, you will get a crash course in recurrent neural networks for deep learning, acquiring just enough understanding to start using LSTM networks in Python with Keras.

## Support for Sequences in Neural Networks

Жишээлбэл, хувьцааны үнэ цаг хугацааны хувьд зэрэг нэг хувьсах хугацааны цуврал бодлогыг авч үзье. Энэхүү өгөгдлийн багцыг цонхны хэмжээг (жишээ нь, 5) тодорхойлж, оролтын тогтмол хэмжээтэй цонхноос богино хугацааны таамаглал дэвшүүлж сурахад сүлжээг сургах замаар сонгодог олон давхаргат перцептрон сүлжээг урьдчилан таамаглах асуудал болгон гаргаж болно. 

Энэ нь ажиллах болно, гэхдээ маш хязгаарлагдмал. Оролтын цонх нь асуудалд санах ой нэмдэг боловч зөвхөн тодорхой тооны цэгээр хязгаарлагддаг бөгөөд асуудлын талаар хангалттай мэдлэгтэй байх ёстой. Гэнэн цонх нь таамаглал гаргахад хамаатай байж болох минут, цаг, өдрийн чиг хандлагыг харуулахгүй. Нэг таамаглалаас нөгөөд нь сүлжээ нь зөвхөн өгсөн тодорхой оролтын талаар л мэддэг.

Consider the following taxonomy of sequence problems that require mapping an input to output (taken from Andrej Karpathy).

* `One-to-Many:` зургийн тайлбарын дарааллын гаралт
* `Many-to-One:` sequence input for sentiment classification
* `Many-to-Many:` sequence in and out for machine translation
* `Synched Many-to-Many:` synced sequences in and out for video classification

Гаралтад оруулах нэг нэгээр нь оруулах жишээ нь дүрс ангилах гэх мэт таамаглалын даалгаварт зориулсан сонгодог дамжуулагч мэдрэлийн сүлжээний жишээ болохыг та харж болно.

Мэдрэлийн сүлжээн дэх дарааллыг дэмжих нь асуудлын чухал ангилал бөгөөд гүнзгий суралцах нь саяхан гайхалтай үр дүнг харуулсан нэг асуудал юм. Хамгийн сүүлийн үеийн үр дүн нь давтагдах мэдрэлийн сүлжээ гэж нэрлэгддэг дарааллын асуудалд тусгайлан зориулсан сүлжээг ашиглаж байна.


## Recurrent Neural Network 

Стандарт дамжуулагч олон давхаргат Perceptron сүлжээг авч үзвэл давтагдах мэдрэлийн сүлжээг архитектурт нэмэлт гогцоо гэж үзэж болно. Жишээлбэл, өгөгдсөн давхаргад нейрон бүр дохиогоо дараагийн давхарга руу дамжуулахаас гадна сүүлд (хажуу тийш) дамжуулж болно. Сүлжээний гаралт нь дараагийн оролтын векторын хамт сүлжээнд оролт болгон буцааж өгч болно. гэх мэт.

The recurrent connections add state or memory to the network and allow it to learn broader abstractions from the input sequences.

The field of recurrent neural networks is well established with popular methods. For the techniques to be effective on real problems, two major issues needed to be resolved for the network to be useful.

1. How to train the network with backpropagation
2. How to stop gradients vanishing or exploding during training

## Long Short-Term Memory Networks

The Long Short-Term Memory or LSTM network is a recurrent neural network trained using Backpropagation Through Time and overcomes the vanishing gradient problem.

Урт Богино Хугацаа Санах ой буюу LSTM сүлжээ нь цаг хугацааны Backpropagation ашиглан сургагдсан давтагдах мэдрэлийн сүлжээ бөгөөд алга болох градиент асуудлыг даван туулдаг.

As such, it can be used to create large (stacked) recurrent networks that, in turn, can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results.

Instead of neurons, LSTM networks have memory blocks connected into layers.

Блок нь сонгодог нейроноос илүү ухаалаг болгодог бүрэлдэхүүн хэсгүүдтэй бөгөөд сүүлийн үеийн дарааллыг санах ойтой байдаг. Блок нь блокийн төлөв болон гаралтыг удирдах хаалгануудыг агуулдаг. Нэгж нь оролтын дарааллаар ажилладаг бөгөөд нэгж доторх хаалга бүр нь идэвхжүүлсэн эсэхээ хянахын тулд сигмоид идэвхжүүлэх функцийг ашигладаг бөгөөд төлөвийн өөрчлөлт болон нэгжээр дамжиж буй мэдээллийн нэмэлтийг нөхцөлт болгодог.

* `Forget Gate:` conditionally decides what information to discard from the unit.
* `Input Gate:` conditionally decides which values from the input to update the memory state.
* `Output Gate:` conditionally decides what to output based on input and the memory of the unit.

