# Crash Course on Multi-Layer Perceptron Neural Networks

Artificial neural networks are a fascinating area of study, although they can be intimidating when just getting started.

There is a lot of specialized terminology used when describing the data structures and algorithms used in the field.

## 1. Multi-Layer Perceptrons

Хиймэл мэдрэлийн сүлжээний салбарыг ихэвчлэн мэдрэлийн сүлжээ эсвэл олон давхаргат мэдрэгч гэж нэрлэдэг бөгөөд магадгүй хамгийн хэрэгцээтэй мэдрэлийн сүлжээ юм. Перцептрон бол том мэдрэлийн сүлжээнүүдийн урьдал болсон нэг нейроны загвар юм.

The power of neural networks comes from their ability to learn the representation in your training data and how best to relate it to the output variable you want to predict. In this sense, neural networks learn mapping. Mathematically, they are capable of learning any mapping function and have been proven to be a universal approximation algorithm.

The predictive capability of neural networks comes from the hierarchical or multi-layered structure of the networks. The data structure can pick out (learn to represent) features at different scales or resolutions and combine them into higher-order features, for example, from lines to collections of lines to shapes.

## 2. Neurons, Weights, and Activations

The building blocks for neural networks are artificial neurons.

Эдгээр нь жигнэсэн оролтын дохиотой бөгөөд идэвхжүүлэх функцийг ашиглан гаралтын дохио үүсгэдэг энгийн тооцооллын нэгжүүд юм.

![modelofSimpleNeuron](https://machinelearningmastery.com/wp-content/uploads/2016/05/Neuron.png)

### Neuron Weights 

Оролт дээрх жин нь регрессийн тэгшитгэлд ашигласан коэффициентүүдтэй маш төстэй байдаг шугаман регрессийн талаар та сайн мэддэг байх. 

Шугаман регрессийн нэгэн адил нейрон бүр нь хэвийсэн утгатай байдаг бөгөөд үүнийг үргэлж 1.0 утгатай оролт гэж үзэж болох бөгөөд энэ нь бас жинтэй байх ёстой. 

Жишээлбэл, нейрон нь хоёр оролттой байж болох бөгөөд үүнд гурван жин шаардагддаг - оролт тус бүрт нэг, хэвийх нь нэг. Жинг ихэвчлэн 0-ээс 0.3 хүртэлх утгууд гэх мэт санамсаргүй жижиг утгууд болгон эхлүүлдэг ч илүү төвөгтэй эхлүүлэх схемийг ашиглаж болно. 

Шугаман регрессийн нэгэн адил том жин нь нарийн төвөгтэй байдал, эмзэг байдлыг илтгэнэ. Сүлжээнд жинг хадгалах нь зүйтэй бөгөөд тогтмолжуулах арга техникийг ашиглаж болно.

### Activations.

Жинсэн оролтыг нэгтгэн идэвхжүүлэх функцээр дамжуулдаг бөгөөд үүнийг заримдаа дамжуулах функц гэж нэрлэдэг.

Идэвхжүүлэх функц нь нейроны гаралтын нийлбэр жинтэй оролтыг энгийн зураглал юм. Энэ нь нейрон идэвхжсэн босго болон гаралтын дохионы хүчийг зохицуулдаг тул үүнийг идэвхжүүлэх функц гэж нэрлэдэг. 

Түүхийн хувьд, жишээ нь нийлбэр оролт нь 0.5-аас дээш босго байх үед энгийн алхам идэвхжүүлэх функцуудыг ашигладаг байсан. Дараа нь нейрон 1.0 утгыг гаргана; тэгэхгүй бол 0.0 гарна. 

Уламжлал ёсоор шугаман бус идэвхжүүлэх функцийг ашигладаг. Энэ нь сүлжээнд оролтуудыг илүү төвөгтэй аргаар нэгтгэх боломжийг олгодог бөгөөд эргээд загварчилж болох функцүүдэд илүү баялаг боломжийг олгодог. 

Сигмоид функц гэж нэрлэгддэг логистик гэх мэт шугаман бус функцуудыг s хэлбэрийн тархалттай 0-ээс 1 хүртэлх утгыг гаргахад ашигласан. Танх гэж нэрлэгддэг гипербол тангенсийн функц нь -1-ээс +1 хүртэлх мужид ижил тархалтыг гаргадаг. Саяхан Шулуутгагчийг идэвхжүүлэх функц нь илүү сайн үр дүнг өгдөг болохыг харуулсан.

## 3. Networks of Neurons

Neurons are arranged into networks of neurons.

Мөрний мэдрэлийн эсийг давхарга гэж нэрлэдэг бөгөөд нэг сүлжээ нь олон давхаргатай байж болно. Сүлжээний нейронуудын архитектурыг ихэвчлэн сүлжээний топологи гэж нэрлэдэг.

![model simple network](https://machinelearningmastery.com/wp-content/uploads/2016/05/Network.png)

### Input or Visible Layers.

Таны өгөгдлийн багцаас мэдээлэл авдаг доод давхаргыг харагдах давхарга гэж нэрлэдэг, учир нь энэ нь сүлжээний нээлттэй хэсэг юм. Мэдрэлийн сүлжээг ихэвчлэн өгөгдлийн багц дахь оролтын утга эсвэл баганад нэг нейрон бүхий харагдах давхаргаар зурдаг. Эдгээр нь дээр дурьдсан мэдрэлийн эсүүд биш бөгөөд зөвхөн оролтын утгыг дараагийн давхарга руу дамжуулдаг.

### Hidden layers

Layers after the input layer are called hidden layers because they are not directly exposed to the input. The simplest network structure is to have a single neuron in the hidden layer that directly outputs the value.

Given increases in computing power and efficient libraries, very deep neural networks can be constructed. Deep learning can refer to having many hidden layers in your neural network. They are deep because they would have been unimaginably slow to train historically but may take seconds or minutes to train using modern techniques and hardware.

### Output layers

The final hidden layer is called the output layer, and it is responsible for outputting a value or vector of values that correspond to the format required for the problem.

The choice of activation function in the output layer is strongly constrained by the type of problem that you are modeling. For example:

* Регрессийн асуудал нь нэг гаралтын нейронтой байж болох ба нейрон нь идэвхжүүлэх функцгүй байж болно. 
* Хоёртын ангиллын асуудал нь нэг гаралтын нейронтой байж болох ба 1-р ангийн утгыг таамаглах магадлалыг илэрхийлэхийн тулд 0-1-ийн хоорондох утгыг гаргахын тулд сигмоид идэвхжүүлэх функцийг ашиглаж болно. Үүнийг threshold ашиглан тодорхой ангиллын утга болгон хувиргаж болно. 0.5-аас бага байх ба 0, эс тэгвээс 1-ээс бага утгыг авна. 

* Олон ангиллын ангиллын асуудал нь гаралтын давхаргад хэд хэдэн мэдрэлийн эсүүдтэй байж болно, анги тус бүрт нэг (жишээ нь, цахилдаг цэцгийн ангиллын гурван ангийн гурван мэдрэлийн эсүүд). Энэ тохиолдолд ангийн утгууд бүрийг урьдчилан таамаглах сүлжээний магадлалыг гаргахын тулд softmax идэвхжүүлэх функцийг ашиглаж болно. Хамгийн өндөр магадлалтай гаралтыг сонгосноор ангиллын тодорхой утгыг гаргаж болно.

## 4. Training Networks

Once configured, the neural network needs to be trained on your dataset.

### Data Preparation

### Stochastic Gradient Descent

### Weight Updates

### Prediction




