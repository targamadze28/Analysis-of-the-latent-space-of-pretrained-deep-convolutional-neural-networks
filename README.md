# Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks
##### Решаемая нами задача - трехклассовая семантическая сегментация прямоугольных разноцветных объектов на изображении.
Входными данными являются трехканальные синтетические изображения пересекающихся прямоугольников различного цветового диапазона, а результирующие значения - семантические маски, показывающие отношение пикселей к определенному классу. Центральным подходом в данной работе является глубокое обучение, которое состоит в обучении глубокой свёрточной нейронной сети решать поставленную задачу. 
Задача семантической сегментации является хорошо изученной и с высокой точностью может быть решена нейронный сетями, поэтому сама задача сегментации прямоугольных разноцветных объектов не представляет научного интереса, однако в данной работе она выступает в качестве вспомогательной задачи.
Основной задачей является исследования структуры и свойств латентного (скрытного) пространства обученной, на задаче сегментации, нейронной сети, с целью практического изучения гипотезы компактности искусственной нейронной сети.  

## Входные Данные
Входными данными являются трехканальные RGB изображения размером 64 х 64 х 3. На самих картинках изображены прямоугольники с параметром цвета &delta;, который характеризует длину цветового диапазона, как указано в таблице 1. В последующих разделах, если не указано обратное, значение параметра &delta; по умолчанию равно 15 (&delta; = 15). Цвет прямоугольника для каждого класса выбирает случайно по равномерному закону распределения из представленных диапазонов. Таким образом на каждом изображении представлены три прямоугольника представителя класса. 
##### Таблица 1: Цветовые диапазоны классов, -- задавыемый парамтер ()   
|               |       I       |          II        |  III        | 
|--------------:| :-----------: |:------------------:| :----------:|
|       R       | [42- &delta;, 42 + &delta;]  | [127- &delta;, 127+ &delta;]       | [212- &delta;, 212+ &delta;]|
|       G       | [127- &delta;, 127 + &delta;]  | [212- &delta;, 212+ &delta;]       | [42- &delta;, 42+ &delta;]|
|       B       | [212- &delta;, 212 + &delta;]  | [42- &delta;, 42+ &delta;]       | [127- &delta;, 127+ &delta;]|

![alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/ImageExampleWithoutSelection.png?raw=true) ![alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/ImageExampleWithSelection.png?raw=true)

## Результирующие Маски 
В данной задаче результирующими величинами являются семантические маски, которые для определенного класса представляют собой бинарные изображение 64 х 64 х 1 и содержат ненулевые пиксели там, где исходное изображение содержит пиксели соответствующего класса.  Строго говоря, в задаче фигурируют четыре класса, а именно: три класса прямоугольников и фоновый класс, который не содержит пикселей первых трех.  
![alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/FullMask1.png?raw=true) ![alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/FullMask2.png?raw=true) ![alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/FullMask3.png?raw=true) [alt text](https://github.com/targamadze28/Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks/blob/main/FullMask4.png?raw=true)
### Разведывательный Анализ Данных (EDA)
Для проведения разведывательного анализа данных, было создано две статистические выборки данных, размер каждый составляет 50 000 экземпляров, причем первая выборка была собрана без учета условия полного вхождения прямоугольника, а вторая при выполнении этого условия.

## Используемые Нейронные Сети 
Были сформированы три нейронные сети различной архитектуры, различающиеся по объему параметров, количеству скрытых слоев, а следовательно и по выразительности, а так же по своей топологии.

Используемые архитектуры:

**NN <sub>1</sub>** - сверточный автокодировщик, состоящий из последовательного стека сверточных и пулинговых слоев;  
**NN <sub>2</sub>** - U-net подобная архитектура, отличающаяся меньшей шириной, глубиной и естественно емкостью;  
**NN <sub>3</sub>** - U-net подобная архитектура, идентичная NN <sub>2</sub> в этой сети, обычные свертки (Conv2D) заменены на разделенные свертки (Separable Convolutions);

## Обучение 
Для обучения искусственных нейронных сетей был сгенерировать обучающий набор данных, общий размер которого составляет 200 000 экземпляров. Для тестирования сетей создан отдельный набор данных, объем которого равен 15 000 экземпляров.  
Для обучения были выбраны следующие функции потерь:  

1.   weighted categorical crossentropy - взвешанная категориальная перекрестная энтропия, веса которой выбранные таким образом, чтобы сбалансировать классы по площади;
2. categorical focal loss - взвешенный фокальная функция потерь;  
3. dice loss - функция потерь, основанная на коэффициенте дайса;  

Для проведения сравнительного анализа, представленные нейронные сети будут обучаться на всех, выше перечисленных функциях потерь.   
Выбранная стратегия обучения сетей одинакова для всех и формируется следующим образом:
1. Обучение в течении 25 эпох с размером мини-пакета 2048 на оптимизаторе Nadam, настройки оптимизатора по умолчанию;  
2. Дообучение модели, длительностью 50 эпох с размером мини-пакета 2048 на оптимизаторе SGD, начальная скорость обучения 0.02, которая снижается по экспоненциальному закону с параметром затухания 0.02/50.  

## Построение Латентных Пространств 
В этом разделе выполним визуализацию пространства. В силу большой размерности пространства, полная визуализация невозможно, поэтому воспользуемся алгоритмом нелинейного снижения размерности UMAP, который в сравнении с другими методами нелинейного проектирования лучше сохраняет глобальную структуру многообразий данных. А так же, для визуализации многомерных пространств воспользуемся алгоритмом линейного сжатия PCA, который выделяет главные направления (размерности) исходного пространства.

## Использование Перцептрона

В этом разделе, опишем применение перцептрона для исследования структуры многомерного пространства. Для начала, необходимо напомнить, что перцептрон это линейный классификатор, то есть разделение данных происходит линейной гиперплоскостью, соответственно, если в многомерном пространстве существуют обособленные кластеры, то их можно линейно разделить.  
Исходя из представленных выше рассуждений, обучим перцептрон на классификацию многомерного латентного пространства, со стратегией обучения один против всех. Для каждого класса будет свой перцептрон. Для проверки точности классификации будем использовать метрику F1, набор данных на обучающий и проверочный разделять нет необходимости, потому что необходимо оценить аппроксимирующую способность перцептрона, то есть обучение и оценка точности будут только на обучающей (всей) выборке размеченного латентного пространства.





