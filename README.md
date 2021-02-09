# Analysis-of-the-latent-space-of-pretrained-deep-convolutional-neural-networks
Решаемая нами задача -- трехклассовая семантическая сегментация прямоугольных разноцветных объектов на изображении. 


Входными данными являются трехканальные синтетические изображения пересекающихся прямоугольников различного цветового диапазона, а результирующие значения --- семантические маски, показывающие отношение пикселей к определенному классу. Центральным подходом в данной работе является глубокое обучение, которое состоит в обучении глубокой свёрточной нейронной сети решать поставленную задачу. 
Задача семантической сегментации является хорошо изученной и с высокой точностью может быть решена нейронный сетями, поэтому сама задача сегментации прямоугольных разноцветных объектов не представляет научного интереса, однако в данной работе она выступает в качестве вспомогательной задачи. Основной задачей является исследования структуры и свойств латентного (скрытного) пространства обученной, на задаче сегментации, нейронной сети, с целью практического изучения гипотезы компактности искусственной нейронной сети.

