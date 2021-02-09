from random import randrange
import h5py
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import time as tm

import cv2 as cv

np.set_printoptions(threshold=sys.maxsize)

#%%
"""
The output data of segmentation are semantic masks, which are binary images 64 x 64. The number of recognized classes is equal to 4:three colored rectangular objects and a background class.
"""
class Rectangle:

    rangeOffset = 64
    rangeSize = 64
    rangeForColors1 = 255
    rangeForColors2 = 255
    rangeForColors3 = 255 
    nextZOrder = 0

    def __init__(self, setId, RectangleId):
        #print("Constructor called")
        self.SetId = setId
        self.RectangleId = RectangleId
        self.X = randrange(Rectangle.rangeOffset-1) #27
        self.Y = randrange(Rectangle.rangeOffset-1) #25
       
        
        self.Width = randrange(1, Rectangle.rangeSize - self.X) #13
        
        self.Height = randrange(1,Rectangle.rangeSize - self.Y ) #12

        #self.Color1= randrange(70,130,1)7
        #self.Color2= randrange(55,56,1)
        #self.Color3= randrange(60,61,1)
        self.ZOrder = Rectangle.nextZOrder #RectangleId
        Rectangle.nextZOrder = Rectangle.nextZOrder + 1
        if self.RectangleId==1:
            self.Color1 = randrange(70,149,1)
            self.Color2 = randrange(160,245,1)
            self.Color3 = randrange(30,137,1)
        elif self.RectangleId==2:
            self.Color1 = randrange(5,59,1)
            self.Color2 = randrange(84,143,1)
            self.Color3 = randrange(157,240,1)
        elif self.RectangleId==3:
            self.Color1 = randrange(170,255,1)  #(170,255,1)
            self.Color2 = randrange(12,70,1) #(12,70,1)
            self.Color3 = randrange(14,24,1) #(14,24,1)
                    
    def printValues(self):
        #s = "Rectangle {0}: ({1},{2},{3},{4}) Area={5} "
        #s.format()
        #print("Rectangle {0}: ({1},{2},{3},{4}) Area={5} Color={6}".format(
        #    self.RectangleId, self.X, self.Y, self.Width, self.Height,
        #    self.getArea() , self.Color  )    )
        
        '''print( "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10},".format( 
            self.RectangleId, 
            self.X, 
            self.Y, 
            self.Width, 
            self.Height,
            self.Color1,
            self.Color2,
            self.Color3,
            self.ZOrder,
            self.X +  self.Width,
            self.Y +  self.Height
                  )    )'''

    def getRecord(self):
      return [
            self.SetId,
            self.RectangleId, 
            self.X, 
            self.Y, 
            self.Width, 
            self.Height,
            self.Color1,
            self.Color2,
            self.Color3,
            self.ZOrder
            #,self.X +  self.Width,
            #self.Y +  self.Height
      ]


    def getArea(self):
        return self.Width * self.Height
        #print("Area={0}".format(  ) )
  

    #@staticmethod
    #def getArea():
    #    return self.Width * self.Height
    #    #print("Area={0}".format(  ) )

NUM_SETS = 100000 # это задает числи сетов <---------------

def getMask(npArray):
    
  global NUM_SETS
  fieldSize = 64      
  # empty n-dimentional set
  field64x64 = np.zeros( (NUM_SETS,fieldSize,fieldSize) ,  dtype=np.uint8 )

  NUM_ROWS = npArray.shape[0]
  #print( "npArray rows: " + str(NUM_ROWS))     

  #populate the data from npArray for each set      
  for row in range (NUM_ROWS):
  #get all needed values to draw a rectangle:
      SetId = int(npArray[row, 0]) #take the setId from npArray
      X = int(npArray[row, 2])
      Y = int(npArray[row, 3])
      width=int(npArray[row, 4])
      height=int(npArray[row, 5])

      #print ("добавление " + str(row) + " прямоугольника: {0} {1} {2} {3}".format(X,Y,width, height) )

      if X + width > 63:
        #print ( " X + width > 63 ! пропускаю ") 
        continue
      if Y + height > 63:
        #print ( " Y + height > 63 ! пропускаю ") 
        continue


      #addRectangle (X, Y, width, height , row , npArray)
      for i in range ( int(height) ): 
          for j in range ( int(width) ) :
              field64x64[SetId, X + j, Y + i] = 1 + row% 3
              
              
              

      #print ("После добавления " + str(row) + " прямоугольника:")
      #print( field64x64 )
  
  return field64x64
  #print (field64x64.shape)  

def getField(npArray):
    
  global NUM_SETS
  fieldSize = 64 
    
  field64x64 = np.zeros((NUM_SETS, fieldSize,fieldSize, 3),  dtype=np.uint8)
  
  NUM_ROWS = npArray.shape[0]
  #print( "npArray rows: " + str(NUM_ROWS))
  for row in range (NUM_ROWS):
  #get all needed values to draw a rectangle:
      SetId = int(npArray[row, 0]) #take the setId from npArray
      X = int(npArray[row, 2])
      Y = int(npArray[row, 3])
      width=int(npArray[row, 4])
      height=int(npArray[row, 5])
      Color1=int(npArray[row, 6])
      Color2=int(npArray[row,7])
      Color3=int(npArray[row,8])

      #print ("добавление " + str(row) + " прямоугольника: {0} {1} {2} {3}".format(X,Y,width, height) )

      if X + width > 63:
        #print ( " X + width > 63 ! пропускаю ") 
        continue
      if Y + height > 63:
        #print ( " Y + height > 63 ! пропускаю ") 
        continue


      #addRectangle (X, Y, width, height , row , npArray)
      for i in range ( int(height) ): 
          for j in range ( int(width) ) :
              field64x64[SetId, X + j, Y + i,0] = Color1
              field64x64[SetId, X + j, Y + i,1] = Color2
              field64x64[SetId, X + j, Y + i,2] = Color3


      #print ("После добавления " + str(row) + " прямоугольника:")
      #print( field64x64 )
  
  return field64x64




def main():
    global NUM_SETS
    #NUM_SETS = 3
    NUM_COLUMNS  = 10 #+ 1
    npArray = np.empty(( 0, NUM_COLUMNS ));
    col_names = ["SetId", "Id","X","Y","Width","Height","Color1","Color2","Color3","ZOrder"]
    #npArray = np.append( npArray , [col_names], axis=0)
    ##print (npArray.shape)
    ##print( "SetId, Id,X,Y,Widht,Height,Color1,Color2,Color3,ZOrder")
    for setId in range(NUM_SETS):
        print(setId, ' / ', NUM_SETS)
        for i in range(1,4):
            
            r = Rectangle(setId, i)
            
                
          #r.printValues()
            npArray = np.append( npArray,[r.getRecord()],  axis=0)

    #print("Array:\n")
    ##print (npArray)
    
    
    resultField = getField(npArray)
    resultMask = getMask(npArray)
    #print(resultField.shape)
    #print(resultMask.shape)
    #print( resultField[:,0,0]  )
    
    '''plt.figure(figsize = (12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(resultField[0])
    plt.subplot(1, 2, 2)
    plt.imshow(resultMask[0])
    plt.show()'''
        
    
    with h5py.File('FullDataSet.hdf5', 'w') as f:
        f.create_dataset("im", data=resultField)
        #dset.attrs['column_names'] =  col_names
        f.create_dataset("mask", data=getMask(npArray))
        #dset.attrs['column_names'] =  col_names
        
        #d1 = f['column_names']
        #d2 = f['array_2']
    
    #resultField = getField(npArray)
    #print( resultField  )
    #np.savetxt('test.txt', resultField, delimiter=',', fmt='%u')
    
    #ff = h5py.File('one_rectangle_freshman2.hdf5', 'r')
    #d1 = ff['im'][...]
    #d2 = ff['mask'][...]

    #permut = np.random.permutation(len(d1))[0]
    #print(permut)
    
    #ff.close()
    
    '''print(d1.shape, d2.shape)
    
    plt.figure(figsize = (12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(d1[permut])
    plt.subplot(1, 2, 2)
    plt.imshow(d2[permut])
    plt.show()
    #print(type(d1), type(d2))
    #print(d1.dtype, d2.dtype)'''
    
begin_time = tm.time()
if __name__ == "__main__":
    main()
end_time = tm.time()
print(end_time - begin_time)
#%%

#for i in range(25):
#    print(1 + i%3)