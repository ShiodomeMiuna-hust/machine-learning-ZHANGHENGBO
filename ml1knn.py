import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy
import heapq

dataBase="data_base"
persons=[]
faces=[]
for i in range(1, 41):
    persons.append("s" + str(i))
for i in range(1, 11):
    faces.append(str(i) + ".pgm")

def parseImageToVector(path):
    """
    功能：将图像转换为特征矢量
    输入：图像的路径
    返回值：特征向量（numpy一维数组)
    """
    img = numpy.array(Image.open(path))
    return img.flatten()

def eularDistance(vec1, vec2):
    """
    功能：计算两个特征向量的eular距离
    输入：特征向量1，2
    返回值：欧拉距离
    """
    return numpy.sum(numpy.square(vec1-vec2))

def trainSetInitialization(faces):
    """
    功能：初始化训练数据集
    输入：用作训练集的人脸ID列表
    返回值：初始化的数据集，（数据, 标签, 图像）元组列表
    """
    trainSet = []
    for person in persons:
        for face in faces:
            imgPath = dataBase + "/" + person + "/" + str(face) + ".pgm"
            imgVec = parseImageToVector(imgPath)
            trainSet.append((imgVec, person, str(face)))
    return trainSet

def faceRecognition(face, trainSet, k):
    """
    功能：识别人脸，计算训练数据集中的哪张脸与此脸相同（KNN实现）
    输入：face数组中的测试脸，trainDataSet训练后的数据，kNN参数
    返回：数据集中最相同的面孔的标签
    """
    heap = [] #小根堆，储存（距离，标号，人脸）元组
    neighbors = [] # 保存前k个点的信息
    result = {} # { key : val } 表示一组k近邻点中 { 标签 : 标签数量(1<=n<=k) }
    # 计算前k个最近的点，压入小根堆heap
    for trainData in trainSet:
        # trainData[1]对应person标签, trainData[0]对应该标签下的某个特征向量
        heapq.heappush(heap, (eularDistance(face, trainData[0]), trainData[1], trainData[2]) )

    # 找到前k个最近的点中数量最多的标签，并加入结果result
    for i in range(k):
        first = heapq.heappop(heap)
        top = first[1] # 标签
        topImg = first[2] # 图像
        neighbors.append((top, topImg))
        if top in result:
            result[top] = result[top] + 1
        else:
            result[top] = 1
    maximum = (None, 0)
    for label in result:
        if result[label] > maximum[1]:
            maximum = (label, result[label])
    # 显示信息
    print("测试图片路径:" + maximum[0])
    print("标签维度" + str(result))
    print("相似人脸图片反馈:")
    for neighbor in neighbors:
        path = dataBase + "/" + neighbor[0] + "/" + neighbor[1] + ".pgm"
        print(path)
    print("-------------分界线--------------")
    return maximum[0]

def main():
    fault = 0
    total = 0
    kList = []
    misclassificationRateList = []
    for k in range(1, 21):
        for testIndex in range(1, 11):
            # 初始化训练集
            trainImages = []
            for trainImage in range(1,11):
                trainImages.append(trainImage)
            trainImages.remove(testIndex)
            trainSet = trainSetInitialization(trainImages)

            # 测试
            for person in persons:
                path = dataBase + "/" + person + "/" + str(testIndex) + ".pgm"
                faceVec = parseImageToVector(path)
                print("测试人脸的路径:" + path)
                result = faceRecognition(faceVec, trainSet, k)
                if person != result:
                    fault = fault + 1
                total = total + 1
        kList.append(k)
        misclassificationRateList.append(fault / total)
        print("misclassification rate:", fault / total)
    # 显示图像
    plt.plot(kList, misclassificationRateList, alpha=0.7)
    plt.xticks(kList, kList)
    plt.ylabel("Misclassification Rate")
    plt.show()

if __name__ == "__main__":
    main()