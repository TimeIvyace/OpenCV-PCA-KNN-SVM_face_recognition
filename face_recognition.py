import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cv2
from PIL import Image

IMAGE_SIZE = 224

def LoadData(): #载入训练数据集
    data = []
    label = []
    num = 100
    path_cwd = "/train/"
    for j in range(1, 4):
        path = path_cwd + 's' + str(j)
        for number in range(num):
            path_full = path + '/' + str(number) +'.jpg'
            image = Image.open(path_full).convert('L')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            img = np.reshape(image, (1, IMAGE_SIZE*IMAGE_SIZE))
            data.extend(img)
        label.extend(np.ones(num, dtype=np.int) * j)
    data = np.reshape(data, (num*j, IMAGE_SIZE*IMAGE_SIZE))
    return np.matrix(data), np.matrix(label).T       #返回数据和标签


def svm(trainDataSimplified, trainLabel, testDataSimplified):
    clf3 = SVC(C=3.0)  # C为分类数目
    clf3.fit(trainDataSimplified, trainLabel)
    return clf3.predict(testDataSimplified)

def knn(neighbor, traindata, trainlabel, testdata):
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(traindata, trainlabel)
    return neigh.predict(testdata)

if __name__ == '__main__':

    Data, Label = LoadData()
    pca = PCA(0.9, True, True)  # 建立pca类，设置参数，保留90%的数据方差
    trainDataS = pca.fit_transform(Data)  # 拟合并降维训练数据

    #框住人脸的矩形边框颜色
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    #人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt.xml"

    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频

        #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                #截取脸部图像提交给模型识别这是谁
                m = frame_gray[y - 10: y + h + 10, x - 10: x + w + 10]

                top, bottom, left, right = (0, 0, 0, 0)
                image = m
                # 获取图像尺寸
                h, w = image.shape

                # 对于长宽不相等的图片，找到最长的一边
                longest_edge = max(h, w)

                # 计算短边需要增加多上像素宽度使其与长边等长
                if h < longest_edge:
                    dh = longest_edge - h
                    top = dh // 2
                    bottom = dh - top
                elif w < longest_edge:
                    dw = longest_edge - w
                    left = dw // 2
                    right = dw - left
                else:
                    pass

                BLACK = [0]

                # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
                constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

                # 调整图像大小并返回
                image = cv2.resize(constant, (IMAGE_SIZE, IMAGE_SIZE))
                img_test = np.reshape(image, (1, IMAGE_SIZE * IMAGE_SIZE))
                testDataS = pca.transform(img_test)  # 降维测试数据
                result = svm(trainDataS, Label, testDataS)  # 使用SVM进行分类
                # result = knn(5, trainDataS, Label, testDataS)  # 使用KNN进行分类，5为最近邻居数
                faceID = result[0]


                #如果是“我”
                if faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,'A',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,'B',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)
                elif faceID == 3:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,'C',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)

        cv2.imshow("find me", frame)

        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
