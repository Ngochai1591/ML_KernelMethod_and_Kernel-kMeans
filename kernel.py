import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

def Lay_Du_Lieu():
    # Lấy dữ liệu hoa Iris từ dataset
    iris = datasets.load_iris()

    return iris

def Ve_So_Do_2D(iris):
    X = iris.data[:, :2]  #  Lấy 2 đặc tính
    Y = iris.target
    X_min, X_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    Y_min, Y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(X_min, X_max)
    plt.ylim(Y_min, Y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def Ve_So_Do_SVM(iris):
    X = iris.data[:, :2]  # Lấy 2 thuộc tính đầu tiên của dataset
    y = iris.target

    h = .01  # độ mỏng của Lưới tọa độ(Mesh) trên đồ thị

    # Tạo 1 thực thể của SVM và làm nó vừa với dữ liệu
    C = 1.0  #  tham số chính quy của SVM

    # Linear
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    # Radial Basic Function
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    # Polynomial
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    #Linear SVC
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # Tạo lưới để vẽ sơ đồ
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Tiêu đề cho các sơ đồ
    Tieu_De = ['SVM with linear kernel',
              'LinearSVC (linear kernel)',
              'SVM with RBF kernel',
              'SVM with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Vẽ sơ đồ ranh giới
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Trả kết quả vào sơ đồ
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        # Chiều dài đài hoa
        plt.xlabel('Sepal length')
        # Chiều rộng đài hoa
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(Tieu_De[i])

    plt.show()

if __name__ == "__main__":
    iris = Lay_Du_Lieu()
    if iris is not None:
        Ve_So_Do_SVM(iris)
