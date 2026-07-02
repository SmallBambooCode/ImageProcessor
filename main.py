import os
import random
import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsScene, QFileDialog, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from matplotlib import pyplot as plt
from mainWindow import Ui_MainWindow
from propertyWindow import Ui_Form

# 适应高DPI设备
# QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

# 自定义启动画面类
class TransparentSplash(QSplashScreen):
    def __init__(self, pixmap):
        # 设置窗口在最顶层（置顶），删除窗口边框和标题，让窗口没有边界
        super().__init__(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMask(pixmap.mask())
    # 禁用窗口点击时间
    def mousePressEvent(self, event):
        # 禁用点击关闭
        event.ignore()

# 工具函数，显示启动图
def show_startup_image(image_path, duration_ms, main_window, max_size=600):
    pixmap = QPixmap(image_path)

    # 限制最大尺寸，保持比例缩放
    if pixmap.width() > max_size or pixmap.height() > max_size:
        pixmap = pixmap.scaled(
            max_size, max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

    splash = TransparentSplash(pixmap)
    screen = QApplication.primaryScreen().availableGeometry()
    splash_size = splash.size()
    # 屏幕中央
    splash.move(
        (screen.width() - splash_size.width()) // 2,
        (screen.height() - splash_size.height()) // 2
    )
    splash.show()

    QTimer.singleShot(duration_ms, lambda: (
        splash.close(),
        main_window.show()
    ))

# 预处理窗口类
# 就是弹出来的调整各种属性值的小窗口
class PropertyWindow(QWidget, Ui_Form):
    # 信号只能在Object的子类中创建，并且只能在创建类的时候的时候添加，而不能等类已经定义完再作为动态属性添加进去。
    # 自定义的信号在__init__()函数之前定义。
    # 自定义一个信号signal，有一个object类型的参数
    signal = QtCore.pyqtSignal(object)

    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(PropertyWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)

        # 绑定窗口组件响应事件的处理函数（将窗口中的组件被用户触发的点击、值变化等事件绑定到处理函数）
        # 数值框的值改变
        self.spinBox.valueChanged.connect(self.__spinBoxChange)
        # 滑动条的值改变
        self.slider.valueChanged.connect(self.__sliderChange)
        # 点击确认按钮
        self.submitButton.clicked.connect(self.__valueConfirm)

    # 数值框值改变的处理函数
    def __spinBoxChange(self):
        # 获取数值框的当前值
        value = self.spinBox.value()
        # 与滑动条进行数值同步
        self.slider.setValue(value)
        # 发送信号到主窗口，参数是当前数值（主窗口有自定义的接收并处理该信号的函数）
        self.signal.emit(value)

    # 滑动条值改变的处理函数
    def __sliderChange(self):
        # 获取滑动条的当前值
        value = self.slider.value()
        # 与数值框进行数值同步
        # 注意：该操作也会触发数值框的数值改变，即会触发调用__spinBoxChange()，所以不要需要在此处重复发信号到主窗口
        self.spinBox.setValue(value)

    # 确认按钮按下的处理函数
    def __valueConfirm(self):
        # 发送确认修改信号
        self.signal.emit("ok")
        # 关闭窗口
        self.close()

    # 重写窗口关闭处理
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        # 发送取消修改信号，关闭窗口时触发
        self.signal.emit("close")


# 主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(MainWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)
        # 设置图标
        self.setWindowIcon(QIcon("favicon.ico"))
        # 图像属性调整的子窗口，初始化默认空（亮度、对比度、锐度、饱和度、色调、旋转、缩放调节窗口）
        self.__propertyWindow = None

        # 当前打开的图片文件名，初始化默认空
        self.__fileName = None

        # 保存图片原始数据，初始化默认空
        self.__srcImageRGB = None
        # 保存图片最终处理结果的数据，初始化默认空
        self.__outImageRGB = None
        # 保存图片暂时修改的数据，初始化默认空
        # （在修改图像属性未点击确认时，需要暂存修改数据，如果确认后将临时数据同步为最终结果数据，如果未确认将复原数据）
        self.__tempImageRGB = None
        # 历史步骤（图像）
        self.__historyStack = []
        # 最大保存的历史图像数量
        self.__maxHistorySize = 20

        # 绑定窗口事件的响应函数
        # 文件菜单
        # 打开文件
        self.openFileAction.triggered.connect(self.__openFileAndShowImage)
        # 保存文件
        self.saveFileAction.triggered.connect(self.saveFile)
        # 另存为文件
        self.saveFileAsAction.triggered.connect(self.saveFileAs)
        # 关闭文件
        self.closeFileAction.triggered.connect(self.closeFile)
        # 退出程序
        self.exitAppAction.triggered.connect(self.close)

        # 撤销菜单
        # 重置图像
        self.resetImageAction.triggered.connect(self.__resetImage)
        # 恢复到上一步
        self.undoAction.triggered.connect(self.__undoImage)

        # 直接灰度映射菜单
        # 灰度化
        self.grayAction.triggered.connect(self.__toGrayImage)
        # 二值化
        self.binaryAction.triggered.connect(self.__toBinaryImage)
        # 颜色反转
        self.reverseAction.triggered.connect(self.__reverseImage)
        # 亮度调整
        self.lightAction.triggered.connect(self.__openLightWindow)
        # 对比度调整
        self.contrastAction.triggered.connect(self.__openContrastWindow)
        # 锐度调整
        self.sharpAction.triggered.connect(self.__openSharpWindow)
        # 饱和度调整
        self.saturationAction.triggered.connect(self.__openSaturationWindow)
        # 色度调整
        self.hueAction.triggered.connect(self.__openHueWindow)

        # 图像运算菜单
        # 加
        self.imageAddAction.triggered.connect(self.__addImage)
        # 减
        self.imageSubtractAction.triggered.connect(self.__subtractImage)
        # 乘
        self.imageMultiplyAction.triggered.connect(self.__multiplyImage)
        # 缩放
        self.zoomAction.triggered.connect(self.__openZoomWindow)
        # 旋转
        self.rotateAction.triggered.connect(self.__openRotateWindow)

        # 直方图均衡菜单
        # 归一化直方图
        self.histogramAction.triggered.connect(self.__histogram)
        # 直方图均衡化
        self.histogramEqAction.triggered.connect(self.__histogramEqualization)

        # 噪声菜单
        # 加高斯噪声
        self.addGaussianNoiseAction.triggered.connect(self.__addGasussNoise)
        # 加均匀噪声
        self.addUiformNoiseAction.triggered.connect(self.__addUniformNoise)
        # 加脉冲（椒盐）噪声
        self.addImpulseNoiseAction.triggered.connect(self.__addImpulseNoise)

        # 空域滤波菜单
        # 均值滤波
        self.meanValueAction.triggered.connect(self.__meanValueFilter)
        # 中值滤波
        self.medianValueAction.triggered.connect(self.__medianValueFilter)
        # Sobel算子锐化
        self.sobelAction.triggered.connect(self.__sobel)
        # Prewitt算子锐化
        self.prewittAction.triggered.connect(self.__prewitt)
        # 拉普拉斯算子锐化
        self.laplacianAction.triggered.connect(self.__laplacian)

        # 关于菜单
        # 关于作者
        self.aboutAction.triggered.connect(self.__aboutAuthor)

    # 保存当前步骤
    def _saveCurrentStateToHistory(self):
        if self.__outImageRGB is not None:
            # 入栈
            self.__historyStack.append(self.__outImageRGB.copy())
            # 如果栈满则删除最早的图片
            if len(self.__historyStack) > self.__maxHistorySize:
                self.__historyStack.pop(0)

    # -----------------------------------文件-----------------------------------
    # 打开文件并在主窗口中显示打开的图像
    def __openFileAndShowImage(self):
        # 打开文件选择窗口
        __fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", ".", "Image Files(*.png *.jpeg *.jpg *.bmp)")
        # 文件存在
        if __fileName and os.path.exists(__fileName):
            # 设置打开的文件名属性
            self.__fileName = __fileName
            # 转换颜色空间，cv2默认打开BGR空间，Qt界面显示需要RGB空间，所以就统一到RGB吧
            # __bgrImg = cv2.imread(self.__fileName)
            # cv2 读取不了有中文名的图像文件 ！！！
            # 所以我用的numpy读取数据，再用cv2.imdecode解码数据来解决。
            __bgrImg = cv2.imdecode(np.fromfile(self.__fileName, dtype=np.uint8), -1)
            # 设置初始化数据
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__tempImageRGB = self.__srcImageRGB.copy()
            # 清空历史
            self.__historyStack = []
            self._saveCurrentStateToHistory()
            # 在窗口中左侧QGraphicsView区域显示图片
            self.__drawImage(self.srcImageView, self.__srcImageRGB)
            # 在窗口中右侧QGraphicsView区域显示图片
            self.__drawImage(self.outImageView, self.__srcImageRGB)

    # 在窗口中指定的QGraphicsView区域（左或右）显示指定类型（rgb、灰度、二值）的图像
    def __drawImage(self, location, img):
        # RBG图
        if len(img.shape) > 2:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width, __channel = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width * __channel, QImage.Format_RGB888)
        # 灰度图、二值图
        else:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)

        # 创建QPixmap对象
        __qPixmap = QPixmap.fromImage(__qImg)
        # 获取显示区域大小
        view_size = location.size()
        view_width = view_size.width()
        view_height = view_size.height()
        # 缩放QPixmap以适应QGraphicsView，同时保持纵横比
        if __qPixmap.width() > view_width or __qPixmap.height() > view_height:
            __qPixmap = __qPixmap.scaled(view_width, view_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 创建显示容器QGraphicsScene对象
        __scene = QGraphicsScene()
        # 填充QGraphicsScene对象
        __scene.addPixmap(__qPixmap)
        # 将QGraphicsScene对象设置到QGraphicsView区域实现图片显示
        location.setScene(__scene)
        # 使视图适配新场景的边界矩形
        location.fitInView(__scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # 更新图片基本信息
        if location == self.outImageView and self.__fileName:
            info_text = f"当前图像信息：尺寸（{img.shape[1]}x{img.shape[0]}）"
            self.infoLabel.setText(info_text)

    # 执行保存图片文件的操作
    def __saveImg(self, fileName):
        # 检查文件名是否有效（不为空）
        if fileName:
            try:
                __bgrImg = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
                # 从文件名中提取扩展名格式
                # 然后调用cv2.imencode将图像数据按此格式编码成二进制流，存入内存变量buffer
                retval, buffer = cv2.imencode(os.path.splitext(fileName)[1], __bgrImg)
                # 果编码成功
                if retval:
                    # 使用f.write写入文件
                    with open(fileName, "wb") as f:
                        f.write(buffer)
                    # 提示成功
                    QMessageBox.information(self, "提示", f"文件已成功保存到:\n{fileName}")
                else:
                    QMessageBox.warning(self, "错误", "图像编码失败，无法保存！")
            except Exception as e:
                # 捕获可能发生的其他异常（如权限问题）
                QMessageBox.critical(self, "保存失败", f"保存文件时发生错误：\n{e}")
        # 如果 fileName 为空（例如用户取消了对话框），则不执行任何操作
        else:
            QMessageBox.information(self, "保存失败", "文件保存失败，请先打开图片或选择正确的路径！")

    # 保存文件，覆盖原始文件
    def saveFile(self):
        self.__saveImg(self.__fileName)

    # 文件另存
    def saveFileAs(self):
        # 已经打开了文件才能保存
        if self.__fileName:
            # 打开文件保存的选择窗口
            __fileName, _ = QFileDialog.getSaveFileName(self, "保存图片", "Image",
                                                        "Image Files(*.png *.jpeg *.jpg *.bmp)")
            self.__saveImg(__fileName)
        else:
            # 消息提示窗口
            QMessageBox.information(self, "提示", "文件保存失败！")
    def closeFile(self):
        # 1. 清除左侧原始图片显示区域
        self.srcImageView.setScene(QGraphicsScene())
        # 2. 清除右侧处理结果图片显示区域
        self.outImageView.setScene(QGraphicsScene())

        # 3. 重置所有图片数据相关的内部变量为 None
        self.__fileName = None
        self.__srcImageRGB = None
        self.__outImageRGB = None
        self.__tempImageRGB = None

        # 4. 给出提示信息
        QMessageBox.information(self, "提示", "文件已关闭，界面已恢复初始状态。")

    # 重写窗口关闭事件函数，来关闭所有窗口。因为默认关闭主窗口子窗口依然存在。
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        sys.exit(0)

    # 重写大小调整事件，以便在窗口大小改变时自动重绘图像
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # 如果当前已加载图像，则在两个视图中重新绘制它
        if self.__fileName:
            if self.__srcImageRGB is not None:
                self.__drawImage(self.srcImageView, self.__srcImageRGB)
            if self.__outImageRGB is not None:
                self.__drawImage(self.outImageView, self.__outImageRGB)
    # -----------------------------------重置图片-----------------------------------
    # 重置图片到初始状态
    def __resetImage(self):
        if self.__fileName:
            # 还原文件打开时的初始化图片数据
            self.__outImageRGB = self.__srcImageRGB.copy()
            # 清空历史记录
            self.__historyStack = []
            self._saveCurrentStateToHistory()
            # 窗口显示图片
            self.__drawImage(self.outImageView, self.__outImageRGB)
    # 回到上一步
    def __undoImage(self):
        if self.__fileName is None:
            QMessageBox.information(self, '提示', '没有打开的图片。')
            return

        # 检查历史记录中是否至少有两个状态（当前状态 + 先前状态）
        # 栈中的最后一个元素是当前状态。我们需要回到它之前的那个状态。
        if len(self.__historyStack) > 1:
            # 出栈
            self.__historyStack.pop()
            # 替换当前图片为上一步的图片
            self.__outImageRGB = self.__historyStack[-1].copy()
            # 更新展示状态
            self.__drawImage(self.outImageView, self.__outImageRGB)
        else:
            QMessageBox.information(self, '提示', '无法再回到上一步了。')

    # -----------------------------------图像预处理-----------------------------------
    # 灰度化
    def __toGrayImage(self):
        # 只有RGB图才能灰度化
        if self.__fileName and len(self.__outImageRGB.shape) > 2:
            # 灰度化使得三通道RGB图变成单通道灰度图
            self.__outImageRGB = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 二值化
    def __toBinaryImage(self):
        # 先灰度化
        self.__toGrayImage()
        if self.__fileName:
            # 后阈值化为二值图
            _, self.__outImageRGB = cv2.threshold(self.__outImageRGB, 127, 255, cv2.THRESH_BINARY)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 反转图片颜色
    def __reverseImage(self):
        if self.__fileName:
            self.__outImageRGB = cv2.bitwise_not(self.__outImageRGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 执行打开属性调节子窗口（亮度、对比度、锐度、饱和度、色调、缩放、旋转）
    def __openPropertyWindow(self, propertyName, func):
        if self.__fileName:
            if self.__propertyWindow:
                self.__propertyWindow.close()
            self.__propertyWindow = PropertyWindow()
            # 设置窗口内容
            self.__propertyWindow.setWindowTitle(propertyName)
            self.__propertyWindow.propertyLabel.setText(propertyName)
            # 接收信号
            # 设置主窗口接收子窗口发送的信号的处理函数
            self.__propertyWindow.signal.connect(func)
            # 禁用主窗口菜单栏，子窗口置顶，且无法切换到主窗口
            self.__propertyWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            self.__propertyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
            # 显示子窗口
            self.__propertyWindow.show()

    # 亮度调节子窗口
    def __openLightWindow(self):
        self.__openPropertyWindow("亮度", self.__changeLight)
        self._saveCurrentStateToHistory()

    # 对比度调节子窗口
    def __openContrastWindow(self):
        self.__openPropertyWindow("对比度", self.__changeContrast)
        self._saveCurrentStateToHistory()

    # 锐度调节子窗口
    def __openSharpWindow(self):
        self.__openPropertyWindow("锐度", self.__changeSharp)
        self._saveCurrentStateToHistory()

    # 饱和度调节子窗口
    def __openSaturationWindow(self):
        self.__openPropertyWindow("饱和度", self.__changeSaturation)
        self._saveCurrentStateToHistory()

    # 色调调节子窗口
    def __openHueWindow(self):
        self.__openPropertyWindow("色调", self.__changeHue)
        self._saveCurrentStateToHistory()

    # 预处理信号
    def __dealSignal(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        # 如果是灰度图要转为RGB图
        if len(__img.shape) < 3:
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)

        value = str(val)
        # 确认修改
        if value == "ok":
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
            self._saveCurrentStateToHistory()
            return None
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == "close":
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
            return None
        # 暂时修改
        else:
            return __img

    # 修改亮度
    def __changeLight(self, val):
        # 先处理信号，得到基础图像 __img（来自 __dealSignal）
        __img = self.__dealSignal(val)
        # 如果是滑块在调整数值（而非 "ok" 或 "close"）
        if np.size(__img) > 1:
            # 计算 β：滑块 val 的范围是 [-100,100]，此处把它映射到 [-255, +255]
            beta = int(val) * (255 / 100)

            # 如果当前图是灰度图，就先转成三通道 RGB，以便后续统一做 addWeighted
            if len(__img.shape) < 3:
                img_rgb = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = __img

            rows, cols, channels = img_rgb.shape
            blank = np.zeros([rows, cols, channels], img_rgb.dtype)
            modified = cv2.addWeighted(img_rgb, 1.0, blank, 0.0, beta)

            # 把修改后的结果先存到临时变量，再实时预览
            self.__tempImageRGB = modified
            self.__drawImage(self.outImageView, modified)

    # 修改对比度
    def __changeContrast(self, val):
        # 先处理信号，得到基础图像 __img（可能是预览或确认）
        __img = self.__dealSignal(val)
        # 如果是真正调整值（非“ok”或“close”）
        if np.size(__img) > 1:
            k = int(val)  # 滑块值范围大约 [-100,100]
            if k != -100:
                alpha = (k + 100) / 100  # k=-100 → 0.01; k=0 → 1.0; k=100 → 2.0
            else:
                alpha = 0.01
            # 将图像转 float32 以便叠加
            img_float = __img.astype(np.float32)
            # 按公式：g = alpha * (f - 128) + 128
            tmp = alpha * (img_float - 128.0) + 128.0
            # 截断并转回 uint8
            tmp = np.clip(tmp, 0, 255).astype(np.uint8)
            # 暂存并实时画出来
            self.__tempImageRGB = tmp
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改锐度
    def __changeSharp(self, val):
        __img = self.__dealSignal(val)
        if np.size(__img) > 1:
            # 把滑块值映射到 [-1, +1] 作为锐度系数 beta
            beta = int(val) / 100.0
            # 转为 float 做运算
            img_float = __img.astype(np.float32)
            # 先做一次高斯模糊（sigma=1）——当然你可以根据需要调整 sigma
            blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=1.0, sigmaY=1.0)
            # 融合：output = (1+beta)*原图 - beta * blurred
            tmp = (1 + beta) * img_float - beta * blurred
            # 截断并转 uint8
            tmp = np.clip(tmp, 0, 255).astype(np.uint8)
            self.__tempImageRGB = tmp
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改饱和度
    def __changeSaturation(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if np.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (255 / 100)
            # 切片修改S分量，并限制色彩数值在0-255之间
            __img[:, :, 2] = np.clip(__img[:, :, 2] + k, 0, 255)
            # 暂存修改数据
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改色调
    # OpenCV中hue通道的取值范围是0 - 180
    def __changeHue(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if np.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (90 / 100)
            # 切片修改H分量，并限制色彩数值在0-180之间
            __img[:, :, 0] = (__img[:, :, 0] + k) % 180
            # 暂存修改数据
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------图像运算-----------------------------------
    # 加、减、乘操作
    def __operation(self, func):
        if self.__fileName:
            __fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", ".", "Image Files(*.png *.jpeg *.jpg *.bmp)")
            if __fileName and os.path.exists(__fileName):
                __bgrImg = cv2.imdecode(np.fromfile(__fileName, dtype=np.uint8), -1)
                if self.__outImageRGB.shape == __bgrImg.shape:
                    __rgbImg = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
                    self.__outImageRGB = func(self.__outImageRGB, __rgbImg)
                    self.__drawImage(self.outImageView, self.__outImageRGB)
                    self._saveCurrentStateToHistory()
                else:
                    QMessageBox.information(None, "提示", "图像尺寸不一致，无法进行操作！")

    # 加
    def __addImage(self):
        self.__operation(cv2.add)

    # 减
    def __subtractImage(self):
        self.__operation(cv2.subtract)

    # 乘
    def __multiplyImage(self):
        self.__operation(cv2.multiply)

    # 缩放调节子窗口
    def __openZoomWindow(self):
        self.__openPropertyWindow("缩放", self.__changeZoom)
        self._saveCurrentStateToHistory()

    # 缩放
    def __changeZoom(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if np.size(__img) > 1:
            # 计算比例
            i = int(val)
            if i == -100:
                k = 0.01
            elif i >= 0:
                k = (i + 10) / 10
            else:
                k = (i + 100) / 100
            # 直接cv2.resize()缩放
            self.__tempImageRGB = cv2.resize(__img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)


    # 旋转调节子窗口
    def __openRotateWindow(self):
        self.__openPropertyWindow("旋转", self.__changeRotate)
        if self.__fileName:
            # 重设属性值取值范围
            self.__propertyWindow.slider.setMaximum(360)
            self.__propertyWindow.slider.setMinimum(-360)
            self.__propertyWindow.spinBox.setMaximum(360)
            self.__propertyWindow.spinBox.setMinimum(-360)
        self._saveCurrentStateToHistory()

    # 旋转
    def __changeRotate(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if np.size(__img) > 1:
            # 比例
            k = int(val)
            (h, w) = __img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # 绕图片中心旋转
            m = cv2.getRotationMatrix2D((cX, cY), k, 1.0)
            # 计算调整后的图片显示大小，使得图片不会被切掉边缘
            cos = np.abs(m[0, 0])
            sin = np.abs(m[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            m[0, 2] += (nW / 2) - cX
            m[1, 2] += (nH / 2) - cY
            # 变换，并设置旋转调整后产生的无效区域为白色
            self.__tempImageRGB = __img = cv2.warpAffine(__img, m, (nW, nH), borderValue=(255, 255, 255))
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------直方图均衡-----------------------------------
    # 归一化直方图
    def __histogram(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                # __hist = cv2.calcHist([self.__outImageRGB], [0], None, [256], [0, 256])
                # __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                # plt.plot(__hist)
                # 使用 matplotlib 的绘图功能同时绘制单通道的直方图
                # density的类型是 bool型，指定为True,则为频率直方图，反之为频数直方图
                plt.hist(self.__outImageRGB.ravel(), bins=255, rwidth=0.8, range=(0, 256), density=True)
            # 如果是RGB图
            else:
                color = {"r", "g", "b"}
                # 使用 matplotlib 的绘图功能同时绘制多通道 RGB 的直方图
                for i, col in enumerate(color):
                    __hist = cv2.calcHist([self.__outImageRGB], [i], None, [256], [0, 256])
                    __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                    plt.plot(__hist, color=col)
            # x轴长度区间
            plt.xlim([0, 256])
            # 显示直方图
            plt.show()

    # 直方图均衡化
    def __histogramEqualization(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                self.__outImageRGB = cv2.equalizeHist(self.__outImageRGB)
            # 如果是RGB图
            else:
                # 分解通道，各自均衡化，再合并通道
                (r, g, b) = cv2.split(self.__outImageRGB)
                rh = cv2.equalizeHist(r)
                gh = cv2.equalizeHist(g)
                bh = cv2.equalizeHist(b)
                self.__outImageRGB = cv2.merge((rh, gh, bh))
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # -----------------------------------噪声-----------------------------------
    # 加高斯噪声
    def __addGasussNoise(self):
        if self.__fileName:
            # 图片灰度标准化
            self.__outImageRGB = np.array(self.__outImageRGB / 255, dtype=float)
            # 产生高斯噪声
            noise = np.random.normal(0, 0.001 ** 0.5, self.__outImageRGB.shape)
            # 叠加图片和噪声
            out = cv2.add(self.__outImageRGB, noise)
            # 还原灰度并截取灰度区间
            self.__outImageRGB = np.clip(np.uint8(out * 255), 0, 255)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 加均匀噪声
    def __addUniformNoise(self):
        if self.__fileName:
            # 起始范围
            low = 100
            # 终止范围
            height = 150
            # 搞一个与图片同规模数组
            out = np.zeros(self.__outImageRGB.shape, np.uint8)
            # 噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        # 生成[low，height]的随机值
                        out[i][j] = random.randint(low, height)
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = random.randint(low, height)
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 加脉冲噪声
    def __addImpulseNoise(self):
        if self.__fileName:
            # 搞一个与图片同规模数组
            out = np.zeros(self.__outImageRGB.shape, np.uint8)
            # 椒盐噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        out[i][j] = 0
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = 255
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # -----------------------------------空域滤波-----------------------------------
    # 均值滤波
    def __meanValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.blur(self.__outImageRGB, (5, 5))
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 中值滤波
    def __medianValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.medianBlur(self.__outImageRGB, 5)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # Sobel算子锐化
    def __sobel(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Sobel(self.__outImageRGB, -1, 1, 1, 3)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # Prewitt算子锐化
    def __prewitt(self):
        if self.__fileName:
            # Prewitt 算子
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
            # 通过自定义卷积核实现卷积
            imgx = cv2.filter2D(self.__outImageRGB, -1, kernelx)
            imgy = cv2.filter2D(self.__outImageRGB, -1, kernely)
            # 合并
            self.__outImageRGB = cv2.add(imgx, imgy)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # 拉普拉斯算子锐化
    def __laplacian(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Laplacian(self.__outImageRGB, -1, ksize=3)
            self.__drawImage(self.outImageView, self.__outImageRGB)
            self._saveCurrentStateToHistory()

    # -----------------------------------关于-----------------------------------
    # 关于作者
    def __aboutAuthor(self):
        QMessageBox.information(None, "关于作者", "数字图像处理工具箱V1.0\n\nCopyright © 2025 \n\nGitHub SmallBambooCode")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    show_startup_image("startup.png", duration_ms=3000, main_window=main_window)
    sys.exit(app.exec())
