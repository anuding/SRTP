#include "mymenu.h"  
#include "QMenu"  
#include "QMenuBar"  
#include "QAction"  
#include "QMessageBox"  
#include "QFileDialog"  
#include "QDebug"  
#include "QListWidget"  
/**************************************** 
* Qt中创建菜单和工具栏需要如下步骤： 
* 1. 建立行为Aciton 
* 2. 创建菜单并使它与一个行为关联 
* 3. 创建工具条并使它与一个行为关联 
*****************************************/  
mymenu::mymenu(QWidget *parent):QMainWindow(parent)  
{  
   createAction();  
   createMenu();  
   createContentMenu();  
   this->resize(300,400);  
}  
void mymenu::createAction()  
{  
    //创建打开文件动作  
    fileOpenAction = new QAction(tr("打开文件"),this);  
    //摄者打开文件的快捷方式  
	fileOpenAction->setShortcut(tr("Ctrl+O"));
    //设置打开文件动作提示信息
	fileOpenAction->setStatusTip("open a file");
	//fileOpenAction->setStatusTip("打开一个文件");
    //关联打开文件动作的信号和槽  
    connect(fileOpenAction,SIGNAL(triggered()),this,SLOT(fileOpenActionSlot()));  
}  
void mymenu::createMenu()  
{  
    menu = this->menuBar()->addMenu(tr("文件"));  
    menu->addAction(fileOpenAction);  
}  
  
void mymenu::createContentMenu()  
{  
    this->addAction(fileOpenAction);  
    this->setContextMenuPolicy(Qt::ActionsContextMenu);  
}  
  
void mymenu::fileOpenActionSlot()  
{  
    //QMessageBox::warning(this,tr("提示"),tr("打开文件"),QMessageBox::Yes|QMessageBox::No);  
    selectFile();  
}  
/**************************************** 
* Qt中使用文件选择对话框步骤如下： 
* 1. 定义一个QFileDialog对象 
* 2. 设置路径、过滤器等属性 
*****************************************/  
void mymenu::selectFile()  
{  
    //定义文件对话框类  
    QFileDialog *fileDialog = new QFileDialog(this);  
    //定义文件对话框标题  
    fileDialog->setWindowTitle(tr("打开图片"));  
    //设置默认文件路径  
    fileDialog->setDirectory(".");  
    //设置文件过滤器  
    fileDialog->setNameFilter(tr("Images(*.png *.jpg *.jpeg *.bmp)"));  
    //设置可以选择多个文件,默认为只能选择一个文件QFileDialog::ExistingFiles  
    fileDialog->setFileMode(QFileDialog::ExistingFiles);  
    //设置视图模式  
    fileDialog->setViewMode(QFileDialog::Detail);  
    //打印所有选择的文件的路径  
    if(fileDialog->exec())  
    {  
        fileNames = fileDialog->selectedFiles();  
        showImageList();  
    }  
    for(auto tmp:fileNames)  
        qDebug()<<tmp<<endl;  
}  
/**************************************** 
* Qt中使用文件选择对话框步骤如下： 
* 1. 定义一个QListWidget对象 
* 2. 设置ViewMode等属性 
* 3. 定义单元项并添加到QListWidget中 
* 4. 调用QListWidget对象的show()方法 
*****************************************/  
void mymenu::showImageList()  
{  
    //定义QListWidget对象  
    QListWidget *imageList = new QListWidget;  
    imageList->resize(365,400);  
    //设置QListWidget的显示模式  
    imageList->setViewMode(QListView::IconMode);  
    //设置QListWidget中单元项的图片大小  
    imageList->setIconSize(QSize(100,100));  
    //设置QListWidget中单元项的间距  
    imageList->setSpacing(10);  
    //设置自动适应布局调整（Adjust适应，Fixed不适应），默认不适应  
    imageList->setResizeMode(QListWidget::Adjust);  
    //设置不能移动  
    imageList->setMovement(QListWidget::Static);  
    for(auto tmp : fileNames)  
    {  
        //定义QListWidgetItem对象  
        QListWidgetItem *imageItem = new QListWidgetItem;  
        //为单元项设置属性  
        imageItem->setIcon(QIcon(tmp));  
        //imageItem->setText(tr("Browse"));  
        //重新设置单元项图片的宽度和高度  
        imageItem->setSizeHint(QSize(100,120));  
        //将单元项添加到QListWidget中  
        imageList->addItem(imageItem);  
    }  
    //显示QListWidget  
    imageList->show();  
}  