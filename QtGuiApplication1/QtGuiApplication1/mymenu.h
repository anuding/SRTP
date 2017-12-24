#ifndef MYMENU_H  
#define MYMENU_H  
  
#include "QMainWindow"  
class mymenu : public QMainWindow  
{  
    Q_OBJECT  // must include this if you use Qt signals/slots  
public:  
    //构造函数  
    explicit mymenu(QWidget *parent = 0);//QWidget *parent = 0 表示mymenu控件不是任何控件的子控件  
  
private slots:  
    void fileOpenActionSlot();//打开文件动作对应的槽函数  
private:  
    void createAction();     //创建动作  
    void createMenu();       //创建菜单  
    void createContentMenu();//创建上下文菜单  
private:  
    void selectFile();       //弹出选择文件对话框  
    void showImageList();    //用缩略图显示图片  
private:  
    QAction *fileOpenAction; //创建一个QAction指针，打开文件动作  
    QMenu *menu;             //创建一个QMenu指针  
private:  
    QStringList fileNames;  
};  
  
#endif // MYMENU_H  