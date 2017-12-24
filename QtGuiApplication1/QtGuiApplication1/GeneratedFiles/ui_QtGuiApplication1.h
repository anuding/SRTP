/********************************************************************************
** Form generated from reading UI file 'QtGuiApplication1.ui'
**
** Created by: Qt User Interface Compiler version 5.9.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTGUIAPPLICATION1_H
#define UI_QTGUIAPPLICATION1_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtGuiApplication1Class
{
public:
    QWidget *centralWidget;
    QPushButton *Btn_Import;
    QTextEdit *Edt_VideoAddress;
    QLabel *Lab_VideoSummary;
    QListWidget *Lst_KeyFrames;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtGuiApplication1Class)
    {
        if (QtGuiApplication1Class->objectName().isEmpty())
            QtGuiApplication1Class->setObjectName(QStringLiteral("QtGuiApplication1Class"));
        QtGuiApplication1Class->resize(1004, 635);
        centralWidget = new QWidget(QtGuiApplication1Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        Btn_Import = new QPushButton(centralWidget);
        Btn_Import->setObjectName(QStringLiteral("Btn_Import"));
        Btn_Import->setGeometry(QRect(910, 0, 93, 28));
        Edt_VideoAddress = new QTextEdit(centralWidget);
        Edt_VideoAddress->setObjectName(QStringLiteral("Edt_VideoAddress"));
        Edt_VideoAddress->setGeometry(QRect(20, 0, 861, 31));
        Lab_VideoSummary = new QLabel(centralWidget);
        Lab_VideoSummary->setObjectName(QStringLiteral("Lab_VideoSummary"));
        Lab_VideoSummary->setGeometry(QRect(20, 49, 391, 501));
        QFont font;
        font.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        font.setPointSize(22);
        Lab_VideoSummary->setFont(font);
        Lab_VideoSummary->setAlignment(Qt::AlignCenter);
        Lst_KeyFrames = new QListWidget(centralWidget);
        Lst_KeyFrames->setObjectName(QStringLiteral("Lst_KeyFrames"));
        Lst_KeyFrames->setGeometry(QRect(480, 50, 491, 511));
        QtGuiApplication1Class->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtGuiApplication1Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1004, 26));
        QtGuiApplication1Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtGuiApplication1Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        QtGuiApplication1Class->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(QtGuiApplication1Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        QtGuiApplication1Class->setStatusBar(statusBar);

        retranslateUi(QtGuiApplication1Class);
        QObject::connect(Btn_Import, SIGNAL(clicked()), QtGuiApplication1Class, SLOT(myExitButtonFuc()));

        QMetaObject::connectSlotsByName(QtGuiApplication1Class);
    } // setupUi

    void retranslateUi(QMainWindow *QtGuiApplication1Class)
    {
        QtGuiApplication1Class->setWindowTitle(QApplication::translate("QtGuiApplication1Class", "QtGuiApplication1", Q_NULLPTR));
        Btn_Import->setText(QApplication::translate("QtGuiApplication1Class", "\345\257\274\345\205\245", Q_NULLPTR));
        Lab_VideoSummary->setText(QApplication::translate("QtGuiApplication1Class", "\350\257\267\345\257\274\345\205\245\344\270\200\346\256\265\350\247\206\351\242\221", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class QtGuiApplication1Class: public Ui_QtGuiApplication1Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTGUIAPPLICATION1_H
