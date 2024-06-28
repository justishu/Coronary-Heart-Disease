# Coronary-Heart-Disease
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df=pd.read_csv("C:\\Users\\manis\\Downloads\\framingham.csv")
#education column not needed in CHD
df.drop(columns=['education'],inplace=True)
#replace all missing value by its mdeian
import numpy as np
df.replace(np.nan,df.median(),inplace=True)
#drop all duplicates
df.drop_duplicates(inplace=True)

df.drop(columns=['diaBP','cigsPerDay'],inplace=True)

#separate x and y
x=df.drop(columns=['TenYearCHD'])
y=df['TenYearCHD']

#split data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
############################################################################
#DEFINE FUNCTIONS
def knn():
    #KNN
    #create model knn
    from sklearn.neighbors import KNeighborsClassifier
    k=KNeighborsClassifier(n_neighbors=5)
    #train the model
    k.fit(x_train,y_train)
    #test model
    y_pred_knn=k.predict(x_test)
    #find accuracy
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(y_test,y_pred_knn)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title='KNN',message="Accuracy is "+str(acc_knn))
def dt():
    #apply decision tree model
    from sklearn.tree import DecisionTreeClassifier
    d=DecisionTreeClassifier()
    d.fit(x_train,y_train)
    #test
    y_pred_dt=d.predict(x_test)
    #find accuracy
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(y_test,y_pred_dt)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title='DT',message="Accuracy is "+str(acc_dt))
def lg():
    #LogicalRegression
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression(solver='liblinear')
    #Train the model
    L.fit(x_train,y_train)
    #Test the model
    y_pred_lg=L.predict(x_test)
    #find accuracy
    from sklearn.metrics import accuracy_score
    acc_lg=accuracy_score(y_test,y_pred_lg)
    acc_lg=round(acc_lg*100,2)
    m.showinfo(title='LG',message="Accuracy is "+str(acc_lg))
def nb():
    #naive bayes
    from sklearn.naive_bayes import GaussianNB
    N=GaussianNB()
    #Train model
    N.fit(x_train,y_train)
    #test model
    y_pred_nb=N.predict(x_test)
    #find accuracy
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(y_test,y_pred_nb)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title='NB',message="Accuracy is "+str(acc_nb))
def svm():
    #create SVM model
    from sklearn.svm import SVC #support vector classifier SVC #support vector regressor SVR
    model=SVC(C=10,kernel='rbf')
    model.fit(x_train,y_train)
    y_pred_svm=model.predict(x_test)
    #find accuracy
    from sklearn.metrics import accuracy_score
    acc_svm=accuracy_score(y_test,y_pred_svm)
    acc_svm=round(acc_svm*100,2)
    m.showinfo(title='SVM',message="Accuracy is "+str(acc_svm))
def compare():
    
    #compare all the accuracies
    import matplotlib.pyplot as plt
    model=['KNN','LG','DT','NB','SVM']
    acc=[acc_knn,acc_lg,acc_dt,acc_nb,acc_svm]
    plt.bar(model,acc,color=['blue','green','red','yellow','orange'])
    plt.show()
def submit():
    male=float(emale.get())
    age=float(eage.get())
    smoker=float(esmoker.get())
    meds=float(emeds.get())
    stroke=float(estroke.get())
    hyp=float(ehyp.get())
    diabetes=float(ediabetes.get())
    chol=float(echol.get())
    sysbp=float(esysbp.get())
    bmi=float(ebmi.get())
    rate=float(erate.get())
    glu=float(eglu.get())
    result=L.predict([[male,age,smoker,meds,stroke,hyp,diabetes,chol,sysbp,bmi,rate,glu]])
    if result[0]==1:
        msg='You may have heart prblm'
    else:
        msg='You dnt have any heart prblm'
    m.showinfo(title='HEART',message=msg)
def reset():
    emale.delete(0,END)
    eage.delete(0,END)
    esmoker.delete(0,END)
    emeds.delete(0,END)
    estroke.delete(0,END)
    ehyp.delete(0,END)
    ediabetes.delete(0,END)
    echol.delete(0,END)
    esysbp.delete(0,END)
    ebmi.delete(0,END)
    erate.delete(0,END)
    eglu.delete(0,END)
    
####################################################################################################
###################################################################################################
##################################################################################################


from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.configure(bg='white')
L1=Label(w,text='HEART DISEASE PREDICTION',font=('Baskerville Old Face',40,'bold'),bg='Maroon',fg='white')
L1.grid(row=1,column=1,columnspan=5)

#2nd row
bknn=Button(w,text='KNN Model',command=knn,font=('Courier New',15,'bold'),bg='Pink')
blg=Button(w,text='Logical Regression Model',command=lg,font=('Courier New',15,'bold'),bg='BlanchedAlmond')
bdt=Button(w,text='Decision Tree Model',command=dt,font=('Courier New',15,'bold'),bg='YellowGreen')

bknn.grid(row=2,pady=20,column=1,columnspan=2)
blg.grid(row=2,pady=20,column=2,columnspan=5)

bdt.grid(row=3,pady=20,column=1,columnspan=2)


#row 3
bnb=Button(w,text='Naive Bayes Model',command=nb,font=('Courier New',15,'bold'),bg='SkyBlue')
bsvm=Button(w,text='SVM',command=svm,font=('Courier New',15,'bold'),bg='Orchid')

bnb.grid(row=3,pady=20,column=2,columnspan=3)
bsvm.grid(row=3,pady=20,column=4,columnspan=3)

#row=4
bcmp=Button(w,text='COMPARE',command=compare,font=('Times New Roman',14,'bold'),bg='black',fg='white')
bcmp.grid(row=6,pady=20,column=1,columnspan=5)

#row=5
L2=Label(w,text='Enter data:',font=('Arial',13,'bold'))
L2.grid(row=8,column=1)
#row 6
lmale=Label(w,text='sex',font=('Tahoma',10,'bold'))
lmale.grid(row=10,pady=15,column=1)
emale=Entry(w,relief='solid')
emale.grid(row=10,pady=15,column=2)
#row 7
lage=Label(w,text='Age',font=('Tahoma',10,'bold'))
lage.grid(row=10,pady=15,column=3)
eage=Entry(w,relief='solid')
eage.grid(row=10,pady=15,column=4)
#row8
lsmoker=Label(w,text='currentSmoker',font=('Tahoma',10,'bold'))
lsmoker.grid(row=11,pady=15,column=1)
esmoker=Entry(w,relief='solid')
esmoker.grid(row=11,pady=15,column=2)
#row9
lmeds=Label(w,text='BPMeds',font=('Tahoma',10,'bold'))
lmeds.grid(row=11,pady=15,column=3)
emeds=Entry(w,relief='solid')
emeds.grid(row=11,pady=15,column=4)
#row10
lstroke=Label(w,text='prevalentStroke',font=('Tahoma',10,'bold'))
lstroke.grid(row=12,pady=15,column=1)
estroke=Entry(w,relief='solid')
estroke.grid(row=12,pady=15,column=2)
#row11
lhyp=Label(w,text='prevalentHyp',font=('Tahoma',10,'bold'))
lhyp.grid(row=12,pady=15,column=3)
ehyp=Entry(w,relief='solid')
ehyp.grid(row=12,pady=15,column=4)
#row12
ldiabetes=Label(w,text='diabetes',font=('Tahoma',10,'bold'))
ldiabetes.grid(row=13,pady=15,column=1)
ediabetes=Entry(w,relief='solid')
ediabetes.grid(row=13,pady=15,column=2)
#row13
lchol=Label(w,text='totChol',font=('Tahoma',10,'bold'))
lchol.grid(row=13,pady=15,column=1)
echol=Entry(w,relief='solid')
echol.grid(row=13,column=2,pady=15)
#row14
lsysbp=Label(w,text='sysBP',font=('Tahoma',10,'bold'))
lsysbp.grid(row=13,column=3,pady=15)
esysbp=Entry(w,relief='solid')
esysbp.grid(row=13,column=4,pady=15)
#row15
lbmi=Label(w,text='BMI',font=('Tahoma',10,'bold'))
lbmi.grid(row=14,column=1,pady=15)
ebmi=Entry(w,relief='solid')
ebmi.grid(row=14,column=2,pady=15)
#row 16
lrate=Label(w,text='heartRate',font=('Tahoma',10,'bold'))
lrate.grid(row=14,column=3,pady=15)
erate=Entry(w,relief='solid')
erate.grid(row=14,column=4,pady=15)
#row17
lglu=Label(w,text='glucose',font=('Tahoma',10,'bold'))
lglu.grid(row=15,column=1,pady=15)
eglu=Entry(w,relief='solid')
eglu.grid(row=15,column=2,pady=15)
#row18
bsub=Button(w,text='Submit',command=submit,font=('Arial',10,'bold'),bg='MintCream',relief='solid')
bsub.grid(row=20,pady=15,column=1,columnspan=2)
bres=Button(w,text='Reset',command=reset,font=('Arial',10,'bold'),bg='MintCream',relief='solid')
bres.grid(row=20,pady=15,column=2,columnspan=4)
w.mainloop()
