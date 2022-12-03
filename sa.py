from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression

ws = Tk()
ws.geometry('1920x1080')
ws.title('Metric for Number of PhD Students Graduated')
ws['bg']='deep sky blue'
f = ("Georgia", 16)
 
def basic():
    ws.destroy()
    import page1basic

def go():
    ws.destroy()
    import page2go

def add():
    t3.delete(0, 'end')
    gphd=pd.read_csv("C:/Trisha/Abhyas/Project-NIRF/Final Project - Final Dataset with calculations for 100 colleges.csv")
    #Define X and Y and reshape the arrays
    X_phd=gphd['Average of PhD graduating students'].values.reshape(-1,1)
    Y_phd=gphd['GPHD_graph value'].values.reshape(-1,1)

    """POLYNOMIAL REGRESSION-GPHD"""

    #Import Polynomial Regression Features
    from sklearn.preprocessing import PolynomialFeatures
    phd_polynomial_reg = PolynomialFeatures(degree=2)

    #Define X and Y and apply polyfit function
    Xphd_poly = phd_polynomial_reg.fit_transform(X_phd)
    Yphd_poly = phd_polynomial_reg.fit_transform(Y_phd)

    #Split the dataset into train and testing data (80:20)
    from sklearn.model_selection import train_test_split
    #print(X.shape)
    Xphd_train, Xphd_test, Yphd_train, Yphd_test = train_test_split(Xphd_poly, Yphd_poly, 
                                                            test_size = 0.2, random_state
    =0)
    #print(Xphd_train.shape)
    #print(Xphd_test.shape)
    #print(Yphd_train.shape)
    #print(Yphd_test.shape)

    #Data Fitting
    phd_lin_reg = LinearRegression()
    phd_lin_reg.fit(Xphd_train,Yphd_train)

    #Perform predictions
    Yphd_pred_poly = phd_lin_reg.predict(Xphd_test) 
    #print(Yphd_pred_poly)#.flatten())

    #Find Accuracy
    #print('Mean Squared Error:', metrics.mean_squared_error(Yphd_test, Yphd_pred_poly))
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Yphd_test,Yphd_pred_poly)))
    #accuracy=phd_lin_reg.score(Xphd_test,Yphd_test)
    #print("The accuracy using Polynomial Regression is",accuracy)

    #df = pd.DataFrame({'Actual': Yphd_test.flatten(), 'Predicted': Yphd_pred_poly.flatten()})
    #print(df)
    
    #prediction = model.model(image).flatten()
    
    num1=int(txtfld_1.get())
    num2=int(txtfld_2.get())
    num3=int(txtfld_3.get())
    num4=int(txtfld_4.get())
    num5=int(txtfld_5.get())
    num6=int(txtfld_6.get())

    x = [num1, num2, num3, num4, num5, num6]
    y = phd_lin_reg.predict(x)
    #result1=num1+num2
    #result2=num3+num4
    #result3=num5+num6
    #result=(result1+result2+result3)/3
    t3.insert(END, str(y))

btn=Button(ws, command=basic, text="Go Back to the Main Page", fg='black', height=3, font=f)
btn.place(x=400, y=500)

btn1=Button(ws, command=go, text=" Go to Graduation Outcome", fg='black', height=3, font=f)
btn1.place(x=800, y=500)

btn2=Button(ws, command=add, text="Calculate", fg='Black', bg='deep sky blue', font=("Georgia", 12))
btn2.place(x=550, y=400)

lbl=Label(ws, text="PHD Graduating FT 18-19", fg='black', bg='deep sky blue', font=("Georgia", 12))
lbl.place(x=280, y=110)

txtfld_1=Entry(ws, text="FT 18-19", bd=5)
txtfld_1.place(x=280, y=130)

lb2=Label(ws, text="PHD Graduating PT 18-19", fg='black', bg='deep sky blue', font=("Georgia", 12))
lb2.place(x=620, y=110)

txtfld_2=Entry(ws, text="PT 18-19", bd=5)
txtfld_2.place(x=620, y=130)

lb3=Label(ws, text="PHD Graduating FT 17-18",fg='Black', bg='deep sky blue', font=("Georgia", 12))
lb3.place(x=1000, y=110)

txtfld_3=Entry(ws, text="FT 17-18", bd=5)
txtfld_3.place(x=1000, y=130)

lb4=Label(ws, text="PHD Graduating PT 17-18", fg='black', bg='deep sky blue', font=("Georgia", 12))
lb4.place(x=280, y=230)

txtfld_4=Entry(ws, text="PT 17-18", bd=5)
txtfld_4.place(x=280, y=250)

lb5=Label(ws, text="PHD Graduating FT 16-17", fg='black', bg='deep sky blue', font=("Georgia", 12))
lb5.place(x=620, y=230)

txtfld_5=Entry(ws, text="FT 16-17", bd=5)
txtfld_5.place(x=620, y=250)

lb6=Label(ws, text="PHD Graduating PT 16-17", fg='black', bg='deep sky blue', font=("Georgia", 12))
lb6.place(x=1000, y=230)

txtfld_6=Entry(ws, text="PT 16-17", bd=5)
txtfld_6.place(x=1000, y=250)

lb7=Label(ws, text="Metric for Number of PhD Students Graduated", bg='deep sky blue', font=("Georgia",18))
lb7.place(x=470, y=0)

t3=Entry(ws, text="Widget", bd=5)
t3.place(x=700, y=400)

ws.mainloop()