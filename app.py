from flask import Flask,render_template,request,jsonify
import pandas as pd
import pickle

app = Flask(__name__)
#########

@app.route("/chart0")
def overview():
    return render_template('page1.html')


@app.route("/chart1")
def chart1():
    return render_template('page2-1.html')

    
@app.route("/chart2")
def chart2():
    AGE = request.args.get('_AGEG5YR')
    EDU = request.args.get('_EDUCAG')
    INC = request.args.get('INCOME2')
    _PA = request.args.get('_PAREC2')
    DRN = request.args.get('_DRNKWK1')
    PER = request.args.get('PERSDOC2')
    SMO = request.args.get('_SMOKER3')

    datatat=[]
    if AGE != None and AGE != 'None' :
        dataY10 = {'_AGEG5YR':AGE, 'INCOME2':INC, '_DRNKWK1':DRN, '_PAREC2':_PA, '_EDUCAG':EDU}
        dataY21 = {'_AGEG5YR':AGE, '_EDUCAG':EDU, 'INCOME2':INC, '_DRNKWK1':DRN, 'PERSDOC2':PER}
        dataY22 = {'_AGEG5YR':AGE, '_EDUCAG':EDU, 'INCOME2':INC, '_SMOKER3':SMO, '_DRNKWK1':DRN}

        query_dfY10 = pd.DataFrame([dataY10])
        print(query_dfY10)
        modelY10 = pickle.load(open('CatBoost_Y1.sav', 'rb'))
        Y10=modelY10.predict(query_dfY10)
        print('Y10預測:',Y10[0])
        #################################
        query_dfY21 = pd.DataFrame([dataY21])
        print(query_dfY21)
        modelY21 = pickle.load(open('CatBoost_Y21.sav', 'rb'))
        Y21=modelY21.predict(query_dfY21)
        print('Y21預測:',Y21[0])
        #################################
        query_dfY22 = pd.DataFrame([dataY22])
        print(query_dfY22)
        modelY22 = pickle.load(open('ExtraTrees_model.sav', 'rb'))
        Y22=modelY22.predict(query_dfY22)
        print(type(Y22[0]))
        print('Y22預測:',Y22[0])
        #################################
        datatat = [AGE,INC,DRN,_PA,EDU,str(Y10[0]),PER,str(Y21[0]),SMO,str(Y22[0])]
        #          0  , 1 , 2 , 3 , 4 ,     5     , 6 ,      7    ,8  ,  9
        print(datatat)
    return render_template('page2-11.html',args =datatat)


@app.route("/chart3")
def chart3():
    return render_template('page3.html')

#########
app.run(port=5000,debug=True)