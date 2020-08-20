import streamlit as st
import pandas as pd

# importing dataset
df = pd.read_csv('breast cancer dataset.csv')

st.markdown('<style>body{background-color: #B7E2A6;}</style>',unsafe_allow_html=True)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#F1F9EE,#F1F9EE);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


# data preprocessing
df = df[['radius_mean', 'perimeter_mean', 'area_mean', 'concave points_mean', 'radius_worst',
         'perimeter_worst', 'area_worst', 'concave points_worst', 'diagnosis']]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])



# ui
html_temp = """
<div style="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;font-size:45px">Breast Cancer Detection </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("   ")
st.write("  ")

if st.checkbox("Show dataset"):
    df

i = st.slider("Enter test size :", 0.1, 0.5)
j = st.selectbox(" Enter the algorithm :", ('SVM', 'KNN', 'Random Forest'))
st.markdown("    ")
st.markdown("    ")
st.markdown("    ")



a="""
<style>color: blue;hello</style>
"""



# input
st.sidebar.header('Features')

st.markdown(a,unsafe_allow_html=True)
a = st.sidebar.slider("Enter radius_mean :", float(df['radius_mean'].min()), float(df['radius_mean'].max()))
b = st.sidebar.slider("Enter perimeter_mean :", float(df['perimeter_mean'].min()), float(df['perimeter_mean'].max()))
c = st.sidebar.slider("Enter area_mean :", float(df['area_mean'].min()), float(df['area_mean'].max()))
d = st.sidebar.slider("Enter concave points_mean :", float(df['concave points_mean'].min()),
                      float(df['concave points_mean'].max()))
e = st.sidebar.slider("Enter radius_worst :", float(df['radius_worst'].min()), float(df['radius_worst'].max()))
f = st.sidebar.slider("Enter perimeter_worst :", float(df['perimeter_worst'].min()), float(df['perimeter_worst'].max()))
g = st.sidebar.slider("Enter area_worst :", float(df['area_worst'].min()), float(df['area_worst'].max()))
h = st.sidebar.slider("Enter concave points_worst :", float(df['concave points_worst'].min()),
                      float(df['concave points_worst'].max()))






# model
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  #### It will take mean as 1 and S.d as O
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

if j == "SVM":
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

elif j == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=9)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

else:
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=20)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)






# prediction and display
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
prediction = classifier.predict([[a, b, c, d, e, f, g, h]])


safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;font-size:50px"> You doont have symptoms of cancer (begin)</h2>
       </div>
    """
danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;font-size:300px"> You have symptoms of cancer (miligan)</h2>
       </div>
    """


if st.button('Run the model'):
    if prediction == 0:
        st.markdown(safe_html,unsafe_allow_html=True)
    else:
        st.markdown(danger_html,unsafe_allow_html=True)

    st.write(" ")
    st.write("Accuracy is :", acc)
